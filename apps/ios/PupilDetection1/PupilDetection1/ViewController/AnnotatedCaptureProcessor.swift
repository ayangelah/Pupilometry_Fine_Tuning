import AVFoundation
import CoreImage
import Foundation
import UIKit

struct VideoAnnotationSample {
    let videoTimeS: Double
    let eye: String
    let accepted: Bool
    let normalizedROI: CGRect
    let iris: LiveCircle?
    let pupil: LiveCircle?

    var pupilDiameterMM: Double? {
        guard accepted,
              let iris,
              let pupil,
              iris.radius > 0,
              pupil.radius > 0 else { return nil }
        return pupil.radius / iris.radius * MeasurementCalculator.irisReferenceMM
    }
}

struct EyeMeasurementSummary: Codable {
    let eye: String
    let medianDiameterMM: Double?
    let baselineMM: Double?
    let minimumMM: Double?
    let percentConstriction: Double?
    let latencyS: Double?
    let meanConstrictionVelocityMMPerS: Double?
    let validSampleCount: Int
}

struct AnnotatedCaptureResult {
    let annotatedVideoURL: URL
    let firstFrame: UIImage
    let lastFrame: UIImage
    let measurements: [EyeMeasurementSummary]
}

enum AnnotatedCaptureProcessor {
    private static let exportQueue = DispatchQueue(
        label: "com.visualyze.annotation-export",
        qos: .utility
    )

    static func process(
        sourceURL: URL,
        outputURL: URL,
        samples: [VideoAnnotationSample],
        lightOnsetVideoTimeS: Double?,
        completion: @escaping (Result<AnnotatedCaptureResult, Error>) -> Void
    ) {
        exportQueue.async {
            do {
                try? FileManager.default.removeItem(at: outputURL)
                let measurements = MeasurementCalculator.summaries(
                    samples: samples,
                    lightOnsetVideoTimeS: lightOnsetVideoTimeS
                )
                let renderer = AnnotationRenderer(
                    samples: samples,
                    lightOnsetVideoTimeS: lightOnsetVideoTimeS
                )
                let asset = AVURLAsset(url: sourceURL)
                let composition = AVVideoComposition(asset: asset) { request in
                    let timeS = CMTimeGetSeconds(request.compositionTime)
                    let annotated = renderer.render(
                        source: request.sourceImage,
                        videoTimeS: timeS
                    )
                    request.finish(with: annotated, context: nil)
                }

                guard let exporter = AVAssetExportSession(
                    asset: asset,
                    presetName: AVAssetExportPresetHEVCHighestQuality
                ) else {
                    throw AnnotationError.cannotCreateExporter
                }
                exporter.outputURL = outputURL
                exporter.outputFileType = exporter.supportedFileTypes.contains(.mov) ? .mov : .mp4
                exporter.shouldOptimizeForNetworkUse = false
                exporter.videoComposition = composition
                exporter.exportAsynchronously {
                    exportQueue.async {
                        switch exporter.status {
                        case .completed:
                            Task {
                                do {
                                    let frames = try await makeEndpointFrames(from: outputURL)
                                    completion(.success(AnnotatedCaptureResult(
                                        annotatedVideoURL: outputURL,
                                        firstFrame: frames.first,
                                        lastFrame: frames.last,
                                        measurements: measurements
                                    )))
                                } catch {
                                    completion(.failure(error))
                                }
                            }
                        case .failed, .cancelled:
                            completion(.failure(exporter.error ?? AnnotationError.exportFailed))
                        default:
                            completion(.failure(AnnotationError.exportFailed))
                        }
                    }
                }
            } catch {
                completion(.failure(error))
            }
        }
    }

    private static func makeEndpointFrames(
        from url: URL
    ) async throws -> (first: UIImage, last: UIImage) {
        let asset = AVURLAsset(url: url)
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = CMTime(value: 1, timescale: 15)
        let first = try generator.copyCGImage(at: .zero, actualTime: nil)
        let duration = CMTimeGetSeconds(try await asset.load(.duration))
        let lastTime = CMTime(seconds: max(0, duration - 0.05), preferredTimescale: 600)
        let last = try generator.copyCGImage(at: lastTime, actualTime: nil)
        return (UIImage(cgImage: first), UIImage(cgImage: last))
    }

    private enum AnnotationError: LocalizedError {
        case cannotCreateExporter
        case exportFailed

        var errorDescription: String? {
            switch self {
            case .cannotCreateExporter: return "Could not create the annotated-video exporter."
            case .exportFailed: return "Annotated-video export did not complete."
            }
        }
    }
}

enum MeasurementCalculator {
    static let irisReferenceMM = 11.7
    private static let baselineStartS = 0.030
    private static let baselineEndS = 0.150
    private static let minimumLatencyS = 0.180

    static func summaries(
        samples: [VideoAnnotationSample],
        lightOnsetVideoTimeS: Double?
    ) -> [EyeMeasurementSummary] {
        let eyes = Array(Set(samples.map(\.eye))).sorted(by: eyeSort)
        return eyes.map { eye in
            summary(
                eye: eye,
                points: samples
                    .filter { $0.eye == eye }
                    .compactMap { sample in
                        sample.pupilDiameterMM.map { (sample.videoTimeS, $0) }
                    }
                    .sorted { $0.0 < $1.0 },
                onset: lightOnsetVideoTimeS
            )
        }
    }

    static func liveText(
        eye: String,
        points: [(time: Double, diameterMM: Double)],
        lightOnsetVideoTimeS: Double?
    ) -> String {
        guard let latest = points.last else { return "\(eye): --" }
        guard let onset = lightOnsetVideoTimeS else {
            return String(format: "%@: %.2f mm", eye, latest.diameterMM)
        }
        let result = summary(eye: eye, points: points, onset: onset)
        var pieces = [String(format: "%@: %.2f mm", eye, latest.diameterMM)]
        if let percent = result.percentConstriction {
            pieces.append(String(format: "%.0f%%", max(0, percent)))
        }
        if let velocity = result.meanConstrictionVelocityMMPerS {
            pieces.append(String(format: "%.2f mm/s", velocity))
        }
        return pieces.joined(separator: "  ·  ")
    }

    private static func summary(
        eye: String,
        points: [(time: Double, diameterMM: Double)],
        onset: Double?
    ) -> EyeMeasurementSummary {
        guard let onset else {
            return EyeMeasurementSummary(
                eye: eye,
                medianDiameterMM: points.isEmpty ? nil : finiteOrNil(median(points.map(\.diameterMM))),
                baselineMM: nil,
                minimumMM: nil,
                percentConstriction: nil,
                latencyS: nil,
                meanConstrictionVelocityMMPerS: nil,
                validSampleCount: points.count
            )
        }

        let baselineValues = points.compactMap { point -> Double? in
            let relative = point.time - onset
            return (relative >= baselineStartS && relative <= baselineEndS)
                ? point.diameterMM : nil
        }
        guard baselineValues.count >= 3 else {
            return EyeMeasurementSummary(
                eye: eye,
                medianDiameterMM: points.isEmpty ? nil : finiteOrNil(median(points.map(\.diameterMM))),
                baselineMM: nil,
                minimumMM: nil,
                percentConstriction: nil,
                latencyS: nil,
                meanConstrictionVelocityMMPerS: nil,
                validSampleCount: points.count
            )
        }

        let baseline = median(baselineValues)
        let postLatency = points.filter { $0.time >= onset + minimumLatencyS }
        let smoothed = rollingMedian(postLatency)
        let minimum = smoothed.map(\.diameterMM).min()
        let percent = minimum.map { (baseline - $0) / baseline * 100 }
        let latency = points.first {
            $0.time >= onset && (baseline - $0.diameterMM) / baseline >= 0.05
        }.map { $0.time - onset }

        var constrictionSlopes: [Double] = []
        if let minimum,
           let end = smoothed.firstIndex(where: { $0.diameterMM <= minimum + 0.05 * max(0, baseline - minimum) }) {
            let active = Array(smoothed.prefix(end + 1)).filter { $0.diameterMM <= baseline * 0.98 }
            if active.count >= 2 {
                for pair in zip(active, active.dropFirst()) {
                    let dt = pair.1.time - pair.0.time
                    let velocity = (pair.0.diameterMM - pair.1.diameterMM) / dt
                    if dt > 0, velocity.isFinite, velocity > 0 { constrictionSlopes.append(velocity) }
                }
            }
        }

        return EyeMeasurementSummary(
            eye: eye,
            medianDiameterMM: points.isEmpty ? nil : finiteOrNil(median(points.map(\.diameterMM))),
            baselineMM: finiteOrNil(baseline),
            minimumMM: minimum.flatMap(finiteOrNil),
            percentConstriction: percent.flatMap(finiteOrNil),
            latencyS: latency.flatMap(finiteOrNil),
            meanConstrictionVelocityMMPerS: constrictionSlopes.isEmpty
                ? nil : finiteOrNil(constrictionSlopes.reduce(0, +) / Double(constrictionSlopes.count)),
            validSampleCount: points.count
        )
    }

    private static func rollingMedian(
        _ points: [(time: Double, diameterMM: Double)]
    ) -> [(time: Double, diameterMM: Double)] {
        points.indices.map { index in
            let lo = max(points.startIndex, index - 2)
            let hi = min(points.endIndex, index + 3)
            return (points[index].time, median(points[lo..<hi].map(\.diameterMM)))
        }
    }

    private static func median(_ values: [Double]) -> Double {
        let sorted = values.sorted()
        guard !sorted.isEmpty else { return .nan }
        let middle = sorted.count / 2
        return sorted.count.isMultiple(of: 2)
            ? (sorted[middle - 1] + sorted[middle]) / 2
            : sorted[middle]
    }

    private static func finiteOrNil(_ value: Double) -> Double? {
        value.isFinite ? value : nil
    }

    private static func eyeSort(_ lhs: String, _ rhs: String) -> Bool {
        let order = ["Right": 0, "Left": 1, "Eye": 2]
        return order[lhs, default: 3] < order[rhs, default: 3]
    }
}

private final class AnnotationRenderer {
    private let samplesByEye: [String: [VideoAnnotationSample]]
    private let lightOnsetVideoTimeS: Double?
    private let lock = NSLock()
    private var imageCache: [String: CIImage] = [:]

    init(samples: [VideoAnnotationSample], lightOnsetVideoTimeS: Double?) {
        self.samplesByEye = Dictionary(grouping: samples, by: \.eye)
            .mapValues { $0.sorted { $0.videoTimeS < $1.videoTimeS } }
        self.lightOnsetVideoTimeS = lightOnsetVideoTimeS
    }

    func render(source: CIImage, videoTimeS: Double) -> CIImage {
        let extent = source.extent
        var output = source
        for eye in samplesByEye.keys.sorted() {
            guard let samples = samplesByEye[eye],
                  let (index, sample) = nearestSample(in: samples, at: videoTimeS),
                  abs(sample.videoTimeS - videoTimeS) <= 0.20 else { continue }
            let key = "\(eye)-\(index)-\(Int(extent.width))x\(Int(extent.height))"
            let overlay = cachedImage(for: key) {
                makeEyeOverlay(sample: sample, samples: samples, extent: extent)
            }
            output = overlay.composited(over: output)
        }
        return output.cropped(to: extent)
    }

    private func nearestSample(
        in samples: [VideoAnnotationSample],
        at timeS: Double
    ) -> (Int, VideoAnnotationSample)? {
        guard !samples.isEmpty else { return nil }
        var low = 0, high = samples.count
        while low < high {
            let middle = (low + high) / 2
            if samples[middle].videoTimeS <= timeS { low = middle + 1 } else { high = middle }
        }
        let candidates = [max(0, low - 1), min(samples.count - 1, low)]
        guard let index = candidates.min(by: {
            abs(samples[$0].videoTimeS - timeS) < abs(samples[$1].videoTimeS - timeS)
        }) else { return nil }
        return (index, samples[index])
    }

    private func cachedImage(for key: String, make: () -> CIImage) -> CIImage {
        lock.lock()
        defer { lock.unlock() }
        if let existing = imageCache[key] { return existing }
        let created = make()
        imageCache[key] = created
        return created
    }

    private func makeEyeOverlay(
        sample: VideoAnnotationSample,
        samples: [VideoAnnotationSample],
        extent: CGRect
    ) -> CIImage {
        var result = CIImage.empty()
        if let iris = sample.iris {
            result = ellipse(
                circle: iris,
                roi: sample.normalizedROI,
                extent: extent,
                color: .systemBlue,
                lineWidth: max(4, extent.width / 500)
            ).composited(over: result)
        }
        if let pupil = sample.pupil {
            result = ellipse(
                circle: pupil,
                roi: sample.normalizedROI,
                extent: extent,
                color: sample.accepted ? .systemGreen : .systemOrange,
                lineWidth: max(4, extent.width / 500)
            ).composited(over: result)
        }

        let points = samples
            .filter { $0.videoTimeS <= sample.videoTimeS }
            .compactMap { item in
                item.pupilDiameterMM.map { (time: item.videoTimeS, diameterMM: $0) }
            }
        let text = MeasurementCalculator.liveText(
            eye: sample.eye,
            points: points,
            lightOnsetVideoTimeS: lightOnsetVideoTimeS
        )
        let fontSize = max(22, extent.width / 45)
        let textImage = labelImage(text: text, fontSize: fontSize)
        let x = extent.minX + sample.normalizedROI.midX * extent.width - textImage.extent.width / 2
        let y = extent.minY + (1 - sample.normalizedROI.minY) * extent.height + fontSize * 0.4
        return textImage.transformed(by: CGAffineTransform(translationX: x, y: y))
            .composited(over: result)
    }

    private func ellipse(
        circle: LiveCircle,
        roi: CGRect,
        extent: CGRect,
        color: UIColor,
        lineWidth: CGFloat
    ) -> CIImage {
        guard circle.x.isFinite,
              circle.y.isFinite,
              circle.radius.isFinite,
              circle.radius > 0,
              extent.width.isFinite,
              extent.height.isFinite,
              extent.width > 1,
              extent.height > 1 else { return .empty() }
        let normalizedX = roi.minX + CGFloat(circle.x / Double(LivePupilSegmenter.modelWidth)) * roi.width
        let normalizedY = roi.minY + CGFloat(circle.y / Double(LivePupilSegmenter.modelHeight)) * roi.height
        let centerX = extent.minX + normalizedX * extent.width
        let centerY = extent.minY + (1 - normalizedY) * extent.height
        let radiusX = CGFloat(circle.radius / Double(LivePupilSegmenter.modelWidth)) * roi.width * extent.width
        let radiusY = CGFloat(circle.radius / Double(LivePupilSegmenter.modelHeight)) * roi.height * extent.height
        guard centerX.isFinite,
              centerY.isFinite,
              radiusX.isFinite,
              radiusY.isFinite,
              radiusX >= 1,
              radiusY >= 1,
              radiusX <= extent.width,
              radiusY <= extent.height else { return .empty() }
        let size = CGSize(width: max(2, radiusX * 2 + lineWidth * 2),
                          height: max(2, radiusY * 2 + lineWidth * 2))
        let format = UIGraphicsImageRendererFormat()
        format.opaque = false
        format.scale = 1
        let image = UIGraphicsImageRenderer(size: size, format: format).image { _ in
            color.setStroke()
            let bounds = CGRect(origin: .zero, size: size).insetBy(dx: lineWidth, dy: lineWidth)
            let path = UIBezierPath(ovalIn: bounds)
            path.lineWidth = lineWidth
            path.stroke()
        }
        guard let cgImage = image.cgImage else { return .empty() }
        return CIImage(cgImage: cgImage).transformed(by: CGAffineTransform(
            translationX: centerX - size.width / 2,
            y: centerY - size.height / 2
        ))
    }

    private func labelImage(text: String, fontSize: CGFloat) -> CIImage {
        let font = UIFont.monospacedSystemFont(ofSize: fontSize, weight: .semibold)
        let attributes: [NSAttributedString.Key: Any] = [
            .font: font,
            .foregroundColor: UIColor.white,
            .strokeColor: UIColor.black,
            .strokeWidth: -3.0
        ]
        let measured = (text as NSString).size(withAttributes: attributes)
        let size = CGSize(width: ceil(measured.width + 24), height: ceil(measured.height + 16))
        let format = UIGraphicsImageRendererFormat()
        format.opaque = false
        format.scale = 1
        let image = UIGraphicsImageRenderer(size: size, format: format).image { _ in
            (text as NSString).draw(at: CGPoint(x: 12, y: 8), withAttributes: attributes)
        }
        guard let cgImage = image.cgImage else { return .empty() }
        return CIImage(cgImage: cgImage)
    }
}

struct RecordingManifest: Codable {
    let id: UUID
    let timestamp: Date
    let annotatedVideoFilename: String
    let firstFrameFilename: String
    let lastFrameFilename: String
    let measurements: [EyeMeasurementSummary]
}

enum CaptureLibrary {
    static func save(
        result: AnnotatedCaptureResult,
        timestamp: Date = Date()
    ) throws -> Recording {
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let id = UUID()
        let firstName = "\(id.uuidString)_first.png"
        let lastName = "\(id.uuidString)_last.png"
        guard let firstData = result.firstFrame.pngData(),
              let lastData = result.lastFrame.pngData() else {
            throw LibraryError.imageEncodingFailed
        }
        try firstData.write(to: documents.appendingPathComponent(firstName), options: .atomic)
        try lastData.write(to: documents.appendingPathComponent(lastName), options: .atomic)
        let manifest = RecordingManifest(
            id: id,
            timestamp: timestamp,
            annotatedVideoFilename: result.annotatedVideoURL.lastPathComponent,
            firstFrameFilename: firstName,
            lastFrameFilename: lastName,
            measurements: result.measurements
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        let manifestURL = documents.appendingPathComponent("\(id.uuidString).capture_manifest.json")
        try encoder.encode(manifest).write(to: manifestURL, options: .atomic)
        return recording(from: manifest, documents: documents)
    }

    static func load() -> [Recording] {
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let urls = (try? FileManager.default.contentsOfDirectory(
            at: documents,
            includingPropertiesForKeys: nil
        )) ?? []
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return urls
            .filter { $0.lastPathComponent.hasSuffix(".capture_manifest.json") }
            .compactMap { url -> Recording? in
                guard let data = try? Data(contentsOf: url),
                      let manifest = try? decoder.decode(RecordingManifest.self, from: data),
                      FileManager.default.fileExists(
                        atPath: documents.appendingPathComponent(manifest.annotatedVideoFilename).path
                      ) else { return nil }
                return recording(from: manifest, documents: documents)
            }
            .sorted { $0.timestamp > $1.timestamp }
    }

    private static func recording(from manifest: RecordingManifest, documents: URL) -> Recording {
        let first = UIImage(contentsOfFile: documents.appendingPathComponent(manifest.firstFrameFilename).path)
            ?? UIImage()
        let last = UIImage(contentsOfFile: documents.appendingPathComponent(manifest.lastFrameFilename).path)
            ?? first
        return Recording(
            id: manifest.id,
            thumbnail: first,
            firstFrame: first,
            lastFrame: last,
            videoURL: documents.appendingPathComponent(manifest.annotatedVideoFilename),
            measurements: manifest.measurements,
            timestamp: manifest.timestamp
        )
    }

    private enum LibraryError: LocalizedError {
        case imageEncodingFailed

        var errorDescription: String? { "Could not encode capture thumbnails." }
    }
}
