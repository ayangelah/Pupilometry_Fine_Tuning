import CoreImage
import CoreML
import Foundation
import QuartzCore


struct LiveCircle: Codable {
    let x: Double
    let y: Double
    let radius: Double
}


struct LiveEyePrediction {
    let iris: LiveCircle?
    let pupil: LiveCircle?
    let accepted: Bool
    let latencyMs: Double
    let preprocessingMs: Double
    let modelMs: Double
    let postprocessingMs: Double
}


/// Experimental on-device adapter for the pretrained RITnet model.
///
/// The exported model deliberately accepts the same preprocessed tensor used in Python. This
/// class owns mutable scratch buffers and must only be called from one serial inference queue.
final class LivePupilSegmenter {
    static let modelWidth = 400
    static let modelHeight = 640
    static let modelIdentifier = "ritnet_pretrained_coreml_fp16"
    static let pipelineIdentifier = "single_eye_known_good_v1"

    private let model: MLModel
    private let input: MLMultiArray
    // Preserve the exact Core Image conversion used by the best-performing on-device build.
    private let ciContext = CIContext(options: [.cacheIntermediates: false])
    private let colorSpace = CGColorSpaceCreateDeviceRGB()
    private var rgba = [UInt8](repeating: 0, count: modelWidth * modelHeight * 4)
    private var gammaRed = [UInt8](repeating: 0, count: modelWidth * modelHeight)
    private var equalized = [UInt8](repeating: 0, count: modelWidth * modelHeight)
    private var mask = [UInt8](repeating: 0, count: modelWidth * modelHeight)
    private let gammaLUT: [UInt8] = (0..<256).map { value in
        UInt8(clamping: Int((255.0 * pow(Double(value) / 255.0, 0.8)).rounded(.down)))
    }

    init() throws {
        guard let url = Bundle.main.url(forResource: "RITnet", withExtension: "mlmodelc") else {
            throw SegmenterError.modelMissing
        }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        model = try MLModel(contentsOf: url, configuration: configuration)
        input = try MLMultiArray(
            shape: [
                NSNumber(value: 1), NSNumber(value: 1),
                NSNumber(value: Self.modelHeight), NSNumber(value: Self.modelWidth)
            ],
            dataType: .float32
        )
    }

    func predict(pixelBuffer: CVPixelBuffer, normalizedTopLeftROI roi: CGRect) throws -> LiveEyePrediction {
        let started = CACurrentMediaTime()
        try renderEye(pixelBuffer: pixelBuffer, normalizedTopLeftROI: roi)
        applyGammaAndCLAHE()
        writeInputTensor()
        let preprocessed = CACurrentMediaTime()

        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "eye": MLFeatureValue(multiArray: input)
        ])
        let output = try model.prediction(from: provider)
        let predicted = CACurrentMediaTime()
        guard let logits = output.featureValue(for: "logits")?.multiArrayValue else {
            throw SegmenterError.outputMissing
        }
        try makeMask(from: logits)
        let circles = fitCircles()
        let finished = CACurrentMediaTime()
        return LiveEyePrediction(
            iris: circles.iris,
            pupil: circles.pupil,
            accepted: circles.accepted,
            latencyMs: (finished - started) * 1000,
            preprocessingMs: (preprocessed - started) * 1000,
            modelMs: (predicted - preprocessed) * 1000,
            postprocessingMs: (finished - predicted) * 1000
        )
    }

    private func renderEye(pixelBuffer: CVPixelBuffer, normalizedTopLeftROI requested: CGRect) throws {
        let unit = CGRect(x: 0, y: 0, width: 1, height: 1)
        let roi = requested.standardized.intersection(unit)
        guard roi.width > 0.01, roi.height > 0.01 else { throw SegmenterError.invalidROI }

        let image = CIImage(cvPixelBuffer: pixelBuffer)
        let extent = image.extent
        // AVCaptureVideoPreviewLayer returns a top-left normalized rectangle; Core Image uses a
        // bottom-left origin. The capture connection is configured to portrait before this call.
        let crop = CGRect(
            x: extent.minX + roi.minX * extent.width,
            y: extent.minY + (1 - roi.maxY) * extent.height,
            width: roi.width * extent.width,
            height: roi.height * extent.height
        ).intersection(extent)
        guard crop.width > 1, crop.height > 1 else { throw SegmenterError.invalidROI }

        let translated = image.cropped(to: crop).transformed(
            by: CGAffineTransform(translationX: -crop.minX, y: -crop.minY)
        )
        // The model expects an exact 400x640 tensor. A later Lanczos experiment applied one
        // uniform scale to this portrait crop, producing 400x400 image content in a 400x640
        // destination. Restore the known-good independent X/Y transform.
        let scaled = translated.transformed(by: CGAffineTransform(
            scaleX: CGFloat(Self.modelWidth) / crop.width,
            y: CGFloat(Self.modelHeight) / crop.height
        ))
        let bounds = CGRect(x: 0, y: 0, width: Self.modelWidth, height: Self.modelHeight)
        rgba.withUnsafeMutableBytes { bytes in
            guard let base = bytes.baseAddress else { return }
            ciContext.render(
                scaled,
                toBitmap: base,
                rowBytes: Self.modelWidth * 4,
                bounds: bounds,
                format: .RGBA8,
                colorSpace: colorSpace
            )
        }
    }

    private func applyGammaAndCLAHE() {
        for i in 0..<gammaRed.count {
            gammaRed[i] = gammaLUT[Int(rgba[i * 4])]
        }

        // OpenCV-compatible CLAHE geometry for the fixed 400x640 model input: 8x8 tiles,
        // clipLimit 1.5. This keeps the mobile preprocessing contract close to validation.
        let gridX = 8, gridY = 8
        let tileWidth = Self.modelWidth / gridX
        let tileHeight = Self.modelHeight / gridY
        let tileArea = tileWidth * tileHeight
        let clipLimit = max(Int(1.5 * Double(tileArea) / 256.0), 1)
        var luts = [UInt8](repeating: 0, count: gridX * gridY * 256)

        for tileY in 0..<gridY {
            for tileX in 0..<gridX {
                var histogram = [Int](repeating: 0, count: 256)
                let startX = tileX * tileWidth, startY = tileY * tileHeight
                for y in startY..<(startY + tileHeight) {
                    let row = y * Self.modelWidth
                    for x in startX..<(startX + tileWidth) {
                        histogram[Int(gammaRed[row + x])] += 1
                    }
                }

                var clipped = 0
                for bin in 0..<256 where histogram[bin] > clipLimit {
                    clipped += histogram[bin] - clipLimit
                    histogram[bin] = clipLimit
                }
                let batch = clipped / 256
                if batch > 0 {
                    for bin in 0..<256 { histogram[bin] += batch }
                }
                var residual = clipped - batch * 256
                if residual > 0 {
                    let step = max(256 / residual, 1)
                    var bin = 0
                    while residual > 0 && bin < 256 {
                        histogram[bin] += 1
                        residual -= 1
                        bin += step
                    }
                }

                var cumulative = 0
                let offset = (tileY * gridX + tileX) * 256
                for bin in 0..<256 {
                    cumulative += histogram[bin]
                    luts[offset + bin] = UInt8(clamping: Int(
                        (Double(cumulative) * 255.0 / Double(tileArea)).rounded()
                    ))
                }
            }
        }

        for y in 0..<Self.modelHeight {
            let tileY = Double(y) / Double(tileHeight) - 0.5
            let rawY0 = Int(floor(tileY)), rawY1 = rawY0 + 1
            let y0 = min(max(rawY0, 0), gridY - 1), y1 = min(max(rawY1, 0), gridY - 1)
            let wy = tileY - floor(tileY)
            for x in 0..<Self.modelWidth {
                let tileX = Double(x) / Double(tileWidth) - 0.5
                let rawX0 = Int(floor(tileX)), rawX1 = rawX0 + 1
                let x0 = min(max(rawX0, 0), gridX - 1), x1 = min(max(rawX1, 0), gridX - 1)
                let wx = tileX - floor(tileX)
                let value = Int(gammaRed[y * Self.modelWidth + x])
                let top = (1 - wx) * Double(luts[(y0 * gridX + x0) * 256 + value])
                        + wx * Double(luts[(y0 * gridX + x1) * 256 + value])
                let bottom = (1 - wx) * Double(luts[(y1 * gridX + x0) * 256 + value])
                           + wx * Double(luts[(y1 * gridX + x1) * 256 + value])
                equalized[y * Self.modelWidth + x] = UInt8(clamping: Int(
                    ((1 - wy) * top + wy * bottom).rounded()
                ))
            }
        }
    }

    private func writeInputTensor() {
        let values = input.dataPointer.bindMemory(to: Float32.self, capacity: equalized.count)
        for i in 0..<equalized.count {
            values[i] = Float32(equalized[i]) / 127.5 - 1
        }
    }

    private func makeMask(from logits: MLMultiArray) throws {
        guard logits.dataType == .float32, logits.shape.map(\.intValue) == [1, 4, 640, 400] else {
            throw SegmenterError.unexpectedOutputShape(logits.shape.map(\.intValue))
        }
        let values = logits.dataPointer.bindMemory(to: Float32.self, capacity: logits.count)
        let strides = logits.strides.map(\.intValue)
        for y in 0..<Self.modelHeight {
            for x in 0..<Self.modelWidth {
                var bestClass = 0
                var bestValue = -Float32.infinity
                for cls in 0..<4 {
                    let offset = cls * strides[1] + y * strides[2] + x * strides[3]
                    if values[offset] > bestValue {
                        bestValue = values[offset]
                        bestClass = cls
                    }
                }
                mask[y * Self.modelWidth + x] = UInt8(bestClass)
            }
        }
    }

    private struct Point { let x: Double; let y: Double }

    private func fitCircles() -> (iris: LiveCircle?, pupil: LiveCircle?, accepted: Bool) {
        var limbus = [Point]()
        for y in 2..<(Self.modelHeight - 2) {
            for x in 2..<(Self.modelWidth - 2) {
                let index = y * Self.modelWidth + x
                guard mask[index] == 2 || mask[index] == 3 else { continue }
                var boundary = false, nearSclera = false
                for dy in -2...2 {
                    for dx in -2...2 where dx != 0 || dy != 0 {
                        let neighbor = mask[(y + dy) * Self.modelWidth + x + dx]
                        if abs(dx) <= 1 && abs(dy) <= 1 && neighbor != 2 && neighbor != 3 {
                            boundary = true
                        }
                        if neighbor == 1 { nearSclera = true }
                    }
                }
                if boundary && nearSclera { limbus.append(Point(x: Double(x), y: Double(y))) }
            }
        }
        guard let iris = robustCircle(limbus), iris.radius > 0 else {
            return (nil, nil, false)
        }

        var pupilRim = [Point]()
        for y in 1..<(Self.modelHeight - 1) {
            for x in 1..<(Self.modelWidth - 1) {
                let index = y * Self.modelWidth + x
                guard mask[index] == 3 else { continue }
                let dxCenter = Double(x) - iris.x, dyCenter = Double(y) - iris.y
                guard hypot(dxCenter, dyCenter) <= iris.radius * 1.1 else { continue }
                var boundary = false
                for dy in -1...1 {
                    for dx in -1...1 where dx != 0 || dy != 0 {
                        if mask[(y + dy) * Self.modelWidth + x + dx] != 3 { boundary = true }
                    }
                }
                if boundary { pupilRim.append(Point(x: Double(x), y: Double(y))) }
            }
        }
        guard let pupil = robustCircle(pupilRim) else { return (iris, nil, false) }
        let ratio = pupil.radius / iris.radius
        let concentricity = hypot(pupil.x - iris.x, pupil.y - iris.y) / iris.radius
        let accepted = (0.15...0.75).contains(ratio) && concentricity <= 0.25
        return (iris, pupil, accepted)
    }

    private func robustCircle(_ points: [Point]) -> LiveCircle? {
        guard points.count >= 5, var circle = leastSquaresCircle(points) else { return nil }
        for _ in 0..<3 {
            let retained = points.filter {
                abs(hypot($0.x - circle.x, $0.y - circle.y) - circle.radius) < 6
            }
            guard retained.count >= 5, let refit = leastSquaresCircle(retained) else { break }
            circle = refit
        }
        return circle
    }

    private func leastSquaresCircle(_ points: [Point]) -> LiveCircle? {
        var a = 0.0, b = 0.0, c = 0.0, d = 0.0, e = 0.0
        var g = 0.0, h = 0.0, i = 0.0
        for point in points {
            let px = 2 * point.x, py = 2 * point.y
            let squared = point.x * point.x + point.y * point.y
            a += px * px; b += px * py; c += px
            d += py * py; e += py
            g += px * squared; h += py * squared; i += squared
        }
        let f = Double(points.count)
        let determinant = a * (d * f - e * e) - b * (b * f - e * c) + c * (b * e - d * c)
        guard abs(determinant) > 1e-9 else { return nil }
        let centerX = (g * (d * f - e * e) - b * (h * f - e * i) + c * (h * e - d * i)) / determinant
        let centerY = (a * (h * f - e * i) - g * (b * f - e * c) + c * (b * i - h * c)) / determinant
        let constant = (a * (d * i - h * e) - b * (b * i - h * c) + g * (b * e - d * c)) / determinant
        let radiusSquared = constant + centerX * centerX + centerY * centerY
        guard radiusSquared.isFinite, radiusSquared > 0 else { return nil }
        return LiveCircle(x: centerX, y: centerY, radius: sqrt(radiusSquared))
    }

    private enum SegmenterError: Error {
        case modelMissing
        case invalidROI
        case outputMissing
        case unexpectedOutputShape([Int])
    }
}
