//
//  ContentView.swift
//  PupilDetection1
//
//  Created by Daniel Garcia on 1/31/24.
//

import SwiftUI
import Photos
import AVFoundation
import Vision
import CoreHaptics
import PhotosUI

struct ContentView: View {
    @State private var recordings: [Recording] = []
    @State private var isShowingCamera = false
    
    var body: some View {
        NavigationView {
            VStack {
                VStack(alignment: .center, spacing: 0) {
                    Text("👁️ Visualyze")
                        .font(.largeTitle)
                        .padding(.top)
                    
                    Text("Measure pupil dilation and constriction.")
                        .font(.headline)
                        .foregroundColor(.secondary)
                }
                .padding(.bottom)
                
                Divider()
                
                List {
                    Section(header: Text("Recordings").font(.headline).foregroundColor(.primary)) {
                        ForEach(recordings) { recording in
                            NavigationLink(destination: RecordingDetailView(recording: recording)) {
                                HStack {
                                    Image(uiImage: recording.firstFrame)
                                        .resizable()
                                        .aspectRatio(contentMode: .fill)
                                        .frame(width: 50, height: 50)
                                        .clipShape(Circle())
                                    VStack(alignment: .leading) {
                                        Text(dateToString(recording.timestamp))
                                            .font(.headline).bold()
                                        Text(timeToString(recording.timestamp))
                                            .font(.subheadline)
                                    }
                                    .padding(.leading, 8)
                                }
                            }
                        }
                    }
                }
                .listStyle(InsetGroupedListStyle())
                
                Spacer()
                
                Button(action: {
                    isShowingCamera.toggle()
                }) {
                    Label("Start Measurement", systemImage: "camera.viewfinder")
                        .font(.headline)
                        .padding()
                        .foregroundColor(.white)
                        .background(Color.blue)
                        .cornerRadius(10)
                }
                .padding(.bottom)
            }
            .fullScreenCover(isPresented: $isShowingCamera, content: CustomCameraView.init)
            .navigationBarHidden(true)
            .padding(.horizontal)
        }
        .onAppear {
            NotificationCenter.default.addObserver(forName: .newRecording, object: nil, queue: .main) { notification in
                if let userInfo = notification.userInfo, let newRecording = userInfo["recording"] as? Recording {
                    self.recordings.insert(newRecording, at: 0)
                }
            }
            populateSampleData()
        }
        .onDisappear {
            NotificationCenter.default.removeObserver(self, name: .newRecording, object: nil)
        }
    }
    
    func populateSampleData() {
        let sampleThumbnail = UIImage(named: "SamplePupil") ?? UIImage()
        
        let initialDiameter = Double.random(in: 4.00...6.00).rounded(toPlaces: 2)
        let finalDiameter = Double.random(in: 2.75...4.00).rounded(toPlaces: 2)

        // Calculate percentChange and other values if necessary
        let percentChange = ((finalDiameter - initialDiameter) / initialDiameter) * 100
        
        let averageConstrictionVelocity = abs(finalDiameter - initialDiameter) / 3
        
        recordings = [
            Recording(thumbnail: sampleThumbnail, initialDiameter: initialDiameter, finalDiameter: finalDiameter, percentChange: percentChange, averageConstrictionVelocity: averageConstrictionVelocity, firstFrame: sampleThumbnail, lastFrame: sampleThumbnail, timestamp: Date())
        ]
    }
    
    private func dateToString(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "MMMM d, yyyy" // Full month name, day, year
        return formatter.string(from: date)
    }
    
    private func timeToString(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "h:mm a" // Hour:Minute AM/PM
        formatter.amSymbol = "a.m."
        formatter.pmSymbol = "p.m."
        return formatter.string(from: date).lowercased() // Ensure lowercase AM/PM indicators
    }
}


struct CustomCameraView: UIViewControllerRepresentable {
    @Environment(\.presentationMode) var presentationMode
    
    func makeUIViewController(context: Context) -> UIViewController {
        let controller = CustomCameraViewController()
        controller.dismissHandler = {
            presentationMode.wrappedValue.dismiss()
        }
        return controller
    }
    
    func updateUIViewController(_ uiViewController: UIViewController, context: Context) {
        // Update the UI if needed
    }
}

class CustomCameraViewController: UIViewController {
    private var captureSession: AVCaptureSession?
    private var videoPreviewLayer: AVCaptureVideoPreviewLayer?
    private var cameraOutput: AVCaptureMovieFileOutput?
    private var torchIsOn = false
    private var hapticEngine: CHHapticEngine?
    private var eyeMode: Int = 1  // 1 for single eye, 2 for both eyes
    var dismissHandler: (() -> Void)?
    
    private let topBar: UIView = {
        let view = UIView()
        view.translatesAutoresizingMaskIntoConstraints = false
        view.backgroundColor = .black
        return view
    }()
    
    private let bottomBar: UIView = {
        let view = UIView()
        view.translatesAutoresizingMaskIntoConstraints = false
        view.backgroundColor = .black
        return view
    }()
    
    private let torchButton: UIButton = {
        let button = UIButton(type: .system)
        let symbolConfig = UIImage.SymbolConfiguration(pointSize: 50, weight: .medium, scale: .medium)
        let torchOffImage = UIImage(systemName: "flashlight.slash.circle", withConfiguration: symbolConfig)
        button.setImage(torchOffImage, for: .normal)
        button.tintColor = .systemYellow // Customize as needed
        button.translatesAutoresizingMaskIntoConstraints = false
        return button
    }()
    
    private let recordButton: UIButton = {
        let button = UIButton(type: .system)
        let symbolConfig = UIImage.SymbolConfiguration(pointSize: 50, weight: .regular, scale: .large)
        let recordImage = UIImage(systemName: "record.circle", withConfiguration: symbolConfig)
        button.setImage(recordImage, for: .normal)
        button.tintColor = .red
        button.translatesAutoresizingMaskIntoConstraints = false
        return button
    }()
    
    private let instructionLabel: UILabel = {
        let label = UILabel()
        label.translatesAutoresizingMaskIntoConstraints = false
        label.backgroundColor = UIColor.black.withAlphaComponent(0.7) // Semi-transparent background
        label.textColor = .white
        label.text = "Center patient's pupil in red box."
        label.textAlignment = .center
        label.layer.cornerRadius = 10
        label.clipsToBounds = true
        return label
    }()
    
    private let dismissButton: UIButton = {
        let button = UIButton(type: .system)
        let symbolConfig = UIImage.SymbolConfiguration(pointSize: 25, weight: .medium, scale: .medium)
        button.setImage(UIImage(systemName: "xmark.circle.fill", withConfiguration: symbolConfig), for: .normal)
        button.tintColor = .white
        button.translatesAutoresizingMaskIntoConstraints = false
        return button
    }()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupHapticEngine()
        setupCaptureSession()
        setupUIComponents()
        drawBoundingBox()
    }
    
    private func setupHapticEngine() {
        do {
            hapticEngine = try CHHapticEngine()
            try hapticEngine?.start()
        } catch {
            print("Failed to start haptic engine: \(error)")
        }
    }
    
    private func setupUIComponents() {
        setupBars()
        setupTorchButton()
        setupRecordButton()
        setupDismissButton()
        setupInstructionLabel()
        setupModeButton()
    }
    
    private func setupInstructionLabel() {
        view.addSubview(instructionLabel)
        NSLayoutConstraint.activate([
            instructionLabel.centerYAnchor.constraint(equalTo: dismissButton.centerYAnchor),
            instructionLabel.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -20),
            instructionLabel.widthAnchor.constraint(equalToConstant: 250), // Adjust width as needed
            instructionLabel.heightAnchor.constraint(equalToConstant: 40) // Adjust height as needed
        ])
    }
    
    private func setupBars() {
        // Add top bar
        view.addSubview(topBar)
        NSLayoutConstraint.activate([
            topBar.topAnchor.constraint(equalTo: view.topAnchor),
            topBar.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            topBar.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            topBar.heightAnchor.constraint(equalToConstant: 70) // Increased height for better visibility
        ])
        
        // Add bottom bar, allowing it to extend under the home indicator
        view.addSubview(bottomBar)
        NSLayoutConstraint.activate([
            bottomBar.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            bottomBar.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            bottomBar.bottomAnchor.constraint(equalTo: view.bottomAnchor), // This line lets the bar extend to the bottom edge
            // Instead of setting the top anchor relative to the safe area's bottom, set a height for the bar
            bottomBar.heightAnchor.constraint(equalToConstant: 150) // Set the height of the bottom bar
        ])
    }
    
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        videoPreviewLayer?.frame = view.bounds
        drawBoundingBox()
    }
    
    private func setupDismissButton() {
        view.addSubview(dismissButton)
        
        NSLayoutConstraint.activate([
            dismissButton.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 20),
            dismissButton.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 20)
        ])
        
        dismissButton.addTarget(self, action: #selector(dismissCameraView), for: .touchUpInside)
    }
    
    @objc private func dismissCameraView() {
        dismissHandler?()
    }
    
    
    private func hapticFeedbackStartRecording() {
        guard CHHapticEngine.capabilitiesForHardware().supportsHaptics else { return }
        var events = [CHHapticEvent]()
        
        // Start recording pattern
        let intensity = CHHapticEventParameter(parameterID: .hapticIntensity, value: 1.0)
        let sharpness = CHHapticEventParameter(parameterID: .hapticSharpness, value: 0.5)
        let startEvent = CHHapticEvent(eventType: .hapticTransient, parameters: [intensity, sharpness], relativeTime: 0)
        events.append(startEvent)
        
        do {
            let pattern = try CHHapticPattern(events: events, parameters: [])
            let player = try hapticEngine?.makePlayer(with: pattern)
            try player?.start(atTime: 0)
        } catch {
            print("Failed to play haptic feedback: \(error)")
        }
    }
    
    private func hapticFeedbackStopRecording() {
        guard CHHapticEngine.capabilitiesForHardware().supportsHaptics else { return }
        var events = [CHHapticEvent]()
        
        // Parameters for both haptic taps
        let intensity = CHHapticEventParameter(parameterID: .hapticIntensity, value: 1.0)
        let sharpness = CHHapticEventParameter(parameterID: .hapticSharpness, value: 0.5)
        
        // First haptic tap event
        let firstTap = CHHapticEvent(eventType: .hapticTransient, parameters: [intensity, sharpness], relativeTime: 0)
        events.append(firstTap)
        
        // Second haptic tap event, with a slight delay
        let secondTap = CHHapticEvent(eventType: .hapticTransient, parameters: [intensity, sharpness], relativeTime: 0.1) // Delay the second tap by 0.1 seconds
        events.append(secondTap)
        
        do {
            let pattern = try CHHapticPattern(events: events, parameters: [])
            let player = try hapticEngine?.makePlayer(with: pattern)
            try player?.start(atTime: 0)
        } catch {
            print("Failed to play haptic feedback: \(error)")
        }
    }
    
    
    private func setupCaptureSession() {
        captureSession = AVCaptureSession()
        guard let captureSession = captureSession else { return }

        captureSession.beginConfiguration()
        guard let videoDevice = AVCaptureDevice.default(for: .video) else { return }

        // Configure device for high-speed 1080p@120fps and lock exposure/white-balance
        do {
            try videoDevice.lockForConfiguration()

            // Find and set 1080p @ 120fps high-speed format
            if let format = videoDevice.formats.first(where: { format in
                let dims = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
                let frameRates = format.videoSupportedFrameRateRanges
                return dims.width == 1920 && dims.height == 1080 && frameRates.contains(where: { $0.maxFrameRate >= 120 })
            }) {
                videoDevice.activeFormat = format
                videoDevice.activeVideoMinFrameDuration = CMTimeMake(value: 1, timescale: 120)
                videoDevice.activeVideoMaxFrameDuration = CMTimeMake(value: 1, timescale: 120)
                print("✓ Set 1080p @ 120fps high-speed format")
            } else {
                print("⚠ 1080p @ 120fps format not available, using default")
            }

            // Lock exposure and white balance for stable baseline
            videoDevice.exposureMode = .locked
            videoDevice.whiteBalanceMode = .locked
            print("✓ Locked exposure and white balance")

            videoDevice.unlockForConfiguration()
        } catch {
            print("Error configuring video device: \(error)")
        }

        do {
            let videoInput = try AVCaptureDeviceInput(device: videoDevice)
            if captureSession.canAddInput(videoInput) {
                captureSession.addInput(videoInput)
            }
            
            cameraOutput = AVCaptureMovieFileOutput()
            if captureSession.canAddOutput(cameraOutput!) {
                captureSession.addOutput(cameraOutput!)
            }
            
            videoPreviewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
            videoPreviewLayer?.frame = view.bounds
            videoPreviewLayer?.videoGravity = .resizeAspectFill
            view.layer.addSublayer(videoPreviewLayer!)
            
            captureSession.commitConfiguration()
            
            // Dispatching startRunning to a background queue
            DispatchQueue.global(qos: .userInitiated).async {
                captureSession.startRunning()
            }
        } catch {
            print("Error setting up video input: \(error)")
        }
    }
    
    private func setupTorchButton() {
        bottomBar.addSubview(torchButton)
        torchButton.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            torchButton.bottomAnchor.constraint(equalTo: bottomBar.bottomAnchor, constant: -50),
            torchButton.leadingAnchor.constraint(equalTo: bottomBar.leadingAnchor, constant: 30)
        ])
        torchButton.addTarget(self, action: #selector(toggleTorch), for: .touchUpInside)
    }
    
    private func setupRecordButton() {
        bottomBar.addSubview(recordButton)
        recordButton.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            recordButton.bottomAnchor.constraint(equalTo: bottomBar.bottomAnchor, constant: -50),
            recordButton.trailingAnchor.constraint(equalTo: bottomBar.trailingAnchor, constant: -30)
        ])
        recordButton.addTarget(self, action: #selector(toggleRecording), for: .touchUpInside)
    }

    private func setupModeButton() {
        let modeButton = UIButton(type: .system)
        let symbolConfig = UIImage.SymbolConfiguration(pointSize: 20, weight: .medium, scale: .medium)
        modeButton.setImage(UIImage(systemName: "eye.fill", withConfiguration: symbolConfig), for: .normal)
        modeButton.tintColor = .systemBlue
        modeButton.translatesAutoresizingMaskIntoConstraints = false
        topBar.addSubview(modeButton)
        NSLayoutConstraint.activate([
            modeButton.trailingAnchor.constraint(equalTo: topBar.trailingAnchor, constant: -20),
            modeButton.centerYAnchor.constraint(equalTo: topBar.centerYAnchor)
        ])
        modeButton.addTarget(self, action: #selector(toggleEyeMode), for: .touchUpInside)
    }

    @objc private func toggleEyeMode() {
        eyeMode = eyeMode == 1 ? 2 : 1
        drawBoundingBox()
        print("Eye mode: \(eyeMode) eye\(eyeMode == 2 ? "s" : "")")
    }

    @objc private func toggleTorch() {
        guard let device = AVCaptureDevice.default(for: .video), device.hasTorch else { return }
        do {
            try device.lockForConfiguration()
            device.torchMode = torchIsOn ? .off : .on
            torchIsOn.toggle()
            device.unlockForConfiguration()
            
            // Adjust the button's icon based on the torch's current state.
            let symbolConfig = UIImage.SymbolConfiguration(pointSize: 50, weight: .medium, scale: .medium)
            let torchImage = torchIsOn ? UIImage(systemName: "flashlight.on.circle", withConfiguration: symbolConfig) : UIImage(systemName: "flashlight.slash.circle", withConfiguration: symbolConfig)
            
            DispatchQueue.main.async {
                self.torchButton.setImage(torchImage, for: .normal)
                // Optionally adjust the button's tint color to indicate the torch's state visually.
                self.torchButton.tintColor = self.torchIsOn ? .systemYellow : .systemGray
            }
        } catch {
            print("Torch could not be used: \(error)")
        }
    }
    
    @objc private func toggleRecording() {
        if cameraOutput?.isRecording == true {
            cameraOutput?.stopRecording()
            hapticFeedbackStopRecording()
            let symbolConfig = UIImage.SymbolConfiguration(pointSize: 50, weight: .regular, scale: .large)
            let stopImage = UIImage(systemName: "record.circle", withConfiguration: symbolConfig)
            recordButton.setImage(stopImage, for: .normal)
            recordButton.tintColor = .red
        } else {
            hapticFeedbackStartRecording()
            let documentsPath = NSSearchPathForDirectoriesInDomains(.documentDirectory, .userDomainMask, true)[0]
            let outputPath = "\(documentsPath)/output.mov"
            let outputFileURL = URL(fileURLWithPath: outputPath)
            try? FileManager.default.removeItem(at: outputFileURL)
            
            cameraOutput?.startRecording(to: outputFileURL, recordingDelegate: self)
            let symbolConfig = UIImage.SymbolConfiguration(pointSize: 50, weight: .regular, scale: .large)
            let recordImage = UIImage(systemName: "stop.circle", withConfiguration: symbolConfig)
            recordButton.setImage(recordImage, for: .normal)
            recordButton.tintColor = .red
        }
    }
    
    private func stopHapticEngine() {
        hapticEngine?.stop(completionHandler: nil) // Stop the engine immediately
        hapticEngine = nil // Optionally, set the engine to nil to release it for garbage collection
    }
    
    private var boundingBoxLayer: CAShapeLayer?
    
    private func drawBoundingBox() {
        boundingBoxLayer?.removeFromSuperlayer()

        let boxWidth: CGFloat = 200
        let boxHeight: CGFloat = 200
        let centerY = (view.bounds.height - boxHeight) / 2

        if eyeMode == 1 {
            // Single eye: centered
            let startX = (view.bounds.width - boxWidth) / 2
            let rect = CGRect(x: startX, y: centerY, width: boxWidth, height: boxHeight)
            drawRectangle(at: rect)
        } else {
            // Two eyes: side by side with gap
            let gap: CGFloat = 20
            let totalWidth = boxWidth * 2 + gap
            let startX = (view.bounds.width - totalWidth) / 2

            // Left eye
            let leftRect = CGRect(x: startX, y: centerY, width: boxWidth, height: boxHeight)
            drawRectangle(at: leftRect)

            // Right eye
            let rightRect = CGRect(x: startX + boxWidth + gap, y: centerY, width: boxWidth, height: boxHeight)
            drawRectangle(at: rightRect)
        }
    }

    private func drawRectangle(at rect: CGRect) {
        let shapeLayer = CAShapeLayer()
        shapeLayer.frame = rect
        shapeLayer.borderColor = UIColor.red.cgColor
        shapeLayer.borderWidth = 2
        shapeLayer.cornerRadius = 10
        view.layer.addSublayer(shapeLayer)
        boundingBoxLayer = shapeLayer
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        stopHapticEngine() // Ensure the haptic engine is stopped before the view disappears
        captureSession?.stopRunning()
    }

    private func saveVideoToPhotoLibrary(_ videoURL: URL) {
        PHPhotoLibrary.requestAuthorization(for: .addOnly) { status in
            guard status == .authorized else {
                print("Photo library permission denied")
                return
            }
            PHPhotoLibrary.shared().performChanges {
                PHAssetChangeRequest.creationRequestForAssetFromVideo(atFileURL: videoURL)
            } completionHandler: { success, error in
                if success {
                    print("Video saved to camera roll")
                } else if let error = error {
                    print("Error saving video to camera roll: \(error.localizedDescription)")
                }
            }
        }
    }

}

extension CustomCameraViewController: AVCaptureFileOutputRecordingDelegate {
    func fileOutput(_ output: AVCaptureFileOutput, didFinishRecordingTo outputFileURL: URL, from connections: [AVCaptureConnection], error: Error?) {
        guard error == nil else {
            print("Error recording video: \(error!.localizedDescription)")
            return
        }

        // Save to camera roll
        saveVideoToPhotoLibrary(outputFileURL)

        // Extract frames and create recording
        extractFrames(from: outputFileURL) { firstFrame, lastFrame in
            let initialDiameter = Double.random(in: 4.00...6.00).rounded(toPlaces: 2)
            let finalDiameter = Double.random(in: 2.75...4.00).rounded(toPlaces: 2)
            
            let recording = Recording(
                thumbnail: UIImage(), // Placeholder for thumbnail
                initialDiameter: initialDiameter, // Placeholder
                finalDiameter: finalDiameter, // Placeholder
                percentChange: ((finalDiameter - initialDiameter) / initialDiameter) * 100, // Placeholder
                averageConstrictionVelocity: abs(finalDiameter - initialDiameter) / 3, // Placeholder
                firstFrame: firstFrame ?? UIImage(),
                lastFrame: lastFrame ?? UIImage(),
                timestamp: Date()
            )
            
            NotificationCenter.default.post(name: .newRecording, object: nil, userInfo: ["recording": recording])
        }
        DispatchQueue.main.async {
            self.dismissHandler?()
        }
    }
    
    
    private func extractFrames(from videoURL: URL, completion: @escaping (UIImage?, UIImage?) -> Void) {
        let asset = AVAsset(url: videoURL)
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        
        let times = [NSValue(time: CMTimeMake(value: 0, timescale: 1)), NSValue(time: asset.duration)]
        var firstFrame: UIImage?
        var lastFrame: UIImage?
        
        generator.generateCGImagesAsynchronously(forTimes: times) { requestedTime, image, actualTime, result, error in
            guard let cgImage = image else { return }
            
            let croppedImage = self.cropToSquare(image: cgImage)
            
            if requestedTime == CMTimeMake(value: 0, timescale: 1) {
                firstFrame = croppedImage
            } else {
                lastFrame = croppedImage
            }
            
            if firstFrame != nil && lastFrame != nil {
                completion(firstFrame, lastFrame)
            }
        }
    }
    
    private func cropToSquare(image: CGImage, cropFactor: CGFloat = 0.30) -> UIImage? {
        // Ensure cropFactor is between 0 and 1, where 1 means no extra cropping (just square), and closer to 0 means more cropping.
        let validCropFactor = min(max(cropFactor, 0), 1)
        
        let contextSize = CGSize(width: image.width, height: image.height)
        
        let minLength = CGFloat(min(image.width, image.height))
        let cropLength = minLength * validCropFactor // Apply cropFactor to reduce the size
        
        // Recalculate offsets to center the smaller crop area
        let offsetX = (contextSize.width - cropLength) / 2
        let offsetY = (contextSize.height - cropLength) / 2
        let cropRect = CGRect(x: offsetX, y: offsetY, width: cropLength, height: cropLength)
        
        if let cropped = image.cropping(to: cropRect) {
            return UIImage(cgImage: cropped)
        } else {
            return nil
        }
    }
}


extension CustomCameraViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        let request = VNDetectContoursRequest(completionHandler: handleContours)
        request.contrastAdjustment = 2.0 // Increase contrast to make pupil detection more likely
        request.detectsDarkOnLight = true
        
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up, options: [:])
        try? handler.perform([request])
    }
    
    private func handleContours(request: VNRequest, error: Error?) {
        
        // This approach assumes we find a way to determine a relevant contour for the iris.
        // It's a placeholder to illustrate conversion of a contour to a bounding box.
        // In practice, you would need more sophisticated logic to identify the iris among detected contours.
        
        // Directly drawing bounding box from contours is not straightforward with VNContoursObservation.
        // As an alternative, consider using other Vision requests or custom algorithms for iris detection.
        
        // Example of drawing a placeholder bounding box for illustration
        _ = CGRect(x: 0.25, y: 0.25, width: 0.25, height: 0.25) // Placeholder values
        DispatchQueue.main.async { [weak self] in
            self?.drawBoundingBox()
        }
    }
    
    // Assuming drawBoundingBox is correctly implemented to draw on the UIView coordinate system.
}

struct RecordingDetailView: View {
    var recording: Recording
    
    private func dateToString(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "MMMM d, yyyy" // Full month name, day, year
        return formatter.string(from: date)
    }
    
    private func timeToString(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "h:mm a" // Hour:Minute AM/PM
        formatter.amSymbol = "a.m."
        formatter.pmSymbol = "p.m."
        return formatter.string(from: date).lowercased() // Ensure lowercase AM/PM indicators
    }
    
    var body: some View {
        List {
            Section(header: Text("Pupil Images")) {
                HStack(spacing: 20) {
                    Spacer()
                    Image(uiImage: recording.firstFrame)
                        .resizable()
                        .scaledToFit()
                        .clipShape(RoundedRectangle(cornerRadius: 10))
                        .padding(5)
                    
                    Image(uiImage: recording.lastFrame)
                        .resizable()
                        .scaledToFit()
                        .clipShape(RoundedRectangle(cornerRadius: 10))
                        .padding(5)
                    Spacer()
                }
                .frame(height: 200) // Specify a height to ensure images are displayed properly
            }
            
            Section(header: Text("Measurement Details")) {
                Text("Initial Diameter: \(recording.initialDiameter, specifier: "%.2f") mm")
                Text("Final Diameter: \(recording.finalDiameter, specifier: "%.2f") mm")
                Text("Percent Change: \(recording.percentChange, specifier: "%.2f")%")
                Text("Average Constriction Velocity: \(recording.averageConstrictionVelocity, specifier: "%.2f") mm/s")
                Text("Date: \(dateToString(recording.timestamp))")
                Text("Time: \(timeToString(recording.timestamp))")
            }
        }
        .listStyle(InsetGroupedListStyle())
        .navigationTitle("Recording Details")
        .navigationBarTitleDisplayMode(.inline)
    }
}



struct Recording: Identifiable {
    var id: UUID = UUID()
    var thumbnail: UIImage
    var initialDiameter: Double
    var finalDiameter: Double
    var percentChange: Double
    var averageConstrictionVelocity: Double
    var firstFrame: UIImage
    var lastFrame: UIImage
    var timestamp: Date
}

extension Notification.Name {
    static let newRecording = Notification.Name("newRecording")
}


extension Double {
    /// Rounds the double to decimal places value
    func rounded(toPlaces places:Int) -> Double {
        let divisor = pow(10.0, Double(places))
        return (self * divisor).rounded() / divisor
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
