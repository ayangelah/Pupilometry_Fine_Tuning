//
//  ContentView.swift
//  PupilDetection1
//
//  Created by Daniel Garcia on 1/31/24.
//

import SwiftUI
import Photos
import AVFoundation
import AVKit
import CoreHaptics
import PhotosUI

private struct CaptureFrameSample: Codable {
    let videoTimeS: Double
    let presentationTimeS: Double
    let meanLuma: Double
}

private struct TorchCaptureEvent: Codable {
    let state: String
    let requestedHostTimeS: Double
    let firstFrameAfterRequestPresentationTimeS: Double
    let firstFrameAfterRequestVideoTimeS: Double
}

private struct LiveInferenceSample: Codable {
    let videoTimeS: Double
    let presentationTimeS: Double
    let eye: String
    let roiX: Double
    let roiY: Double
    let roiWidth: Double
    let roiHeight: Double
    let latencyMs: Double
    let preprocessingMs: Double
    let modelMs: Double
    let postprocessingMs: Double
    let accepted: Bool
    let meanLuma: Double
    let sharpness: Double
    let saturatedFraction: Double
    let iris: LiveCircle?
    let pupil: LiveCircle?
}

private struct CaptureQualitySample: Codable {
    let videoTimeS: Double
    let presentationTimeS: Double
    let meanLuma: Double
    let sharpness: Double
    let saturatedFraction: Double
    let darkFraction: Double
}

private struct CaptureSidecar: Codable {
    let schemaVersion: Int
    let captureMode: String
    let requestedFps: Double
    let activeFps: Double
    let activeWidth: Int32
    let activeHeight: Int32
    let videoDataBufferWidth: Int
    let videoDataBufferHeight: Int
    let hdrEnabled: Bool
    let cameraDeviceType: String
    let minimumFocusDistanceMm: Int
    let torchLevel: Float
    let stabilizationMode: String
    let droppedVideoDataFrames: Int
    let modelIdentifier: String
    let pipelineIdentifier: String
    let inferenceStride: Int
    let derivedLightOnsetVideoTimeS: Double?
    let frames: [CaptureFrameSample]
    let qualitySamples: [CaptureQualitySample]
    let torchEvents: [TorchCaptureEvent]
    let liveInferences: [LiveInferenceSample]
}

private struct EyeFrameQuality {
    let meanLuma: Double
    let sharpness: Double
    let saturatedFraction: Double
    let darkFraction: Double
}

private enum CaptureMode: String {
    case measurement120 = "measurement_120"
    case dataset60 = "dataset_4k60"

    var requestedFPS: Double { self == .measurement120 ? 120 : 60 }
    var buttonTitle: String { self == .measurement120 ? "120 PLR" : "4K60 DATA" }
    var fileTag: String { self == .measurement120 ? "plr120" : "data60" }
}

private struct CaptureFormatChoice {
    let format: AVCaptureDevice.Format
    let width: Int32
    let height: Int32
    let fps: Double
    let hdrSupported: Bool
}

struct ContentView: View {
    @State private var recordings: [Recording] = []
    @State private var isShowingCamera = false
    @State private var processingCaptureCount = 0
    
    var body: some View {
        NavigationView {
            VStack {
                VStack(alignment: .center, spacing: 0) {
                    Text("👁️ Visualyze")
                        .font(.largeTitle)
                        .padding(.top)
                    
                    Text("Experimental live pupil and iris tracking.")
                        .font(.headline)
                        .foregroundColor(.secondary)
                }
                .padding(.bottom)
                
                Divider()
                
                List {
                    Section(header: Text("Recordings").font(.headline).foregroundColor(.primary)) {
                        if processingCaptureCount > 0 {
                            HStack {
                                ProgressView()
                                Text("Rendering annotated capture…")
                                    .foregroundColor(.secondary)
                            }
                        }
                        if recordings.isEmpty {
                            Text("Completed annotated captures will appear here. Raw movies and timing sidecars remain available through Files.")
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                        }
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
                    Label("Start Experimental Capture", systemImage: "camera.viewfinder")
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
            recordings = CaptureLibrary.load()
        }
        .onReceive(NotificationCenter.default.publisher(for: .newRecording)) { notification in
            if let newRecording = notification.userInfo?["recording"] as? Recording {
                recordings.removeAll { $0.id == newRecording.id }
                recordings.insert(newRecording, at: 0)
            }
        }
        .onReceive(NotificationCenter.default.publisher(for: .captureProcessingStarted)) { _ in
            processingCaptureCount += 1
        }
        .onReceive(NotificationCenter.default.publisher(for: .captureProcessingFinished)) { _ in
            processingCaptureCount = max(0, processingCaptureCount - 1)
        }
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
    private var videoDataOutput: AVCaptureVideoDataOutput?
    private var cameraDevice: AVCaptureDevice?
    private let sessionQueue = DispatchQueue(label: "com.visualyze.capture.session")
    private let videoOutputQueue = DispatchQueue(label: "com.visualyze.capture.frames")
    private let inferenceQueue = DispatchQueue(label: "com.visualyze.inference", qos: .userInitiated)
    private var liveSegmenter: LivePupilSegmenter?
    private var liveInferenceInFlight = false
    private var liveInferenceFrameCounter = 0
    private var liveInferenceFailureReported = false
    private var liveNormalizedROI: CGRect?
    private var qualityNormalizedROI: CGRect?
    private var liveModelRect: CGRect = .zero
    private let liveIrisLayer = CAShapeLayer()
    private let livePupilLayer = CAShapeLayer()
    private var liveMeasurementPoints: [String: [(time: Double, diameterMM: Double)]] = [:]
    private var lightOnsetVideoTimeS: Double?
    private var captureMode: CaptureMode = .measurement120
    private var activeCaptureFPS: Double = 0
    private var activeCaptureWidth: Int32 = 0
    private var activeHDREnabled = false
    private var activeStabilizationMode = "off"
    private var videoDataBufferWidth = 0
    private var videoDataBufferHeight = 0
    private var force1080pFallback = false
    private var sessionHasStarted = false
    private var cameraSettingsLockedForRecording = false
    private var qualityPassStreak = 0
    private var qualityFailureStreak = 0
    private var captureQualityReady = false
    private let measurementExposureBias: Float = -0.7
    // A conservative, repeatable starting level—not a clinically calibrated illuminance.
    private let provisionalTorchLevel: Float = 0.20
    private var recordingStartPTS: CMTime?
    private var captureFrameSamples: [CaptureFrameSample] = []
    private var captureQualitySamples: [CaptureQualitySample] = []
    private var torchCaptureEvents: [TorchCaptureEvent] = []
    private var liveInferenceSamples: [LiveInferenceSample] = []
    private var pendingTorchRequest: (state: String, hostTimeS: Double)?
    private var droppedVideoDataFrames = 0
    private var torchIsOn = false
    private var hapticEngine: CHHapticEngine?
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
        label.numberOfLines = 2
        label.adjustsFontSizeToFitWidth = true
        label.minimumScaleFactor = 0.75
        label.layer.cornerRadius = 10
        label.clipsToBounds = true
        return label
    }()

    private let measurementLabel: UILabel = {
        let label = UILabel()
        label.translatesAutoresizingMaskIntoConstraints = false
        label.backgroundColor = UIColor.black.withAlphaComponent(0.68)
        label.textColor = .white
        label.font = .monospacedSystemFont(ofSize: 14, weight: .semibold)
        label.text = "Experimental measurement: --"
        label.textAlignment = .center
        label.numberOfLines = 2
        label.layer.cornerRadius = 8
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

    private let captureModeButton: UIButton = {
        let button = UIButton(type: .system)
        button.titleLabel?.font = .monospacedSystemFont(ofSize: 13, weight: .semibold)
        button.setTitleColor(.white, for: .normal)
        button.backgroundColor = UIColor.systemBlue.withAlphaComponent(0.75)
        button.layer.cornerRadius = 8
        button.translatesAutoresizingMaskIntoConstraints = false
        return button
    }()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .black
        setupHapticEngine()
        setupUIComponents()
        instructionLabel.text = "Starting camera…"
        setupLiveSegmentation()
        drawBoundingBox()
        startCameraWhenAuthorized()
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
        setupMeasurementLabel()
        setupCaptureModeButton()
    }
    
    private func setupInstructionLabel() {
        view.addSubview(instructionLabel)
        NSLayoutConstraint.activate([
            instructionLabel.centerYAnchor.constraint(equalTo: dismissButton.centerYAnchor),
            instructionLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            instructionLabel.widthAnchor.constraint(equalToConstant: 250), // Adjust width as needed
            instructionLabel.heightAnchor.constraint(equalToConstant: 48) // Room for actionable camera errors
        ])
    }

    private func setupMeasurementLabel() {
        view.addSubview(measurementLabel)
        NSLayoutConstraint.activate([
            measurementLabel.topAnchor.constraint(equalTo: topBar.bottomAnchor, constant: 10),
            measurementLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            measurementLabel.widthAnchor.constraint(lessThanOrEqualTo: view.widthAnchor, constant: -24),
            measurementLabel.heightAnchor.constraint(greaterThanOrEqualToConstant: 36)
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
        if sessionHasStarted && !cameraSettingsLockedForRecording {
            configureEyePointOfInterest()
        }
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
    
    
    private func startCameraWhenAuthorized() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            setupCaptureSession()
        case .notDetermined:
            instructionLabel.text = "Camera permission required…"
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                DispatchQueue.main.async {
                    guard let self else { return }
                    if granted {
                        self.setupCaptureSession()
                    } else {
                        self.showCameraPermissionDenied()
                    }
                }
            }
        case .denied, .restricted:
            showCameraPermissionDenied()
        @unknown default:
            showCameraFailure("Camera permission status is unavailable.")
        }
    }

    private func showCameraPermissionDenied() {
        showCameraFailure("Camera access denied. Enable Camera in Settings.")
        let alert = UIAlertController(
            title: "Camera access required",
            message: "Visualyze needs camera access for the live eye preview. Enable Camera in Settings, then reopen the capture screen.",
            preferredStyle: .alert
        )
        alert.addAction(UIAlertAction(title: "Not Now", style: .cancel))
        alert.addAction(UIAlertAction(title: "Open Settings", style: .default) { _ in
            guard let url = URL(string: UIApplication.openSettingsURLString) else { return }
            UIApplication.shared.open(url)
        })
        present(alert, animated: true)
    }

    private func showCameraFailure(_ message: String) {
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.clearLivePrediction()
            self.instructionLabel.text = message
            self.instructionLabel.backgroundColor = UIColor.systemRed.withAlphaComponent(0.75)
            self.recordButton.isEnabled = false
            self.torchButton.isEnabled = false
            self.recordButton.alpha = 0.45
            self.torchButton.alpha = 0.45
        }
        print("✗ \(message)")
    }

    private func setupCaptureSession() {
        sessionHasStarted = false
        cameraSettingsLockedForRecording = false
        captureQualityReady = false
        qualityPassStreak = 0
        qualityFailureStreak = 0
        recordButton.isEnabled = false
        recordButton.alpha = 0.45
        captureSession = AVCaptureSession()
        guard let captureSession = captureSession else { return }
        guard let videoDevice = preferredCamera(for: captureMode),
              let choice = chooseCaptureFormat(for: videoDevice, mode: captureMode) else {
            showCameraFailure("No compatible \(Int(captureMode.requestedFPS)) fps back-camera format is available.")
            return
        }
        cameraDevice = videoDevice
        logAvailableFormats(videoDevice)

        NotificationCenter.default.addObserver(
            self, selector: #selector(captureSessionDidStart(_:)),
            name: AVCaptureSession.didStartRunningNotification, object: captureSession
        )
        NotificationCenter.default.addObserver(
            self, selector: #selector(captureSessionRuntimeError(_:)),
            name: AVCaptureSession.runtimeErrorNotification, object: captureSession
        )
        NotificationCenter.default.addObserver(
            self, selector: #selector(captureSessionWasInterrupted(_:)),
            name: AVCaptureSession.wasInterruptedNotification, object: captureSession
        )
        NotificationCenter.default.addObserver(
            self, selector: #selector(captureSessionInterruptionEnded(_:)),
            name: AVCaptureSession.interruptionEndedNotification, object: captureSession
        )
        NotificationCenter.default.addObserver(
            self, selector: #selector(cameraSubjectAreaDidChange(_:)),
            name: AVCaptureDevice.subjectAreaDidChangeNotification, object: videoDevice
        )

        captureSession.beginConfiguration()
        do {
            let videoInput = try AVCaptureDeviceInput(device: videoDevice)
            guard captureSession.canAddInput(videoInput) else { throw CaptureSetupError.cannotAddInput }
            captureSession.addInput(videoInput)

            // inputPriority prevents a later session preset from silently replacing activeFormat.
            if captureSession.canSetSessionPreset(.inputPriority) {
                captureSession.sessionPreset = .inputPriority
            }

            try videoDevice.lockForConfiguration()
            videoDevice.activeFormat = choice.format
            let duration = CMTimeMake(value: 1, timescale: Int32(choice.fps.rounded()))
            videoDevice.activeVideoMinFrameDuration = duration
            videoDevice.activeVideoMaxFrameDuration = duration
            activeCaptureFPS = choice.fps
            activeCaptureWidth = choice.width

            videoDevice.automaticallyAdjustsVideoHDREnabled = false
            activeHDREnabled = choice.hdrSupported
            videoDevice.isVideoHDREnabled = activeHDREnabled
            if activeHDREnabled,
               choice.format.supportedColorSpaces.contains(.HLG_BT2020) {
                videoDevice.activeColorSpace = .HLG_BT2020
            }

            if videoDevice.isExposureModeSupported(.continuousAutoExposure) {
                videoDevice.exposureMode = .continuousAutoExposure
            }
            if videoDevice.isWhiteBalanceModeSupported(.continuousAutoWhiteBalance) {
                videoDevice.whiteBalanceMode = .continuousAutoWhiteBalance
            }
            if videoDevice.isFocusModeSupported(.continuousAutoFocus) {
                videoDevice.focusMode = .continuousAutoFocus
            }
            if videoDevice.isAutoFocusRangeRestrictionSupported {
                videoDevice.autoFocusRangeRestriction = .near
            }
            if videoDevice.isSmoothAutoFocusSupported {
                videoDevice.isSmoothAutoFocusEnabled = false
            }
            videoDevice.isSubjectAreaChangeMonitoringEnabled = true
            let bias = min(max(measurementExposureBias,
                               videoDevice.minExposureTargetBias),
                           videoDevice.maxExposureTargetBias)
            videoDevice.setExposureTargetBias(bias, completionHandler: nil)
            if videoDevice.isLowLightBoostSupported {
                // Low-light boost may skip frames; reserve it for non-kinetic dataset capture.
                videoDevice.automaticallyEnablesLowLightBoostWhenAvailable = captureMode == .dataset60
            }
            videoDevice.unlockForConfiguration()
            print("✓ Selected \(choice.width)x\(choice.height) @ \(Int(choice.fps))fps hdr=\(activeHDREnabled) device=\(videoDevice.deviceType.rawValue)")

            let movieOutput = AVCaptureMovieFileOutput()
            if captureSession.canAddOutput(movieOutput) {
                captureSession.addOutput(movieOutput)
                cameraOutput = movieOutput
                if videoDevice.isVirtualDevice {
                    movieOutput.isPrimaryConstituentDeviceSwitchingBehaviorForRecordingEnabled = true
                    movieOutput.setPrimaryConstituentDeviceSwitchingBehaviorForRecording(
                        .locked, restrictedSwitchingBehaviorConditions: []
                    )
                }
            } else {
                cameraOutput = nil
                print("✗ Movie output unavailable; live preview will continue without recording")
            }

            let dataOutput = AVCaptureVideoDataOutput()
            dataOutput.alwaysDiscardsLateVideoFrames = true
            dataOutput.automaticallyConfiguresOutputBufferDimensions = false
            dataOutput.deliversPreviewSizedOutputBuffers = true
            dataOutput.videoSettings = [
                kCVPixelBufferPixelFormatTypeKey as String:
                    kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
            ]
            dataOutput.setSampleBufferDelegate(self, queue: videoOutputQueue)
            if captureSession.canAddOutput(dataOutput) {
                captureSession.addOutput(dataOutput)
                videoDataOutput = dataOutput
            } else {
                videoDataOutput = nil
                print("✗ Frame-data output unavailable; camera will continue without the model overlay")
            }
            configureVideoConnection(cameraOutput?.connection(with: .video), format: choice.format)
            configureVideoConnection(videoDataOutput?.connection(with: .video), format: choice.format)
            configureMovieCodec(movieOutput)

            captureSession.commitConfiguration()

            let preview = AVCaptureVideoPreviewLayer(session: captureSession)
            preview.frame = view.bounds
            preview.videoGravity = .resizeAspectFill
            view.layer.insertSublayer(preview, at: 0)
            videoPreviewLayer = preview
            configureVideoConnection(preview.connection, format: choice.format)
            drawBoundingBox()
            configureEyePointOfInterest()

            instructionLabel.backgroundColor = UIColor.black.withAlphaComponent(0.7)
            instructionLabel.text = "Starting camera session…"
            sessionQueue.async { [weak self] in
                captureSession.startRunning()
                self?.logCaptureSettings("session running")
            }
        } catch {
            captureSession.commitConfiguration()
            showCameraFailure("Camera setup failed: \(error.localizedDescription)")
        }
    }

    private func preferredCamera(for mode: CaptureMode) -> AVCaptureDevice? {
        let discovery = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInTripleCamera, .builtInDualWideCamera, .builtInWideAngleCamera],
            mediaType: .video,
            position: .back
        )
        return discovery.devices
            .filter { chooseCaptureFormat(for: $0, mode: mode) != nil }
            .max { lhs, rhs in
                guard let left = chooseCaptureFormat(for: lhs, mode: mode),
                      let right = chooseCaptureFormat(for: rhs, mode: mode) else { return false }
                let leftMacroBonus = lhs.isVirtualDevice ? 120 : 0
                let rightMacroBonus = rhs.isVirtualDevice ? 120 : 0
                return formatScore(left, mode: mode) + leftMacroBonus
                    < formatScore(right, mode: mode) + rightMacroBonus
            }
    }

    private func chooseCaptureFormat(
        for device: AVCaptureDevice,
        mode: CaptureMode
    ) -> CaptureFormatChoice? {
        let requestedFPS = mode.requestedFPS
        return device.formats.compactMap { format -> CaptureFormatChoice? in
            guard format.videoSupportedFrameRateRanges.contains(where: {
                $0.minFrameRate <= requestedFPS && $0.maxFrameRate >= requestedFPS
            }) else { return nil }
            let dimensions = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
            if force1080pFallback,
               max(dimensions.width, dimensions.height) > 1920 {
                return nil
            }
            return CaptureFormatChoice(
                format: format,
                width: dimensions.width,
                height: dimensions.height,
                fps: requestedFPS,
                hdrSupported: format.isVideoHDRSupported
            )
        }.max { formatScore($0, mode: mode) < formatScore($1, mode: mode) }
    }

    private func formatScore(_ choice: CaptureFormatChoice, mode: CaptureMode) -> Int {
        let longEdge = max(choice.width, choice.height)
        let shortEdge = min(choice.width, choice.height)
        var score: Int
        if longEdge >= 3840 && shortEdge >= 2160 {
            score = 3_000
        } else if longEdge >= 1920 && shortEdge >= 1080 {
            score = 2_000
        } else {
            score = Int(Int64(longEdge) * Int64(shortEdge) / 10_000)
        }
        if choice.hdrSupported { score += 500 }
        if choice.format.isVideoBinned { score += mode == .dataset60 ? 40 : 100 }
        if choice.format.isVideoStabilizationModeSupported(.standard) { score += 40 }
        if Int64(longEdge) * 9 == Int64(shortEdge) * 16 { score += 100 }
        return score
    }

    private func configureVideoConnection(
        _ connection: AVCaptureConnection?,
        format: AVCaptureDevice.Format
    ) {
        guard let connection else { return }
        configurePortrait(connection)
        guard connection.isVideoStabilizationSupported else { return }
        if #available(iOS 26.0, *), format.isVideoStabilizationModeSupported(.lowLatency) {
            connection.preferredVideoStabilizationMode = .lowLatency
        } else if format.isVideoStabilizationModeSupported(.standard) {
            connection.preferredVideoStabilizationMode = .standard
        }
    }

    private func configureMovieCodec(_ output: AVCaptureMovieFileOutput) {
        guard let connection = output.connection(with: .video),
              output.availableVideoCodecTypes.contains(.hevc) else { return }
        output.setOutputSettings([AVVideoCodecKey: AVVideoCodecType.hevc], for: connection)
    }

    private func restartCaptureSession(reason: String) {
        turnTorchOff()
        cameraSettingsLockedForRecording = false
        captureQualityReady = false
        recordButton.isEnabled = false
        recordButton.alpha = 0.45
        instructionLabel.backgroundColor = UIColor.black.withAlphaComponent(0.7)
        instructionLabel.text = reason
        clearLivePrediction()
        let oldSession = captureSession
        let oldDevice = cameraDevice
        sessionQueue.async { [weak self] in
            if oldSession?.isRunning == true { oldSession?.stopRunning() }
            DispatchQueue.main.async {
                guard let self else { return }
                if let oldSession {
                    NotificationCenter.default.removeObserver(self, name: AVCaptureSession.didStartRunningNotification, object: oldSession)
                    NotificationCenter.default.removeObserver(self, name: AVCaptureSession.runtimeErrorNotification, object: oldSession)
                    NotificationCenter.default.removeObserver(self, name: AVCaptureSession.wasInterruptedNotification, object: oldSession)
                    NotificationCenter.default.removeObserver(self, name: AVCaptureSession.interruptionEndedNotification, object: oldSession)
                }
                if let oldDevice {
                    NotificationCenter.default.removeObserver(self, name: AVCaptureDevice.subjectAreaDidChangeNotification, object: oldDevice)
                }
                self.videoPreviewLayer?.removeFromSuperlayer()
                self.videoPreviewLayer = nil
                self.cameraOutput = nil
                self.videoDataOutput = nil
                self.cameraDevice = nil
                self.captureSession = nil
                self.setupCaptureSession()
            }
        }
    }

    private enum CaptureSetupError: Error {
        case cannotAddInput
    }

    @objc private func captureSessionDidStart(_ notification: Notification) {
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.sessionHasStarted = true
            if (self.videoDataOutput == nil || self.cameraOutput == nil),
               self.activeCaptureWidth >= 3840,
               !self.force1080pFallback {
                self.force1080pFallback = true
                self.restartCaptureSession(reason: "4K output combination unavailable; retrying 1080p…")
                return
            }
            self.instructionLabel.backgroundColor = UIColor.black.withAlphaComponent(0.7)
            self.recordButton.isEnabled = self.cameraOutput != nil
            self.recordButton.alpha = self.recordButton.isEnabled ? 1.0 : 0.45
            self.torchButton.isEnabled = self.cameraDevice?.hasTorch == true
            self.torchButton.alpha = self.torchButton.isEnabled ? 1.0 : 0.45
            self.activeStabilizationMode = self.stabilizationName(
                self.cameraOutput?.connection(with: .video)?.activeVideoStabilizationMode ?? .off
            )
            self.configureEyePointOfInterest()
            if self.liveSegmenter != nil, self.videoDataOutput != nil {
                self.instructionLabel.text = "Record anytime · alignment guidance is advisory."
            } else {
                self.instructionLabel.text = "Camera ready · model preview unavailable."
            }
            self.logCaptureSettings("session ready")
        }
    }

    @objc private func captureSessionRuntimeError(_ notification: Notification) {
        let detail = (notification.userInfo?[AVCaptureSessionErrorKey] as? Error)?.localizedDescription
            ?? "Unknown AVFoundation error"
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            if self.cameraOutput?.isRecording != true,
               self.activeCaptureWidth >= 3840,
               !self.force1080pFallback {
                self.force1080pFallback = true
                self.restartCaptureSession(
                    reason: "4K\(Int(self.captureMode.requestedFPS)) unavailable with live tracking; retrying 1080p\(Int(self.captureMode.requestedFPS))…"
                )
            } else {
                self.showCameraFailure("Camera stopped: \(detail)")
            }
        }
    }

    @objc private func captureSessionWasInterrupted(_ notification: Notification) {
        showCameraFailure("Camera interrupted. Close other camera apps and try again.")
    }

    @objc private func captureSessionInterruptionEnded(_ notification: Notification) {
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.instructionLabel.backgroundColor = UIColor.black.withAlphaComponent(0.7)
            self.instructionLabel.text = "Camera resumed · center one eye."
            self.configureEyePointOfInterest()
        }
    }

    @objc private func cameraSubjectAreaDidChange(_ notification: Notification) {
        DispatchQueue.main.async { [weak self] in
            guard let self, !self.cameraSettingsLockedForRecording else { return }
            self.configureEyePointOfInterest()
        }
    }

    private func configurePortrait(_ connection: AVCaptureConnection?) {
        guard let connection else { return }
        if #available(iOS 17.0, *) {
            if connection.isVideoRotationAngleSupported(90) { connection.videoRotationAngle = 90 }
        } else if connection.isVideoOrientationSupported {
            connection.videoOrientation = .portrait
        }
    }

    private func configureEyePointOfInterest() {
        guard let device = cameraDevice,
              let preview = videoPreviewLayer,
              !cameraSettingsLockedForRecording else { return }
        let viewPoint = liveModelRect.isEmpty
            ? CGPoint(x: view.bounds.midX, y: view.bounds.midY)
            : CGPoint(x: liveModelRect.midX, y: liveModelRect.midY)
        let devicePoint = preview.captureDevicePointConverted(fromLayerPoint: viewPoint)
        do {
            try device.lockForConfiguration()
            if device.isFocusPointOfInterestSupported {
                device.focusPointOfInterest = devicePoint
            }
            if device.isExposurePointOfInterestSupported {
                device.exposurePointOfInterest = devicePoint
            }
            if device.isFocusModeSupported(.continuousAutoFocus) {
                device.focusMode = .continuousAutoFocus
            }
            if device.isExposureModeSupported(.continuousAutoExposure) {
                device.exposureMode = .continuousAutoExposure
            }
            if device.isWhiteBalanceModeSupported(.continuousAutoWhiteBalance) {
                device.whiteBalanceMode = .continuousAutoWhiteBalance
            }
            if device.isAutoFocusRangeRestrictionSupported {
                device.autoFocusRangeRestriction = .near
            }
            let bias = min(max(measurementExposureBias,
                               device.minExposureTargetBias),
                           device.maxExposureTargetBias)
            device.setExposureTargetBias(bias, completionHandler: nil)
            device.isSubjectAreaChangeMonitoringEnabled = true
            device.unlockForConfiguration()
        } catch {
            print("Could not configure eye autofocus/metering: \(error)")
        }
    }

    @discardableResult
    private func lockCameraSettingsForRecording() -> Bool {
        guard let device = cameraDevice else { return false }
        cameraSettingsLockedForRecording = true
        do {
            try device.lockForConfiguration()
            if device.isExposureModeSupported(.locked) { device.exposureMode = .locked }
            if device.isWhiteBalanceModeSupported(.locked) { device.whiteBalanceMode = .locked }
            if device.isFocusModeSupported(.locked) { device.focusMode = .locked }
            device.unlockForConfiguration()
            print("✓ Locked exposure, white balance, and focus after recording started")
            return true
        } catch {
            cameraSettingsLockedForRecording = false
            print("Could not lock metered camera settings: \(error)")
            return false
        }
    }

    private func settleAndLockCameraAfterRecordingStarts(attempt: Int = 0) {
        guard cameraOutput?.isRecording == true,
              !cameraSettingsLockedForRecording,
              let device = cameraDevice else { return }

        let stillAdjusting = device.isAdjustingFocus
            || device.isAdjustingExposure
            || device.isAdjustingWhiteBalance
        let maximumAttempts = 12
        if stillAdjusting && attempt < maximumAttempts {
            instructionLabel.text = "Recording baseline · settling eye focus…"
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.10) { [weak self] in
                self?.settleAndLockCameraAfterRecordingStarts(attempt: attempt + 1)
            }
            return
        }

        if stillAdjusting {
            print("⚠ Camera did not report settled after 1.2 s; locking its current state")
        }
        let locked = lockCameraSettingsForRecording()
        logCaptureSettings(locked ? "post-start camera lock" : "post-start lock failed")

        let torchAvailable = device.hasTorch && device.isTorchAvailable
        torchButton.isEnabled = torchAvailable
        torchButton.alpha = torchAvailable ? 1.0 : 0.45
        instructionLabel.text = captureMode == .measurement120
            ? (locked
                ? "Focus locked · recording baseline · torch ready"
                : "Recording baseline · auto camera · torch ready")
            : (locked ? "Focus locked · recording 4K60 dataset clip"
                      : "Recording 4K60 dataset clip · auto camera")
    }

    private func stabilizationName(_ mode: AVCaptureVideoStabilizationMode) -> String {
        switch mode {
        case .off: return "off"
        case .standard: return "standard"
        case .cinematic: return "cinematic"
        case .cinematicExtended: return "cinematic_extended"
        case .auto: return "auto"
        default:
            if #available(iOS 17.0, *), mode == .previewOptimized { return "preview_optimized" }
            if #available(iOS 26.0, *), mode == .lowLatency { return "low_latency" }
            return "unknown_\(mode.rawValue)"
        }
    }
    
    private func fourCC(_ code: FourCharCode) -> String {
        let bytes = [UInt8((code >> 24) & 255), UInt8((code >> 16) & 255),
                     UInt8((code >> 8) & 255), UInt8(code & 255)]
        return String(bytes: bytes, encoding: .ascii) ?? "????"
    }

    private func modeName(_ m: AVCaptureDevice.ExposureMode) -> String {
        switch m {
        case .locked: return "locked"
        case .autoExpose: return "auto-once"
        case .continuousAutoExposure: return "continuous"
        case .custom: return "custom"
        @unknown default: return "unknown"
        }
    }

    private func modeName(_ m: AVCaptureDevice.WhiteBalanceMode) -> String {
        switch m {
        case .locked: return "locked"
        case .autoWhiteBalance: return "auto-once"
        case .continuousAutoWhiteBalance: return "continuous"
        @unknown default: return "unknown"
        }
    }

    private func modeName(_ m: AVCaptureDevice.FocusMode) -> String {
        switch m {
        case .locked: return "locked"
        case .autoFocus: return "auto-once"
        case .continuousAutoFocus: return "continuous"
        @unknown default: return "unknown"
        }
    }

    /// One-time dump of the formats worth using (high frame rate or 4K), so the format choice
    /// is made from what this phone actually offers rather than from an assumption.
    private func logAvailableFormats(_ device: AVCaptureDevice) {
        print("=== CAPTURE formats on \(device.localizedName) (\(device.formats.count) total, listing >=100fps or >=4K)")
        for (i, format) in device.formats.enumerated() {
            let dims = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
            let maxFps = format.videoSupportedFrameRateRanges.map { $0.maxFrameRate }.max() ?? 0
            guard maxFps >= 100 || dims.width >= 3840 else { continue }
            let ranges = format.videoSupportedFrameRateRanges
                .map { "\(Int($0.minFrameRate))-\(Int($0.maxFrameRate))" }.joined(separator: ",")
            let subtype = fourCC(CMFormatDescriptionGetMediaSubType(format.formatDescription))
            print("  [\(i)] \(dims.width)x\(dims.height) \(subtype) fps \(ranges) hdr=\(format.isVideoHDRSupported) binned=\(format.isVideoBinned)")
        }
    }

    /// Print what the camera is ACTUALLY doing. Console only; it changes no capture behavior.
    private func logCaptureSettings(_ tag: String) {
        guard let device = cameraDevice else { return }
        let dims = CMVideoFormatDescriptionGetDimensions(device.activeFormat.formatDescription)
        let subtype = fourCC(CMFormatDescriptionGetMediaSubType(device.activeFormat.formatDescription))
        let minDur = device.activeVideoMinFrameDuration      // shortest frame = FASTEST fps
        let maxDur = device.activeVideoMaxFrameDuration
        let fastest = minDur.value > 0 ? Double(minDur.timescale) / Double(minDur.value) : 0
        let slowest = maxDur.value > 0 ? Double(maxDur.timescale) / Double(maxDur.value) : 0
        let gains = device.deviceWhiteBalanceGains
        let exposureMs = CMTimeGetSeconds(device.exposureDuration) * 1000
        print("""
        --- CAPTURE [\(tag)]
            format     \(dims.width)x\(dims.height) \(subtype)
            fps        up to \(String(format: "%.2f", fastest)), down to \(String(format: "%.2f", slowest))
            exposure   \(modeName(device.exposureMode))  iso \(String(format: "%.0f", device.iso))  \(String(format: "%.2f", exposureMs)) ms
            whiteBal   \(modeName(device.whiteBalanceMode))  gains R \(String(format: "%.2f", gains.redGain)) G \(String(format: "%.2f", gains.greenGain)) B \(String(format: "%.2f", gains.blueGain))
            focus      \(modeName(device.focusMode))  lens \(String(format: "%.3f", device.lensPosition))
            torch      active=\(device.isTorchActive)  level \(String(format: "%.2f", device.torchLevel))
        """)
    }

    private func setupTorchButton() {
        bottomBar.addSubview(torchButton)
        torchButton.isEnabled = false
        torchButton.alpha = 0.45
        torchButton.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            torchButton.bottomAnchor.constraint(equalTo: bottomBar.bottomAnchor, constant: -50),
            torchButton.leadingAnchor.constraint(equalTo: bottomBar.leadingAnchor, constant: 30)
        ])
        torchButton.addTarget(self, action: #selector(toggleTorch), for: .touchUpInside)
    }
    
    private func setupRecordButton() {
        bottomBar.addSubview(recordButton)
        recordButton.isEnabled = false
        recordButton.alpha = 0.45
        recordButton.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            recordButton.bottomAnchor.constraint(equalTo: bottomBar.bottomAnchor, constant: -50),
            recordButton.trailingAnchor.constraint(equalTo: bottomBar.trailingAnchor, constant: -30)
        ])
        recordButton.addTarget(self, action: #selector(toggleRecording), for: .touchUpInside)
    }

    private func setupCaptureModeButton() {
        captureModeButton.setTitle(captureMode.buttonTitle, for: .normal)
        bottomBar.addSubview(captureModeButton)
        NSLayoutConstraint.activate([
            captureModeButton.topAnchor.constraint(equalTo: bottomBar.topAnchor, constant: 16),
            captureModeButton.centerXAnchor.constraint(equalTo: bottomBar.centerXAnchor),
            captureModeButton.widthAnchor.constraint(equalToConstant: 120),
            captureModeButton.heightAnchor.constraint(equalToConstant: 32)
        ])
        captureModeButton.addTarget(self, action: #selector(toggleCaptureMode), for: .touchUpInside)
    }

    @objc private func toggleCaptureMode() {
        guard cameraOutput?.isRecording != true else { return }
        captureMode = captureMode == .measurement120 ? .dataset60 : .measurement120
        force1080pFallback = false
        captureModeButton.setTitle(captureMode.buttonTitle, for: .normal)
        restartCaptureSession(reason: "Switching to \(captureMode.buttonTitle)…")
    }

    @objc private func toggleTorch() {
        if captureMode == .measurement120, cameraOutput?.isRecording != true {
            let alert = UIAlertController(
                title: "Record the baseline first",
                message: "In 120 PLR mode, start recording before turning on the torch so the movie contains a pre-light baseline and the app can timestamp the stimulus onset.",
                preferredStyle: .alert
            )
            alert.addAction(UIAlertAction(title: "OK", style: .default))
            present(alert, animated: true)
            return
        }
        guard let device = cameraDevice, device.hasTorch else { return }
        guard device.isTorchAvailable else {
            let alert = UIAlertController(
                title: "Torch unavailable",
                message: "The camera reports that the torch is unavailable in its current state. Stop recording, let the phone cool if needed, and retry.",
                preferredStyle: .alert
            )
            alert.addAction(UIAlertAction(title: "OK", style: .default))
            present(alert, animated: true)
            return
        }
        do {
            try device.lockForConfiguration()
            defer { device.unlockForConfiguration() }
            if torchIsOn {
                device.torchMode = .off
            } else {
                try device.setTorchModeOn(level: provisionalTorchLevel)
            }
            torchIsOn.toggle()

            let state = torchIsOn ? "on" : "off"
            let requestHostTimeS = CMTimeGetSeconds(CMClockGetTime(CMClockGetHostTimeClock()))
            videoOutputQueue.async { [weak self] in
                self?.pendingTorchRequest = (state: state, hostTimeS: requestHostTimeS)
            }
            print("=== TORCH \(state.uppercased()) requested at host time \(String(format: "%.6f", requestHostTimeS))")
            logCaptureSettings("torch \(torchIsOn ? "on" : "off")")
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.15) { [weak self] in
                self?.logCaptureSettings("torch state after ramp")
            }
            
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

    private func turnTorchOff() {
        guard torchIsOn, let device = cameraDevice else { return }
        do {
            try device.lockForConfiguration()
            device.torchMode = .off
            device.unlockForConfiguration()
        } catch {
            print("Could not turn torch off: \(error)")
        }
        torchIsOn = false
        let symbolConfig = UIImage.SymbolConfiguration(pointSize: 50, weight: .medium, scale: .medium)
        torchButton.setImage(UIImage(systemName: "flashlight.slash.circle", withConfiguration: symbolConfig), for: .normal)
        torchButton.tintColor = .systemGray
    }
    
    @objc private func toggleRecording() {
        guard let cameraOutput else {
            let alert = UIAlertController(
                title: "Recording unavailable",
                message: "The camera preview started, but this device could not add the movie-recording output.",
                preferredStyle: .alert
            )
            alert.addAction(UIAlertAction(title: "OK", style: .default))
            present(alert, animated: true)
            return
        }
        if cameraOutput.isRecording {
            cameraOutput.stopRecording()
            hapticFeedbackStopRecording()
            torchButton.isEnabled = false
            torchButton.alpha = 0.45
            captureModeButton.isEnabled = true
            let symbolConfig = UIImage.SymbolConfiguration(pointSize: 50, weight: .regular, scale: .large)
            let stopImage = UIImage(systemName: "record.circle", withConfiguration: symbolConfig)
            recordButton.setImage(stopImage, for: .normal)
            recordButton.tintColor = .red
        } else {
            // Let AVFoundation continue the already-targeted autofocus after movie capture begins.
            // The recording delegate locks the settled state before enabling the PLR torch.
            cameraSettingsLockedForRecording = false
            configureEyePointOfInterest()
            liveMeasurementPoints.removeAll(keepingCapacity: true)
            lightOnsetVideoTimeS = nil
            measurementLabel.text = "Preliminary measurement · waiting for valid fits"
            hapticFeedbackStartRecording()
            let stamp = DateFormatter()
            stamp.dateFormat = "yyyyMMdd_HHmmss"
            let fileName = "visualyze_\(captureMode.fileTag)_\(stamp.string(from: Date())).mov"
            let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            let outputFileURL = documentsURL.appendingPathComponent(fileName)

            print("=== RECORDING START \(fileName)\n    folder: \(documentsURL.path)")
            logCaptureSettings("record start")

            videoOutputQueue.async { [weak self] in
                self?.recordingStartPTS = nil
                self?.captureFrameSamples.removeAll(keepingCapacity: true)
                self?.captureQualitySamples.removeAll(keepingCapacity: true)
                self?.torchCaptureEvents.removeAll(keepingCapacity: true)
                self?.liveInferenceSamples.removeAll(keepingCapacity: true)
                self?.pendingTorchRequest = nil
                self?.droppedVideoDataFrames = 0
            }

            cameraOutput.startRecording(to: outputFileURL, recordingDelegate: self)
            captureModeButton.isEnabled = false
            torchButton.isEnabled = false
            torchButton.alpha = 0.45
            instructionLabel.text = "Starting recording · settling eye focus…"
            let symbolConfig = UIImage.SymbolConfiguration(pointSize: 50, weight: .regular, scale: .large)
            let recordImage = UIImage(systemName: "stop.circle", withConfiguration: symbolConfig)
            recordButton.setImage(recordImage, for: .normal)
            recordButton.tintColor = .red
        }
    }
    
    private func stopHapticEngine() {
        // The capture screen can dismiss while the stop-recording pattern is still playing.
        // Releasing the engine avoids racing an active player with stop(), which logs -4810.
        hapticEngine = nil
    }
    
    private var boundingBoxLayers: [CAShapeLayer] = []

    private func setupLiveSegmentation() {
        liveIrisLayer.fillColor = UIColor.clear.cgColor
        liveIrisLayer.strokeColor = UIColor.systemBlue.cgColor
        liveIrisLayer.lineWidth = 2
        livePupilLayer.fillColor = UIColor.clear.cgColor
        livePupilLayer.strokeColor = UIColor.systemGreen.cgColor
        livePupilLayer.lineWidth = 2
        view.layer.addSublayer(liveIrisLayer)
        view.layer.addSublayer(livePupilLayer)
        do {
            liveSegmenter = try LivePupilSegmenter()
            print("✓ Loaded \(LivePupilSegmenter.modelIdentifier) · \(LivePupilSegmenter.pipelineIdentifier)")
        } catch {
            liveSegmenter = nil
            print("✗ Core ML model unavailable: \(error)")
        }
    }
    
    private func drawBoundingBox() {
        boundingBoxLayers.forEach { $0.removeFromSuperlayer() }
        boundingBoxLayers.removeAll(keepingCapacity: true)
        let boxWidth = min(CGFloat(200), view.bounds.width - 32)
        let boxHeight = boxWidth
        let originY = (view.bounds.height - boxHeight) / 2
        let originX = (view.bounds.width - boxWidth) / 2
        let rect = CGRect(x: originX, y: originY, width: boxWidth, height: boxHeight)
        drawRectangle(at: rect)
        updateLiveInferenceROI(around: rect)
    }

    private func updateLiveInferenceROI(around eyeBox: CGRect) {
        // Match Python's OpenEDS reframing: tight eye width fills a 400x640 portrait tensor.
        let width = eyeBox.width * 1.05
        let height = width * CGFloat(LivePupilSegmenter.modelHeight) / CGFloat(LivePupilSegmenter.modelWidth)
        liveModelRect = CGRect(
            x: eyeBox.midX - width / 2,
            y: eyeBox.midY - height / 2,
            width: width,
            height: height
        ).intersection(view.bounds)
        guard let preview = videoPreviewLayer else { return }
        let normalized = preview.metadataOutputRectConverted(fromLayerRect: liveModelRect)
        videoOutputQueue.async { [weak self] in
            self?.liveNormalizedROI = normalized
            self?.qualityNormalizedROI = normalized
        }
    }

    private func drawLivePrediction(_ prediction: LiveEyePrediction, videoTimeS: Double?) {
        func path(for circle: LiveCircle?) -> CGPath? {
            guard let circle, !liveModelRect.isEmpty else { return nil }
            let scaleX = liveModelRect.width / CGFloat(LivePupilSegmenter.modelWidth)
            let scaleY = liveModelRect.height / CGFloat(LivePupilSegmenter.modelHeight)
            let center = CGPoint(x: liveModelRect.minX + CGFloat(circle.x) * scaleX,
                                 y: liveModelRect.minY + CGFloat(circle.y) * scaleY)
            let radiusX = CGFloat(circle.radius) * scaleX
            let radiusY = CGFloat(circle.radius) * scaleY
            return UIBezierPath(ovalIn: CGRect(
                x: center.x - radiusX,
                y: center.y - radiusY,
                width: 2 * radiusX,
                height: 2 * radiusY
            )).cgPath
        }
        liveIrisLayer.path = path(for: prediction.iris)
        livePupilLayer.path = path(for: prediction.pupil)
        livePupilLayer.strokeColor = (prediction.accepted
            ? UIColor.systemGreen : UIColor.systemOrange).cgColor

        if let videoTimeS,
           prediction.accepted,
           let iris = prediction.iris,
           let pupil = prediction.pupil,
           iris.radius > 0 {
            let diameter = pupil.radius / iris.radius * MeasurementCalculator.irisReferenceMM
            liveMeasurementPoints["Eye", default: []].append((videoTimeS, diameter))
        }
        let label = MeasurementCalculator.liveText(
            eye: "Eye",
            points: liveMeasurementPoints["Eye", default: []],
            lightOnsetVideoTimeS: lightOnsetVideoTimeS
        )
        measurementLabel.text = "Preliminary · iris reference 11.7 mm\n" + label
    }

    private func updateCaptureReadiness(
        predictions: [LiveEyePrediction],
        quality: EyeFrameQuality,
        focusAdjusting: Bool,
        exposureAdjusting: Bool
    ) {
        if cameraOutput?.isRecording == true {
            if quality.saturatedFraction > 0.10 {
                instructionLabel.backgroundColor = UIColor.systemRed.withAlphaComponent(0.75)
                instructionLabel.text = "Recording · torch overexposure detected"
            } else {
                instructionLabel.backgroundColor = UIColor.black.withAlphaComponent(0.7)
                if let latency = predictions.map(\.latencyMs).max() {
                    instructionLabel.text = String(format: "Recording · live model %.0f ms", latency)
                } else {
                    instructionLabel.text = "Recording · collecting quality diagnostics"
                }
            }
            return
        }

        let irisDiameters = predictions.compactMap { $0.iris.map { $0.radius * 2 } }
        var passes = false
        var prompt: String

        if quality.saturatedFraction > 0.06 {
            prompt = "Too bright · move back or reduce reflections."
        } else if quality.meanLuma < 22 || quality.darkFraction > 0.70 {
            prompt = "Too dark · add ambient light and hold steady."
        } else if focusAdjusting {
            prompt = "Focusing on the eye…"
        } else if exposureAdjusting {
            prompt = "Adjusting eye exposure…"
        } else if quality.sharpness < 0.015 {
            prompt = qualityFailureStreak > 20
                ? "Still blurry · move the phone back slowly."
                : "Hold still while the eye sharpens…"
        } else if captureMode == .measurement120, predictions.isEmpty {
            prompt = "Model is missing an eye · recording is still available."
        } else if captureMode == .measurement120, irisDiameters.contains(where: { $0 < 90 }) {
            prompt = "Move slightly closer while keeping focus."
        } else if captureMode == .measurement120, irisDiameters.contains(where: { $0 > 330 }) {
            prompt = "Move slightly farther from the eye."
        } else if captureMode == .measurement120, predictions.contains(where: { !$0.accepted }) {
            prompt = "Improve alignment if possible · recording is available."
        } else {
            passes = true
            prompt = "Hold steady…"
        }

        if passes {
            qualityPassStreak += 1
            qualityFailureStreak = 0
            if qualityPassStreak >= 5 { captureQualityReady = true }
        } else {
            qualityPassStreak = 0
            qualityFailureStreak += 1
            if qualityFailureStreak >= 3 { captureQualityReady = false }
        }

        let boxColor = captureQualityReady ? UIColor.systemGreen.cgColor : UIColor.systemOrange.cgColor
        boundingBoxLayers.forEach { $0.borderColor = boxColor }
        instructionLabel.backgroundColor = captureQualityReady
            ? UIColor.systemGreen.withAlphaComponent(0.72)
            : UIColor.black.withAlphaComponent(0.7)
        if captureQualityReady {
            if let latency = predictions.map(\.latencyMs).max() {
                instructionLabel.text = String(
                    format: "Good alignment · sharp %.3f · %.0f ms", quality.sharpness, latency
                )
            } else {
                instructionLabel.text = String(format: "Good alignment · sharp %.3f", quality.sharpness)
            }
        } else {
            instructionLabel.text = prompt
        }
    }

    private func clearLivePrediction() {
        liveIrisLayer.path = nil
        livePupilLayer.path = nil
        measurementLabel.text = "Experimental measurement: --"
    }

    private func drawRectangle(at rect: CGRect) {
        let shapeLayer = CAShapeLayer()
        shapeLayer.frame = rect
        shapeLayer.borderColor = UIColor.red.cgColor
        shapeLayer.borderWidth = 2
        shapeLayer.cornerRadius = 10
        view.layer.addSublayer(shapeLayer)
        boundingBoxLayers.append(shapeLayer)
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        stopHapticEngine()
        turnTorchOff()
        NotificationCenter.default.removeObserver(self)
        if let captureSession {
            sessionQueue.async {
                if captureSession.isRunning { captureSession.stopRunning() }
            }
        }
    }

    private func saveVideosToPhotoLibrary(
        _ videoURLs: [URL],
        completion: @escaping (Bool) -> Void
    ) {
        PHPhotoLibrary.requestAuthorization(for: .addOnly) { status in
            guard status == .authorized || status == .limited else {
                print("Photo library permission denied")
                completion(false)
                return
            }
            PHPhotoLibrary.shared().performChanges {
                for videoURL in videoURLs {
                    PHAssetChangeRequest.creationRequestForAssetFromVideo(atFileURL: videoURL)
                }
            } completionHandler: { success, error in
                if success {
                    print("\(videoURLs.count) video version(s) saved to Photos")
                } else if let error = error {
                    print("Error saving videos to Photos: \(error.localizedDescription)")
                }
                completion(success)
            }
        }
    }

    private func removeTemporaryRawCapture(_ rawVideoURL: URL) {
        do {
            try FileManager.default.removeItem(at: rawVideoURL)
            print("Removed temporary clean source after Photos confirmed both saves")
        } catch {
            print("Could not remove temporary clean source: \(error)")
        }
    }

}

extension CustomCameraViewController: AVCaptureFileOutputRecordingDelegate {
    func fileOutput(
        _ output: AVCaptureFileOutput,
        didStartRecordingTo fileURL: URL,
        from connections: [AVCaptureConnection]
    ) {
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            // Keep the torch disabled until autofocus/exposure/white balance have had a real
            // post-start settling window and their current values are locked.
            self.torchButton.isEnabled = false
            self.torchButton.alpha = 0.45
            self.instructionLabel.backgroundColor = UIColor.black.withAlphaComponent(0.7)
            self.instructionLabel.text = self.captureMode == .measurement120
                ? "Recording baseline · settling eye focus…"
                : "Recording dataset clip · settling eye focus…"
            self.logCaptureSettings("recording delegate started")
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.15) { [weak self] in
                self?.settleAndLockCameraAfterRecordingStarts()
            }
        }
    }

    func fileOutput(_ output: AVCaptureFileOutput, didFinishRecordingTo outputFileURL: URL, from connections: [AVCaptureConnection], error: Error?) {
        DispatchQueue.main.async { [weak self] in
            self?.turnTorchOff()
            self?.cameraSettingsLockedForRecording = false
        }
        if let error {
            print("Error recording video: \(error.localizedDescription)")
            DispatchQueue.main.async { [weak self] in
                self?.showCameraFailure("Recording failed: \(error.localizedDescription)")
            }
            return
        }

        let attrs = try? FileManager.default.attributesOfItem(atPath: outputFileURL.path)
        let mb = Double((attrs?[.size] as? NSNumber)?.int64Value ?? 0) / 1_048_576
        print("=== RECORDING SAVED \(outputFileURL.lastPathComponent)  \(String(format: "%.1f", mb)) MB")

        let dims = cameraDevice.map {
            CMVideoFormatDescriptionGetDimensions($0.activeFormat.formatDescription)
        } ?? CMVideoDimensions(width: 0, height: 0)
        let capturedAt = Date()
        DispatchQueue.main.async { [weak self] in
            NotificationCenter.default.post(name: .captureProcessingStarted, object: nil)
            self?.instructionLabel.text = "Saved raw capture · rendering annotated copy…"
            self?.dismissHandler?()
        }
        videoOutputQueue.async { [self] in
            let onset = self.detectedLightOnsetVideoTimeS()
            let sidecar = CaptureSidecar(schemaVersion: 6,
                                         captureMode: self.captureMode.rawValue,
                                         requestedFps: self.captureMode.requestedFPS,
                                         activeFps: self.activeCaptureFPS,
                                         activeWidth: dims.width, activeHeight: dims.height,
                                         videoDataBufferWidth: self.videoDataBufferWidth,
                                         videoDataBufferHeight: self.videoDataBufferHeight,
                                         hdrEnabled: self.activeHDREnabled,
                                         cameraDeviceType: self.cameraDevice?.deviceType.rawValue ?? "unknown",
                                         minimumFocusDistanceMm: self.cameraDevice?.minimumFocusDistance ?? -1,
                                         torchLevel: self.provisionalTorchLevel,
                                         stabilizationMode: self.activeStabilizationMode,
                                         droppedVideoDataFrames: self.droppedVideoDataFrames,
                                         modelIdentifier: LivePupilSegmenter.modelIdentifier,
                                         pipelineIdentifier: LivePupilSegmenter.pipelineIdentifier,
                                         inferenceStride: 4,
                                         derivedLightOnsetVideoTimeS: onset,
                                         frames: self.captureFrameSamples,
                                         qualitySamples: self.captureQualitySamples,
                                         torchEvents: self.torchCaptureEvents,
                                         liveInferences: self.liveInferenceSamples)
            do {
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                encoder.keyEncodingStrategy = .convertToSnakeCase
                let sidecarURL = outputFileURL.deletingPathExtension().appendingPathExtension("capture.json")
                try encoder.encode(sidecar).write(to: sidecarURL, options: .atomic)
                print("=== CAPTURE SIDECAR SAVED \(sidecarURL.lastPathComponent)")
            } catch {
                print("Could not save capture sidecar: \(error)")
            }

            let annotationSamples = self.liveInferenceSamples.map { sample in
                VideoAnnotationSample(
                    videoTimeS: sample.videoTimeS,
                    eye: sample.eye,
                    accepted: sample.accepted,
                    normalizedROI: CGRect(
                        x: sample.roiX,
                        y: sample.roiY,
                        width: sample.roiWidth,
                        height: sample.roiHeight
                    ),
                    iris: sample.iris,
                    pupil: sample.pupil
                )
            }
            let annotatedName = outputFileURL.deletingPathExtension().lastPathComponent
                + "_annotated.mov"
            let annotatedURL = outputFileURL.deletingLastPathComponent()
                .appendingPathComponent(annotatedName)
            AnnotatedCaptureProcessor.process(
                sourceURL: outputFileURL,
                outputURL: annotatedURL,
                samples: annotationSamples,
                lightOnsetVideoTimeS: onset
            ) { [self] result in
                switch result {
                case .success(let annotated):
                    do {
                        let recording = try CaptureLibrary.save(
                            result: annotated,
                            timestamp: capturedAt
                        )
                        DispatchQueue.main.async {
                            NotificationCenter.default.post(
                                name: .newRecording,
                                object: nil,
                                userInfo: ["recording": recording]
                            )
                        }
                        self.saveVideosToPhotoLibrary([
                            outputFileURL,
                            annotated.annotatedVideoURL
                        ]) { success in
                            if success { self.removeTemporaryRawCapture(outputFileURL) }
                            DispatchQueue.main.async {
                                NotificationCenter.default.post(
                                    name: .captureProcessingFinished,
                                    object: nil
                                )
                            }
                        }
                    } catch {
                        print("Could not add annotated capture to the app library: \(error)")
                        self.saveVideosToPhotoLibrary([
                            outputFileURL,
                            annotated.annotatedVideoURL
                        ]) { success in
                            if success { self.removeTemporaryRawCapture(outputFileURL) }
                            DispatchQueue.main.async {
                                NotificationCenter.default.post(name: .captureProcessingFinished, object: nil)
                            }
                        }
                    }
                case .failure(let error):
                    print("Annotated export failed; exporting the clean capture to Photos: \(error)")
                    self.saveVideosToPhotoLibrary([outputFileURL]) { success in
                        if success { self.removeTemporaryRawCapture(outputFileURL) }
                        DispatchQueue.main.async {
                            NotificationCenter.default.post(name: .captureProcessingFinished, object: nil)
                        }
                    }
                }
            }
        }
    }
}


extension CustomCameraViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let bufferWidth = CVPixelBufferGetWidth(pixelBuffer)
        let bufferHeight = CVPixelBufferGetHeight(pixelBuffer)
        if bufferWidth != videoDataBufferWidth || bufferHeight != videoDataBufferHeight {
            videoDataBufferWidth = bufferWidth
            videoDataBufferHeight = bufferHeight
            let roiDescription = liveNormalizedROI.map {
                String(
                    format: "%.0fx%.0f px",
                    Double($0.width) * Double(bufferWidth),
                    Double($0.height) * Double(bufferHeight)
                )
            } ?? "unavailable"
            print("✓ Video-data buffer \(bufferWidth)x\(bufferHeight); model ROI \(roiDescription)")
        }
        let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        let presentationTimeS = CMTimeGetSeconds(pts)
        var videoTimeS: Double?

        if cameraOutput?.isRecording == true {
            if recordingStartPTS == nil { recordingStartPTS = pts }
            if let start = recordingStartPTS {
                let relativeTime = CMTimeGetSeconds(CMTimeSubtract(pts, start))
                videoTimeS = relativeTime
                captureFrameSamples.append(CaptureFrameSample(
                    videoTimeS: relativeTime,
                    presentationTimeS: presentationTimeS,
                    meanLuma: sampledMeanLuma(pixelBuffer)
                ))

                if let request = pendingTorchRequest {
                    torchCaptureEvents.append(TorchCaptureEvent(
                        state: request.state,
                        requestedHostTimeS: request.hostTimeS,
                        firstFrameAfterRequestPresentationTimeS: presentationTimeS,
                        firstFrameAfterRequestVideoTimeS: relativeTime
                    ))
                    if request.state == "on" {
                        DispatchQueue.main.async { [weak self] in
                            self?.lightOnsetVideoTimeS = relativeTime
                        }
                    }
                    pendingTorchRequest = nil
                }
            }
        }

        scheduleLiveInference(pixelBuffer, presentationTimeS: presentationTimeS,
                              videoTimeS: videoTimeS)
    }

    /// Sample at most every fourth 120-fps frame, and never queue a backlog. If inference is
    /// slower than 30 Hz, incoming preview frames are intentionally skipped while recording
    /// remains on its independent AVCaptureMovieFileOutput path.
    private func scheduleLiveInference(
        _ pixelBuffer: CVPixelBuffer,
        presentationTimeS: Double,
        videoTimeS: Double?
    ) {
        liveInferenceFrameCounter += 1
        guard liveInferenceFrameCounter.isMultiple(of: 4),
              !liveInferenceInFlight,
              let qualityROI = qualityNormalizedROI else { return }

        let quality = sampledEyeQuality(pixelBuffer, normalizedTopLeftROI: qualityROI)
        let focusAdjusting = cameraDevice?.isAdjustingFocus ?? false
        let exposureAdjusting = cameraDevice?.isAdjustingExposure ?? false
        if let videoTimeS {
            captureQualitySamples.append(CaptureQualitySample(
                videoTimeS: videoTimeS,
                presentationTimeS: presentationTimeS,
                meanLuma: quality.meanLuma,
                sharpness: quality.sharpness,
                saturatedFraction: quality.saturatedFraction,
                darkFraction: quality.darkFraction
            ))
        }
        guard let roi = liveNormalizedROI, let segmenter = liveSegmenter else {
            DispatchQueue.main.async { [weak self] in
                self?.updateCaptureReadiness(
                    predictions: [],
                    quality: quality,
                    focusAdjusting: focusAdjusting,
                    exposureAdjusting: exposureAdjusting
                )
            }
            return
        }
        liveInferenceInFlight = true
        inferenceQueue.async { [weak self] in
            guard let self else { return }
            do {
                let prediction = try segmenter.predict(
                    pixelBuffer: pixelBuffer,
                    normalizedTopLeftROI: roi
                )
                DispatchQueue.main.async { [weak self] in
                    guard let self else { return }
                    let displayTime = videoTimeS
                        ?? (self.cameraOutput?.isRecording == true ? nil : presentationTimeS)
                    self.drawLivePrediction(prediction, videoTimeS: displayTime)
                    self.updateCaptureReadiness(
                        predictions: [prediction],
                        quality: quality,
                        focusAdjusting: focusAdjusting,
                        exposureAdjusting: exposureAdjusting
                    )
                }
                self.videoOutputQueue.async { [weak self] in
                    guard let self else { return }
                    self.liveInferenceInFlight = false
                    self.liveInferenceFailureReported = false
                    if let videoTimeS {
                        self.liveInferenceSamples.append(LiveInferenceSample(
                            videoTimeS: videoTimeS,
                            presentationTimeS: presentationTimeS,
                            eye: "Eye",
                            roiX: Double(roi.minX),
                            roiY: Double(roi.minY),
                            roiWidth: Double(roi.width),
                            roiHeight: Double(roi.height),
                            latencyMs: prediction.latencyMs,
                            preprocessingMs: prediction.preprocessingMs,
                            modelMs: prediction.modelMs,
                            postprocessingMs: prediction.postprocessingMs,
                            accepted: prediction.accepted,
                            meanLuma: quality.meanLuma,
                            sharpness: quality.sharpness,
                            saturatedFraction: quality.saturatedFraction,
                            iris: prediction.iris,
                            pupil: prediction.pupil
                        ))
                    }
                }
            } catch {
                DispatchQueue.main.async { [weak self] in
                    self?.updateCaptureReadiness(
                        predictions: [],
                        quality: quality,
                        focusAdjusting: focusAdjusting,
                        exposureAdjusting: exposureAdjusting
                    )
                }
                self.videoOutputQueue.async { [weak self] in
                    guard let self else { return }
                    self.liveInferenceInFlight = false
                    if !self.liveInferenceFailureReported {
                        self.liveInferenceFailureReported = true
                        print("Live segmentation failed: \(error)")
                    }
                }
            }
        }
    }

    func captureOutput(_ output: AVCaptureOutput, didDrop sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        if cameraOutput?.isRecording == true { droppedVideoDataFrames += 1 }
    }

    private func sampledMeanLuma(_ pixelBuffer: CVPixelBuffer) -> Double {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0) else { return .nan }
        let width = CVPixelBufferGetWidthOfPlane(pixelBuffer, 0)
        let height = CVPixelBufferGetHeightOfPlane(pixelBuffer, 0)
        let bytesPerRow = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0)
        let pixels = base.assumingMemoryBound(to: UInt8.self)
        let stepX = max(1, width / 32), stepY = max(1, height / 32)
        var total = 0, count = 0
        for y in stride(from: 0, to: height, by: stepY) {
            for x in stride(from: 0, to: width, by: stepX) {
                total += Int(pixels[y * bytesPerRow + x]); count += 1
            }
        }
        return count > 0 ? Double(total) / Double(count) : .nan
    }

    private func detectedLightOnsetVideoTimeS() -> Double? {
        guard let bound = torchCaptureEvents.first(where: { $0.state == "on" })?
            .firstFrameAfterRequestVideoTimeS else { return nil }
        let before = captureFrameSamples.filter {
            $0.videoTimeS >= bound - 0.20 && $0.videoTimeS < bound && $0.meanLuma.isFinite
        }
        let after = captureFrameSamples.filter {
            $0.videoTimeS >= bound && $0.videoTimeS <= bound + 0.30 && $0.meanLuma.isFinite
        }
        guard before.count >= 3, after.count >= 2 else { return bound }
        let baselineValues = before.map(\.meanLuma).sorted()
        let baseline = baselineValues[baselineValues.count / 2]
        let deviations = baselineValues.map { abs($0 - baseline) }.sorted()
        let mad = deviations[deviations.count / 2]
        let threshold = baseline + max(5, 6 * mad)
        for pair in zip(after, after.dropFirst()) where
            pair.0.meanLuma >= threshold && pair.1.meanLuma >= threshold {
            return pair.0.videoTimeS
        }
        return bound
    }

    private func sampledEyeQuality(
        _ pixelBuffer: CVPixelBuffer,
        normalizedTopLeftROI roi: CGRect
    ) -> EyeFrameQuality {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0) else {
            return EyeFrameQuality(meanLuma: .nan, sharpness: 0,
                                   saturatedFraction: 1, darkFraction: 1)
        }

        let width = CVPixelBufferGetWidthOfPlane(pixelBuffer, 0)
        let height = CVPixelBufferGetHeightOfPlane(pixelBuffer, 0)
        let bytesPerRow = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0)
        let pixels = base.assumingMemoryBound(to: UInt8.self)
        let clipped = roi.standardized.intersection(CGRect(x: 0, y: 0, width: 1, height: 1))
        let x0 = max(1, min(width - 2, Int(clipped.minX * CGFloat(width))))
        let y0 = max(1, min(height - 2, Int(clipped.minY * CGFloat(height))))
        let x1 = max(x0 + 1, min(width - 1, Int(clipped.maxX * CGFloat(width))))
        let y1 = max(y0 + 1, min(height - 1, Int(clipped.maxY * CGFloat(height))))
        let step = max(1, min(x1 - x0, y1 - y0) / 96)

        var total = 0.0
        var laplacianTotal = 0.0
        var saturated = 0
        var dark = 0
        var count = 0
        for y in stride(from: y0 + step, to: y1 - step, by: step) {
            for x in stride(from: x0 + step, to: x1 - step, by: step) {
                let center = Int(pixels[y * bytesPerRow + x])
                let left = Int(pixels[y * bytesPerRow + x - step])
                let right = Int(pixels[y * bytesPerRow + x + step])
                let up = Int(pixels[(y - step) * bytesPerRow + x])
                let down = Int(pixels[(y + step) * bytesPerRow + x])
                total += Double(center)
                laplacianTotal += Double(abs(4 * center - left - right - up - down)) / 255.0
                if center >= 250 { saturated += 1 }
                if center <= 12 { dark += 1 }
                count += 1
            }
        }
        guard count > 0 else {
            return EyeFrameQuality(meanLuma: .nan, sharpness: 0,
                                   saturatedFraction: 1, darkFraction: 1)
        }
        return EyeFrameQuality(
            meanLuma: total / Double(count),
            sharpness: laplacianTotal / Double(count),
            saturatedFraction: Double(saturated) / Double(count),
            darkFraction: Double(dark) / Double(count)
        )
    }
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
            Section(header: Text("Annotated Video")) {
                VideoPlayer(player: AVPlayer(url: recording.videoURL))
                    .frame(minHeight: 360)
                    .clipShape(RoundedRectangle(cornerRadius: 10))
                Text("Experimental overlays are burned into this in-app copy. Both the clean and annotated versions are exported to Photos.")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

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
                .frame(height: 200)
            }

            Section(header: Text("Preliminary Measurements")) {
                if recording.measurements.isEmpty {
                    Text("No accepted pupil/iris fits were available for measurement.")
                        .foregroundColor(.secondary)
                }
                ForEach(Array(recording.measurements.enumerated()), id: \.offset) { _, measurement in
                    VStack(alignment: .leading, spacing: 5) {
                        Text(measurement.eye == "Eye"
                             ? "Eye"
                             : "Patient's \(measurement.eye.lowercased()) eye")
                            .font(.headline)
                        measurementRow("Median measured diameter", measurement.medianDiameterMM, unit: "mm")
                        measurementRow("Resting diameter", measurement.baselineMM, unit: "mm")
                        measurementRow("Minimum diameter", measurement.minimumMM, unit: "mm")
                        measurementRow("Constriction", measurement.percentConstriction, unit: "%", decimals: 1)
                        measurementRow("Latency", measurement.latencyS.map { $0 * 1000 }, unit: "ms", decimals: 0)
                        measurementRow(
                            "Mean constriction velocity",
                            measurement.meanConstrictionVelocityMMPerS,
                            unit: "mm/s"
                        )
                        Text("\(measurement.validSampleCount) accepted live samples")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .padding(.vertical, 4)
                }
                Text("Uses an assumed 11.7 mm visible-iris diameter. Values are experimental and not validated for clinical decisions.")
                    .font(.caption)
                    .foregroundColor(.secondary)
                Text("Date: \(dateToString(recording.timestamp))")
                Text("Time: \(timeToString(recording.timestamp))")
            }
        }
        .listStyle(InsetGroupedListStyle())
        .navigationTitle("Recording Details")
        .navigationBarTitleDisplayMode(.inline)
    }

    @ViewBuilder
    private func measurementRow(
        _ label: String,
        _ value: Double?,
        unit: String,
        decimals: Int = 2
    ) -> some View {
        if let value, value.isFinite {
            let format = decimals == 0 ? "%.0f" : decimals == 1 ? "%.1f" : "%.2f"
            Text("\(label): \(String(format: format, value)) \(unit)")
        } else {
            Text("\(label): unavailable")
                .foregroundColor(.secondary)
        }
    }
}



struct Recording: Identifiable {
    let id: UUID
    let thumbnail: UIImage
    let firstFrame: UIImage
    let lastFrame: UIImage
    let videoURL: URL
    let measurements: [EyeMeasurementSummary]
    let timestamp: Date
}

extension Notification.Name {
    static let newRecording = Notification.Name("newRecording")
    static let captureProcessingStarted = Notification.Name("captureProcessingStarted")
    static let captureProcessingFinished = Notification.Name("captureProcessingFinished")
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
