//
//  ContentView.swift
//  PupilDetection1
//
//  Created by Daniel Garcia on 1/31/24.
//

import SwiftUI
import AVFoundation // Import AVFoundation for flashlight/torch control
import Photos // Import Photos framework for saving videos
import Vision

struct ContentView: View {
    @State private var isShowingCamera = false
    
    var body: some View {
        NavigationView {
            VStack {
                Spacer()
                Text("👁️ Pupillometer")
                    .font(.title)
                    .padding()
                Text("Measure pupil dilation and constriction.")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                Spacer()
                Button(action: {
                    // Toggle camera view
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
            .padding()
            .fullScreenCover(isPresented: $isShowingCamera, content: CustomCameraView.init)
        }
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
        let symbolConfig = UIImage.SymbolConfiguration(pointSize: 18, weight: .medium, scale: .medium)
        button.setImage(UIImage(systemName: "xmark.circle.fill", withConfiguration: symbolConfig), for: .normal)
        button.tintColor = .white
        button.translatesAutoresizingMaskIntoConstraints = false
        return button
    }()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupCaptureSession()
        setupUIComponents()
        drawBoundingBox()
    }
    
    private func setupUIComponents() {
        setupBars()
        setupTorchButton()
        setupRecordButton()
        setupDismissButton()
        setupInstructionLabel()
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
    
    private func setupCaptureSession() {
        captureSession = AVCaptureSession()
        guard let captureSession = captureSession else { return }
        
        captureSession.beginConfiguration()
        guard let videoDevice = AVCaptureDevice.default(for: .video) else { return }
        
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
            let symbolConfig = UIImage.SymbolConfiguration(pointSize: 50, weight: .regular, scale: .large)
            let stopImage = UIImage(systemName: "record.circle", withConfiguration: symbolConfig)
            recordButton.setImage(stopImage, for: .normal)
            recordButton.tintColor = .red
        } else {
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
    
    private var boundingBoxLayer: CAShapeLayer?
    
    private func drawBoundingBox() {
        boundingBoxLayer?.removeFromSuperlayer()
        
        // Define the size of the bounding box
        let boxWidth: CGFloat = 200 // Example width
        let boxHeight: CGFloat = 200 // Example height
        
        // Calculate the starting point for the bounding box to be centered
        let startX = (view.bounds.width - boxWidth) / 2
        let startY = (view.bounds.height - boxHeight) / 2
        
        // Create the bounding box rectangle
        let boundingBoxRect = CGRect(x: startX, y: startY, width: boxWidth, height: boxHeight)
        
        // Create and configure the shape layer for the bounding box
        let shapeLayer = CAShapeLayer()
        shapeLayer.frame = boundingBoxRect
        shapeLayer.borderColor = UIColor.red.cgColor
        shapeLayer.borderWidth = 2
        shapeLayer.cornerRadius = 10 // Optional, for rounded corners
        view.layer.addSublayer(shapeLayer)
        
        // Keep a reference to remove it later
        boundingBoxLayer = shapeLayer
    }

}

extension CustomCameraViewController: AVCaptureFileOutputRecordingDelegate {
    func fileOutput(_ output: AVCaptureFileOutput, didFinishRecordingTo outputFileURL: URL, from connections: [AVCaptureConnection], error: Error?) {
        // Handle video saving to camera roll here
        PHPhotoLibrary.shared().performChanges({
            PHAssetChangeRequest.creationRequestForAssetFromVideo(atFileURL: outputFileURL)
        }) { saved, error in
            if saved {
                DispatchQueue.main.async {
                    let alert = UIAlertController(title: "Scan Complete", message: "Your video has been saved to Photos and is ready to use for pupil examination.", preferredStyle: .alert)
                    alert.addAction(UIAlertAction(title: "OK", style: .default))
                    self.present(alert, animated: true)
                }
            }
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
        let placeholderBoundingBox = CGRect(x: 0.25, y: 0.25, width: 0.25, height: 0.25) // Placeholder values
        DispatchQueue.main.async { [weak self] in
            self?.drawBoundingBox()
        }
    }
    
    // Assuming drawBoundingBox is correctly implemented to draw on the UIView coordinate system.
}



struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
