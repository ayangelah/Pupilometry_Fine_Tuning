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
                Text("👁️ Pupilometry")
                    .font(.title)
                    .padding()
                Text("Measure pupil dilation and constriction")
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
            .sheet(isPresented: $isShowingCamera) {
                CameraView()
            }
        }
    }
}

struct CameraView: UIViewControllerRepresentable {
    typealias UIViewControllerType = UIImagePickerController
    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.sourceType = .camera
        picker.cameraDevice = .rear
        picker.mediaTypes = ["public.movie"]
        picker.cameraCaptureMode = .video
        picker.delegate = context.coordinator
        return picker
    }
    
    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {
        // Intentionally left blank for future updates
    }
    
    func makeCoordinator() -> Coordinator {
        Coordinator()
    }
    
    class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
            guard let url = info[.mediaURL] as? URL else {
                print("Error: Could not get video URL")
                return
            }
            
            PHPhotoLibrary.shared().performChanges {
                PHAssetChangeRequest.creationRequestForAssetFromVideo(atFileURL: url)
            } completionHandler: { success, error in
                if let error = error {
                    print("Error saving video: \(error)")
                } else {
                    print("Video saved successfully!")
                }
            }

            picker.dismiss(animated: true, completion: nil)
        }
        
        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            picker.dismiss(animated: true, completion: nil)
        }
    }
}


struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
