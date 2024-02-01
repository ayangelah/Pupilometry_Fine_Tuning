//
//  ContentView.swift
//  PupilDetection1
//
//  Created by Daniel Garcia on 1/31/24.
//

import SwiftUI
import UIKit
import AVFoundation // Import AVFoundation for flashlight/torch control
import Photos // Import Photos framework for saving videos
import Vision

struct ContentView: View {
    @State private var isShowingCamera = false
    
    var body: some View {
        VStack {
            Text("👁️") // Eye emoji
                .font(.system(size: 100)) // Set font size to extra large
                .foregroundStyle(.tint)
            Text("Pupil Detection Attempt")
            Button(action: {
                // Button action added here
                isShowingCamera.toggle()
                print("Button tapped!")
            }) {
                Text("Access Camera")
                    .padding()
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(8)
            }
        }
        .padding()
        .sheet(isPresented: $isShowingCamera) {
            CameraView()
        }
    }
}

struct CameraView: UIViewControllerRepresentable {
    typealias UIViewControllerType = UIImagePickerController
    
    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.sourceType = .camera
        picker.cameraDevice = .rear // Set to front camera
        picker.mediaTypes = ["public.movie"] // Specify video media type
        picker.cameraCaptureMode = .video // Set capture mode to video
        picker.delegate = context.coordinator // Set delegate
        return picker
        
        
    }
    
    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {
        // Update
    }
    
    func makeCoordinator() -> Coordinator {
        Coordinator()
    }
    
    class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
        var isFlashlightOn = false
        
        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
            guard let url = info[.mediaURL] as? URL else {
                print("Error: Could not get video URL")
                return
            }
            
            // Saves the video to the photo library
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
