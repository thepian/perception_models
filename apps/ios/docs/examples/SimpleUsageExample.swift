//
//  SimpleUsageExample.swift
//  PE Core iOS Integration Example
//
//  Simple examples showing how to use PE Core for classification
//

import UIKit
import CoreML

/// Simple examples demonstrating PE Core usage
class SimpleUsageExample {
    
    // MARK: - Basic Classification Example
    
    func basicClassificationExample() {
        // Initialize the classifier
        let classifier = PECoreClassifier()
        
        // Load an image
        guard let image = UIImage(named: "example_image") else { return }
        
        // Classify the image
        classifier.classify(image: image) { result in
            print("Predicted class: \(result.className)")
            print("Confidence: \(result.confidence)")
            
            if let error = result.error {
                print("Error: \(error)")
            }
        }
    }
    
    // MARK: - Batch Classification Example
    
    func batchClassificationExample() {
        let classifier = PECoreClassifier()
        
        // Prepare multiple images
        let imageNames = ["dog1", "cat1", "car1", "tree1"]
        let images = imageNames.compactMap { UIImage(named: $0) }
        
        // Classify all images at once
        classifier.classifyBatch(images: images) { results in
            for (index, result) in results.enumerated() {
                print("Image \(index): \(result.className) (\(result.confidence))")
            }
        }
    }
    
    // MARK: - Adding Custom Classes Example
    
    func addCustomClassExample() {
        let classifier = PECoreClassifier()
        
        // Prepare example images for a new class
        let exampleImages = [
            UIImage(named: "my_object_1")!,
            UIImage(named: "my_object_2")!,
            UIImage(named: "my_object_3")!
        ]
        
        // Add the new class
        classifier.addNewClass(name: "my_custom_object", examples: exampleImages) { success in
            if success {
                print("Successfully added custom class!")
                
                // Now test with a new image
                if let testImage = UIImage(named: "test_image") {
                    classifier.classify(image: testImage) { result in
                        print("Classification result: \(result.className)")
                    }
                }
            } else {
                print("Failed to add custom class")
            }
        }
    }
    
    // MARK: - Updating Classifications from Server
    
    func updateClassificationsExample() {
        let classifier = PECoreClassifier()
        
        // Update prototypes from server
        classifier.updatePrototypes() { success in
            if success {
                print("Classifications updated successfully!")
                
                // Get the updated list of available classes
                let classes = classifier.getAvailableClasses()
                print("Available classes: \(classes)")
            } else {
                print("Failed to update classifications")
            }
        }
    }
    
    // MARK: - Camera Integration Example
    
    func cameraIntegrationExample() {
        // This would typically be in a view controller with camera access
        let classifier = PECoreClassifier()
        
        // Simulate getting image from camera
        func processImageFromCamera(_ image: UIImage) {
            classifier.classify(image: image) { result in
                DispatchQueue.main.async {
                    // Update UI with classification result
                    self.updateClassificationUI(result: result)
                }
            }
        }
        
        // In a real app, you would set up AVCaptureSession here
        // and call processImageFromCamera in the capture delegate
    }
    
    private func updateClassificationUI(result: ClassificationResult) {
        // Update your UI elements here
        print("UI Update: \(result.className) with confidence \(result.confidence)")
    }
    
    // MARK: - Performance Monitoring Example
    
    func performanceMonitoringExample() {
        let classifier = PECoreClassifier()
        
        // Get classifier statistics
        let stats = classifier.getStatistics()
        print("Model: \(stats.modelName)")
        print("Total classes: \(stats.totalClasses)")
        print("KNN enabled: \(stats.knnEnabled)")
        print("Cache size: \(stats.cacheSize) bytes")
        
        // Measure classification time
        guard let testImage = UIImage(named: "test_image") else { return }
        
        let startTime = CFAbsoluteTimeGetCurrent()
        
        classifier.classify(image: testImage) { result in
            let endTime = CFAbsoluteTimeGetCurrent()
            let timeElapsed = endTime - startTime
            
            print("Classification took \(timeElapsed * 1000) ms")
            print("Result: \(result.className) (\(result.confidence))")
        }
    }
    
    // MARK: - Error Handling Example
    
    func errorHandlingExample() {
        let classifier = PECoreClassifier()
        
        guard let image = UIImage(named: "test_image") else {
            print("Could not load test image")
            return
        }
        
        classifier.classify(image: image) { result in
            switch result.error {
            case nil:
                // Success
                print("Successfully classified as: \(result.className)")
                
            case .embeddingFailed:
                print("Failed to extract embedding from image")
                // Might retry with different image preprocessing
                
            case .lowConfidence:
                print("Low confidence classification: \(result.className)")
                // Might ask user for confirmation or show alternatives
                
            case .noMatch:
                print("No matching class found")
                // Might suggest adding this as a new class
                
            case .modelUnavailable:
                print("Model not available, using fallback")
                // Implement fallback classification or download model
                
            case .some(let error):
                print("Unexpected error: \(error)")
            }
        }
    }
    
    // MARK: - Real-time Classification Example
    
    func realtimeClassificationExample() {
        let classifier = PECoreClassifier()
        var isProcessing = false
        
        // Simulate getting frames from camera
        func processVideoFrame(_ frame: UIImage) {
            // Avoid processing if already busy
            guard !isProcessing else { return }
            
            isProcessing = true
            
            classifier.classify(image: frame) { result in
                defer { isProcessing = false }
                
                // Only show high-confidence results for real-time
                if result.confidence > 0.8 {
                    DispatchQueue.main.async {
                        // Update UI with confident classification
                        print("Real-time: \(result.className)")
                    }
                }
            }
        }
        
        // In a real app, this would be called from AVCaptureVideoDataOutputSampleBufferDelegate
        // You might also want to throttle the frame rate (e.g., process every 3rd frame)
    }
    
    // MARK: - Configuration Example
    
    func configurationExample() {
        // Create classifier with custom configuration
        let config = PECoreClassifier.Configuration(
            modelName: "pe-core-t16-384",  // Use smaller model for speed
            useKNN: true,
            knnK: 3,                       // Use 3 nearest neighbors
            confidenceThreshold: 0.6,      // Lower threshold for more results
            enableCache: true
        )
        
        let classifier = PECoreClassifier(configuration: config)
        
        // Use configured classifier
        guard let image = UIImage(named: "test_image") else { return }
        
        classifier.classify(image: image) { result in
            print("Custom configured result: \(result.className)")
        }
    }
}

// MARK: - View Controller Integration Example

class ClassificationViewController: UIViewController {
    
    @IBOutlet weak var imageView: UIImageView!
    @IBOutlet weak var resultLabel: UILabel!
    @IBOutlet weak var confidenceLabel: UILabel!
    @IBOutlet weak var classifyButton: UIButton!
    
    private let classifier = PECoreClassifier()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupUI()
    }
    
    private func setupUI() {
        classifyButton.addTarget(self, action: #selector(classifyButtonTapped), for: .touchUpInside)
        
        // Listen for model updates
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(modelUpdated),
            name: .classificationUpdated,
            object: nil
        )
    }
    
    @objc private func classifyButtonTapped() {
        guard let image = imageView.image else { return }
        
        // Show loading state
        classifyButton.isEnabled = false
        resultLabel.text = "Classifying..."
        
        classifier.classify(image: image) { [weak self] result in
            DispatchQueue.main.async {
                self?.classifyButton.isEnabled = true
                self?.updateResults(result)
            }
        }
    }
    
    private func updateResults(_ result: ClassificationResult) {
        resultLabel.text = result.className.capitalized
        confidenceLabel.text = String(format: "Confidence: %.1f%%", result.confidence * 100)
        
        // Color code by confidence
        switch result.confidence {
        case 0.8...:
            resultLabel.textColor = .systemGreen
        case 0.6..<0.8:
            resultLabel.textColor = .systemOrange
        default:
            resultLabel.textColor = .systemRed
        }
    }
    
    @objc private func modelUpdated() {
        // Optionally show a notification that new classifications are available
        let alert = UIAlertController(
            title: "Classifications Updated",
            message: "New categories are now available for classification.",
            preferredStyle: .alert
        )
        alert.addAction(UIAlertAction(title: "OK", style: .default))
        present(alert, animated: true)
    }
    
    // MARK: - Image Picker Integration
    
    @IBAction func selectImage() {
        let picker = UIImagePickerController()
        picker.delegate = self
        picker.sourceType = .photoLibrary
        present(picker, animated: true)
    }
}

// MARK: - Image Picker Delegate

extension ClassificationViewController: UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        
        if let image = info[.originalImage] as? UIImage {
            imageView.image = image
            
            // Automatically classify when image is selected
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                self.classifyButtonTapped()
            }
        }
        
        picker.dismiss(animated: true)
    }
}