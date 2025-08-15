//
//  PECoreEmbeddingExtractor.swift
//  PE Core iOS Integration Example
//
//  Embedding extraction using PE Core models with CoreML
//

import Foundation
import CoreML
import Vision
import UIKit
import VideoToolbox

/// Extracts embeddings from images and videos using PE Core models
class PECoreEmbeddingExtractor {
    
    // MARK: - Properties
    
    private var model: VNCoreMLModel?
    private let modelName: String
    private let configuration: Configuration
    private let modelManager = PECoreModelManager.shared
    
    // MARK: - Configuration
    
    struct Configuration {
        let computeUnits: MLComputeUnits
        let inputSize: CGSize
        let normalization: ImageNormalization
        let enableBatching: Bool
        let maxBatchSize: Int
        
        struct ImageNormalization {
            let mean: [Float]
            let std: [Float]
        }
        
        static func forModel(_ modelName: String) -> Configuration {
            switch modelName {
            case "pe-core-s16-384", "pe-core-t16-384":
                return Configuration(
                    computeUnits: .cpuAndNeuralEngine,
                    inputSize: CGSize(width: 384, height: 384),
                    normalization: ImageNormalization(
                        mean: [0.485, 0.456, 0.406],
                        std: [0.229, 0.224, 0.225]
                    ),
                    enableBatching: true,
                    maxBatchSize: 8
                )
            case "pe-core-b16-224":
                return Configuration(
                    computeUnits: .cpuAndNeuralEngine,
                    inputSize: CGSize(width: 224, height: 224),
                    normalization: ImageNormalization(
                        mean: [0.485, 0.456, 0.406],
                        std: [0.229, 0.224, 0.225]
                    ),
                    enableBatching: true,
                    maxBatchSize: 4
                )
            default:
                return Configuration(
                    computeUnits: .cpuAndNeuralEngine,
                    inputSize: CGSize(width: 384, height: 384),
                    normalization: ImageNormalization(
                        mean: [0.5, 0.5, 0.5],
                        std: [0.5, 0.5, 0.5]
                    ),
                    enableBatching: false,
                    maxBatchSize: 1
                )
            }
        }
    }
    
    // MARK: - Initialization
    
    init(modelName: String) {
        self.modelName = modelName
        self.configuration = Configuration.forModel(modelName)
        
        loadModel()
    }
    
    private func loadModel() {
        modelManager.loadModel(name: modelName) { [weak self] result in
            switch result {
            case .success(let mlModel):
                do {
                    // Configure model for optimal performance
                    let config = MLModelConfiguration()
                    config.computeUnits = self?.configuration.computeUnits ?? .cpuAndNeuralEngine
                    config.allowLowPrecisionAccumulationOnGPU = true
                    
                    let coreMLModel = try VNCoreMLModel(for: mlModel, configuration: config)
                    
                    DispatchQueue.main.async {
                        self?.model = coreMLModel
                    }
                } catch {
                    print("Failed to create Vision model: \(error)")
                }
                
            case .failure(let error):
                print("Failed to load model \(modelName): \(error)")
            }
        }
    }
    
    // MARK: - Public Interface
    
    /// Extract embedding from a single image
    func extractEmbedding(from image: UIImage, completion: @escaping ([Float]?) -> Void) {
        guard let model = model else {
            completion(nil)
            return
        }
        
        guard let cgImage = preprocessImage(image) else {
            completion(nil)
            return
        }
        
        let request = VNCoreMLRequest(model: model) { request, error in
            guard error == nil else {
                completion(nil)
                return
            }
            
            let embedding = self.extractEmbeddingFromResults(request.results)
            completion(embedding)
        }
        
        // Configure request
        request.imageCropAndScaleOption = .centerCrop
        
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                try handler.perform([request])
            } catch {
                DispatchQueue.main.async {
                    completion(nil)
                }
            }
        }
    }
    
    /// Extract embeddings from multiple images in batch
    func extractEmbeddings(from images: [UIImage], completion: @escaping ([[Float]]) -> Void) {
        guard configuration.enableBatching && images.count <= configuration.maxBatchSize else {
            // Fall back to sequential processing
            extractEmbeddingsSequentially(from: images, completion: completion)
            return
        }
        
        // TODO: Implement true batch processing when CoreML supports it
        extractEmbeddingsSequentially(from: images, completion: completion)
    }
    
    /// Extract embedding from video frame
    func extractEmbedding(from videoFrame: CVPixelBuffer, completion: @escaping ([Float]?) -> Void) {
        guard let model = model else {
            completion(nil)
            return
        }
        
        let request = VNCoreMLRequest(model: model) { request, error in
            guard error == nil else {
                completion(nil)
                return
            }
            
            let embedding = self.extractEmbeddingFromResults(request.results)
            completion(embedding)
        }
        
        let handler = VNImageRequestHandler(cvPixelBuffer: videoFrame, options: [:])
        
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                try handler.perform([request])
            } catch {
                DispatchQueue.main.async {
                    completion(nil)
                }
            }
        }
    }
    
    /// Extract embeddings from video with temporal averaging
    func extractEmbedding(from videoURL: URL, 
                         frameCount: Int = 8, 
                         completion: @escaping ([Float]?) -> Void) {
        extractVideoFrames(from: videoURL, count: frameCount) { [weak self] frames in
            guard !frames.isEmpty else {
                completion(nil)
                return
            }
            
            self?.extractEmbeddingsFromFrames(frames) { embeddings in
                guard !embeddings.isEmpty else {
                    completion(nil)
                    return
                }
                
                // Average embeddings across frames
                let averagedEmbedding = self?.averageEmbeddings(embeddings)
                completion(averagedEmbedding)
            }
        }
    }
    
    /// Get model information
    func getModelInfo() -> ModelInfo {
        return ModelInfo(
            name: modelName,
            inputSize: configuration.inputSize,
            isLoaded: model != nil,
            computeUnits: configuration.computeUnits
        )
    }
}

// MARK: - Private Implementation

private extension PECoreEmbeddingExtractor {
    
    func preprocessImage(_ image: UIImage) -> CGImage? {
        guard let cgImage = image.cgImage else { return nil }
        
        // Resize to model input size
        let targetSize = configuration.inputSize
        
        UIGraphicsBeginImageContextWithOptions(targetSize, false, 1.0)
        defer { UIGraphicsEndImageContext() }
        
        let rect = CGRect(origin: .zero, size: targetSize)
        
        // Draw image with proper scaling
        let imageAspect = CGFloat(cgImage.width) / CGFloat(cgImage.height)
        let targetAspect = targetSize.width / targetSize.height
        
        var drawRect = rect
        if imageAspect > targetAspect {
            // Image is wider, fit height and center horizontally
            drawRect.size.width = targetSize.height * imageAspect
            drawRect.origin.x = (targetSize.width - drawRect.size.width) / 2
        } else {
            // Image is taller, fit width and center vertically
            drawRect.size.height = targetSize.width / imageAspect
            drawRect.origin.y = (targetSize.height - drawRect.size.height) / 2
        }
        
        UIImage(cgImage: cgImage).draw(in: drawRect)
        
        return UIGraphicsGetImageFromCurrentImageContext()?.cgImage
    }
    
    func extractEmbeddingFromResults(_ results: [VNObservation]?) -> [Float]? {
        guard let results = results as? [VNCoreMLFeatureValueObservation],
              let featureValue = results.first?.featureValue else {
            return nil
        }
        
        // Handle different output types
        switch featureValue.type {
        case .multiArray:
            guard let multiArray = featureValue.multiArrayValue else { return nil }
            return convertMultiArrayToFloat(multiArray)
            
        case .dictionary:
            // Some models output dictionaries, extract the embedding tensor
            guard let dict = featureValue.dictionaryValue,
                  let embeddingValue = dict["embedding"] ?? dict["features"] ?? dict.values.first,
                  let multiArray = embeddingValue.multiArrayValue else {
                return nil
            }
            return convertMultiArrayToFloat(multiArray)
            
        default:
            return nil
        }
    }
    
    func convertMultiArrayToFloat(_ multiArray: MLMultiArray) -> [Float] {
        let count = multiArray.count
        var floatArray = [Float](repeating: 0, count: count)
        
        // Use efficient memory access
        let dataPointer = multiArray.dataPointer.bindMemory(to: Float.self, capacity: count)
        for i in 0..<count {
            floatArray[i] = dataPointer[i]
        }
        
        // Normalize if needed
        return normalizeEmbedding(floatArray)
    }
    
    func normalizeEmbedding(_ embedding: [Float]) -> [Float] {
        // L2 normalization
        let magnitude = sqrt(embedding.reduce(0) { $0 + $1 * $1 })
        guard magnitude > 0 else { return embedding }
        
        return embedding.map { $0 / magnitude }
    }
    
    func extractEmbeddingsSequentially(from images: [UIImage], completion: @escaping ([[Float]]) -> Void) {
        var embeddings: [[Float]] = []
        let group = DispatchGroup()
        
        for image in images {
            group.enter()
            extractEmbedding(from: image) { embedding in
                if let embedding = embedding {
                    embeddings.append(embedding)
                }
                group.leave()
            }
        }
        
        group.notify(queue: .main) {
            completion(embeddings)
        }
    }
    
    func extractVideoFrames(from url: URL, count: Int, completion: @escaping ([CVPixelBuffer]) -> Void) {
        DispatchQueue.global(qos: .userInitiated).async {
            let asset = AVAsset(url: url)
            let generator = AVAssetImageGenerator(asset: asset)
            generator.appliesPreferredTrackTransform = true
            generator.requestedTimeToleranceBefore = .zero
            generator.requestedTimeToleranceAfter = .zero
            
            guard let track = asset.tracks(withMediaType: .video).first else {
                DispatchQueue.main.async { completion([]) }
                return
            }
            
            let duration = asset.duration
            let timeInterval = CMTimeGetSeconds(duration) / Double(count)
            
            var frames: [CVPixelBuffer] = []
            
            for i in 0..<count {
                let time = CMTime(seconds: Double(i) * timeInterval, preferredTimescale: 600)
                
                do {
                    let cgImage = try generator.copyCGImage(at: time, actualTime: nil)
                    
                    // Convert CGImage to CVPixelBuffer
                    if let pixelBuffer = self.createPixelBuffer(from: cgImage) {
                        frames.append(pixelBuffer)
                    }
                } catch {
                    print("Failed to extract frame at time \(time): \(error)")
                }
            }
            
            DispatchQueue.main.async {
                completion(frames)
            }
        }
    }
    
    func extractEmbeddingsFromFrames(_ frames: [CVPixelBuffer], completion: @escaping ([[Float]]) -> Void) {
        var embeddings: [[Float]] = []
        let group = DispatchGroup()
        
        for frame in frames {
            group.enter()
            extractEmbedding(from: frame) { embedding in
                if let embedding = embedding {
                    embeddings.append(embedding)
                }
                group.leave()
            }
        }
        
        group.notify(queue: .main) {
            completion(embeddings)
        }
    }
    
    func averageEmbeddings(_ embeddings: [[Float]]) -> [Float] {
        guard !embeddings.isEmpty else { return [] }
        
        let dimensions = embeddings[0].count
        var averaged = [Float](repeating: 0, count: dimensions)
        
        for embedding in embeddings {
            for i in 0..<dimensions {
                averaged[i] += embedding[i]
            }
        }
        
        let count = Float(embeddings.count)
        averaged = averaged.map { $0 / count }
        
        return normalizeEmbedding(averaged)
    }
    
    func createPixelBuffer(from cgImage: CGImage) -> CVPixelBuffer? {
        let width = cgImage.width
        let height = cgImage.height
        
        let attributes: [String: Any] = [
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true
        ]
        
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32ARGB,
            attributes as CFDictionary,
            &pixelBuffer
        )
        
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            return nil
        }
        
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        
        let context = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
        )
        
        context?.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        return buffer
    }
}

// MARK: - Supporting Types

struct ModelInfo {
    let name: String
    let inputSize: CGSize
    let isLoaded: Bool
    let computeUnits: MLComputeUnits
}

// MARK: - Extensions

extension MLComputeUnits {
    var description: String {
        switch self {
        case .cpuOnly: return "CPU Only"
        case .cpuAndGPU: return "CPU + GPU"
        case .cpuAndNeuralEngine: return "CPU + Neural Engine"
        case .all: return "All"
        default: return "Unknown"
        }
    }
}