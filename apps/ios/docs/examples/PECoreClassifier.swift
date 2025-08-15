//
//  PECoreClassifier.swift
//  PE Core iOS Integration Example
//
//  Complete implementation of dynamic classification with CDN model management
//

import Foundation
import CoreML
import Vision
import UIKit

/// Main classifier that integrates PE Core models with dynamic classification
class PECoreClassifier {
    
    // MARK: - Properties
    
    private let modelManager = PECoreModelManager.shared
    private let embeddingExtractor: PECoreEmbeddingExtractor
    private var classPrototypes: [String: [Float]] = [:]
    private var knnClassifier: KNNClassifier?
    
    // MARK: - Configuration
    
    struct Configuration {
        let modelName: String
        let useKNN: Bool
        let knnK: Int
        let confidenceThreshold: Float
        let enableCache: Bool
        
        static let `default` = Configuration(
            modelName: "pe-core-s16-384",
            useKNN: true,
            knnK: 5,
            confidenceThreshold: 0.7,
            enableCache: true
        )
    }
    
    private let config: Configuration
    
    // MARK: - Initialization
    
    init(configuration: Configuration = .default) {
        self.config = configuration
        self.embeddingExtractor = PECoreEmbeddingExtractor(modelName: configuration.modelName)
        
        if configuration.useKNN {
            self.knnClassifier = KNNClassifier()
        }
        
        setup()
    }
    
    private func setup() {
        // Load cached prototypes
        loadCachedPrototypes()
        
        // Check for updates in background
        checkForUpdates()
    }
    
    // MARK: - Public Interface
    
    /// Classify a single image
    func classify(image: UIImage, completion: @escaping (ClassificationResult) -> Void) {
        embeddingExtractor.extractEmbedding(from: image) { [weak self] embedding in
            guard let self = self, let embedding = embedding else {
                completion(ClassificationResult(className: "unknown", confidence: 0, error: .embeddingFailed))
                return
            }
            
            let result = self.classifyEmbedding(embedding)
            completion(result)
        }
    }
    
    /// Classify multiple images in batch
    func classifyBatch(images: [UIImage], completion: @escaping ([ClassificationResult]) -> Void) {
        let group = DispatchGroup()
        var results = [ClassificationResult](repeating: ClassificationResult.unknown, count: images.count)
        
        for (index, image) in images.enumerated() {
            group.enter()
            classify(image: image) { result in
                results[index] = result
                group.leave()
            }
        }
        
        group.notify(queue: .main) {
            completion(results)
        }
    }
    
    /// Add a new class with example images
    func addNewClass(name: String, examples: [UIImage], completion: @escaping (Bool) -> Void) {
        var embeddings: [[Float]] = []
        let group = DispatchGroup()
        
        for image in examples {
            group.enter()
            embeddingExtractor.extractEmbedding(from: image) { embedding in
                if let embedding = embedding {
                    embeddings.append(embedding)
                }
                group.leave()
            }
        }
        
        group.notify(queue: .main) { [weak self] in
            guard !embeddings.isEmpty else {
                completion(false)
                return
            }
            
            self?.addClassWithEmbeddings(name: name, embeddings: embeddings)
            completion(true)
        }
    }
    
    /// Update classification prototypes from server
    func updatePrototypes(from url: URL? = nil, completion: @escaping (Bool) -> Void) {
        let prototypeURL = url ?? URL(string: "https://cdn.example.com/prototypes/latest.json")!
        
        URLSession.shared.dataTask(with: prototypeURL) { [weak self] data, response, error in
            guard let data = data,
                  let prototypes = try? JSONDecoder().decode([String: [Float]].self, from: data) else {
                DispatchQueue.main.async { completion(false) }
                return
            }
            
            DispatchQueue.main.async {
                self?.classPrototypes = prototypes
                self?.savePrototypesLocally()
                self?.updateKNNClassifier()
                completion(true)
            }
        }.resume()
    }
    
    /// Get available classes
    func getAvailableClasses() -> [String] {
        return Array(classPrototypes.keys).sorted()
    }
    
    /// Get classification statistics
    func getStatistics() -> ClassificationStatistics {
        return ClassificationStatistics(
            totalClasses: classPrototypes.count,
            modelName: config.modelName,
            knnEnabled: config.useKNN,
            cacheSize: getCacheSize()
        )
    }
}

// MARK: - Private Implementation

private extension PECoreClassifier {
    
    func classifyEmbedding(_ embedding: [Float]) -> ClassificationResult {
        if config.useKNN, let knnClassifier = knnClassifier {
            return classifyWithKNN(embedding: embedding, classifier: knnClassifier)
        } else {
            return classifyWithPrototypes(embedding: embedding)
        }
    }
    
    func classifyWithPrototypes(embedding: [Float]) -> ClassificationResult {
        var bestMatch = (className: "unknown", confidence: Float(0))
        
        for (className, prototype) in classPrototypes {
            let similarity = cosineSimilarity(embedding, prototype)
            
            if similarity > bestMatch.confidence {
                bestMatch = (className, similarity)
            }
        }
        
        // Apply confidence threshold
        if bestMatch.confidence < config.confidenceThreshold {
            return ClassificationResult(className: "unknown", confidence: bestMatch.confidence, error: .lowConfidence)
        }
        
        return ClassificationResult(className: bestMatch.className, confidence: bestMatch.confidence, error: nil)
    }
    
    func classifyWithKNN(embedding: [Float], classifier: KNNClassifier) -> ClassificationResult {
        let result = classifier.classify(imageEmbedding: embedding, k: config.knnK)
        
        if result.confidence < config.confidenceThreshold {
            return ClassificationResult(className: "unknown", confidence: result.confidence, error: .lowConfidence)
        }
        
        return ClassificationResult(className: result.class, confidence: result.confidence, error: nil)
    }
    
    func addClassWithEmbeddings(name: String, embeddings: [[Float]]) {
        // Create prototype by averaging embeddings
        let prototype = createPrototype(from: embeddings)
        classPrototypes[name] = prototype
        
        // Add to KNN classifier if enabled
        if let knnClassifier = knnClassifier {
            for embedding in embeddings {
                knnClassifier.addReference(label: name, embedding: embedding)
            }
        }
        
        savePrototypesLocally()
    }
    
    func createPrototype(from embeddings: [[Float]]) -> [Float] {
        guard !embeddings.isEmpty else { return [] }
        
        let dimensions = embeddings[0].count
        var prototype = [Float](repeating: 0, count: dimensions)
        
        // Average all embeddings
        for embedding in embeddings {
            for i in 0..<dimensions {
                prototype[i] += embedding[i]
            }
        }
        
        // Divide by count and normalize
        let count = Float(embeddings.count)
        prototype = prototype.map { $0 / count }
        
        return normalizeEmbedding(prototype)
    }
    
    func updateKNNClassifier() {
        guard config.useKNN else { return }
        
        knnClassifier = KNNClassifier()
        
        // Rebuild KNN with current prototypes
        for (className, prototype) in classPrototypes {
            knnClassifier?.addReference(label: className, embedding: prototype)
        }
    }
    
    func checkForUpdates() {
        modelManager.checkForUpdates { hasUpdates in
            if hasUpdates {
                // Optionally download updates in background
                self.downloadUpdatesIfNeeded()
            }
        }
    }
    
    func downloadUpdatesIfNeeded() {
        modelManager.downloadUpdates(
            progress: { modelName, progress in
                // Optionally notify UI of download progress
                NotificationCenter.default.post(
                    name: .modelDownloadProgress,
                    object: nil,
                    userInfo: ["model": modelName, "progress": progress]
                )
            },
            completion: { result in
                switch result {
                case .success:
                    // Reload classifier with new model
                    self.reloadClassifier()
                case .failure(let error):
                    print("Model update failed: \(error)")
                }
            }
        )
    }
    
    func reloadClassifier() {
        // Reinitialize with updated model
        // This would typically require recreating the embedding extractor
        // Implementation depends on your specific model loading strategy
    }
    
    // MARK: - Storage
    
    func loadCachedPrototypes() {
        guard let data = UserDefaults.standard.data(forKey: "classPrototypes"),
              let prototypes = try? JSONDecoder().decode([String: [Float]].self, from: data) else {
            return
        }
        
        classPrototypes = prototypes
        updateKNNClassifier()
    }
    
    func savePrototypesLocally() {
        guard let data = try? JSONEncoder().encode(classPrototypes) else { return }
        UserDefaults.standard.set(data, forKey: "classPrototypes")
    }
    
    func getCacheSize() -> Int64 {
        return modelManager.getCacheSize()
    }
}

// MARK: - Supporting Types

struct ClassificationResult {
    let className: String
    let confidence: Float
    let error: ClassificationError?
    
    static let unknown = ClassificationResult(className: "unknown", confidence: 0, error: .noMatch)
}

enum ClassificationError: Error {
    case embeddingFailed
    case lowConfidence
    case noMatch
    case modelUnavailable
}

struct ClassificationStatistics {
    let totalClasses: Int
    let modelName: String
    let knnEnabled: Bool
    let cacheSize: Int64
}

// MARK: - Similarity Functions

func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
    guard a.count == b.count else { return 0 }
    
    let dotProduct = zip(a, b).reduce(0) { $0 + $1.0 * $1.1 }
    let magnitudeA = sqrt(a.reduce(0) { $0 + $1 * $1 })
    let magnitudeB = sqrt(b.reduce(0) { $0 + $1 * $1 })
    
    guard magnitudeA > 0 && magnitudeB > 0 else { return 0 }
    
    return dotProduct / (magnitudeA * magnitudeB)
}

func normalizeEmbedding(_ embedding: [Float]) -> [Float] {
    let magnitude = sqrt(embedding.reduce(0) { $0 + $1 * $1 })
    guard magnitude > 0 else { return embedding }
    return embedding.map { $0 / magnitude }
}

// MARK: - Notifications

extension Notification.Name {
    static let modelDownloadProgress = Notification.Name("modelDownloadProgress")
    static let classificationUpdated = Notification.Name("classificationUpdated")
}