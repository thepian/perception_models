# Dynamic Classification with PE Core

This document explains how to implement flexible, updateable classification in iOS apps using PE Core models.

## Understanding PE Core Classification

PE Core models output dense feature embeddings (512-1280 dimensions) rather than classification scores. This enables:

- **Zero-shot classification**: Define new classes without retraining
- **Dynamic class updates**: Modify classifications without app updates
- **Fine-grained distinctions**: Separate highly similar classes
- **Multi-modal understanding**: Leverage vision-language alignment

## Classification Approaches

### 1. Prototype-Based Classification

The most straightforward approach compares image embeddings with class prototype embeddings:

```swift
struct ClassPrototype {
    let name: String
    let embedding: [Float]
    let threshold: Float? // Optional confidence threshold
}

class PrototypeClassifier {
    private var prototypes: [ClassPrototype] = []
    
    func classify(imageEmbedding: [Float]) -> (class: String, confidence: Float) {
        var bestMatch = (class: "unknown", confidence: Float(0))
        
        for prototype in prototypes {
            let similarity = cosineSimilarity(imageEmbedding, prototype.embedding)
            
            if similarity > bestMatch.confidence {
                bestMatch = (prototype.name, similarity)
            }
        }
        
        return bestMatch
    }
}
```

### 2. K-Nearest Neighbors (KNN)

For more robust classification, use KNN with multiple reference embeddings per class:

```swift
class KNNClassifier {
    struct ReferenceEmbedding {
        let label: String
        let embedding: [Float]
        let metadata: [String: Any]? // Optional context
    }
    
    private var references: [ReferenceEmbedding] = []
    
    func classify(imageEmbedding: [Float], k: Int = 5) -> (class: String, confidence: Float) {
        // Calculate distances to all references
        let distances = references.map { reference in
            (label: reference.label, 
             distance: euclideanDistance(imageEmbedding, reference.embedding))
        }
        
        // Get k nearest neighbors
        let kNearest = distances
            .sorted { $0.distance < $1.distance }
            .prefix(k)
        
        // Vote on class
        let votes = Dictionary(grouping: kNearest, by: { $0.label })
            .mapValues { $0.count }
        
        let topClass = votes.max { $0.value < $1.value }
        let confidence = Float(topClass?.value ?? 0) / Float(k)
        
        return (topClass?.key ?? "unknown", confidence)
    }
    
    // Add new reference embeddings
    func addReference(label: String, embedding: [Float], metadata: [String: Any]? = nil) {
        references.append(ReferenceEmbedding(
            label: label, 
            embedding: embedding, 
            metadata: metadata
        ))
    }
}
```

### 3. Hierarchical Classification

Combine multiple classifiers for coarse-to-fine classification:

```swift
class HierarchicalClassifier {
    struct ClassHierarchy {
        let coarseClassifier: PrototypeClassifier
        let fineClassifiers: [String: PrototypeClassifier]
    }
    
    private let hierarchy: ClassHierarchy
    
    func classify(imageEmbedding: [Float]) -> (coarse: String, fine: String, confidence: Float) {
        // First level: coarse classification
        let coarseResult = hierarchy.coarseClassifier.classify(imageEmbedding: imageEmbedding)
        
        // Second level: fine classification within coarse category
        if let fineClassifier = hierarchy.fineClassifiers[coarseResult.class] {
            let fineResult = fineClassifier.classify(imageEmbedding: imageEmbedding)
            return (coarseResult.class, fineResult.class, fineResult.confidence)
        }
        
        return (coarseResult.class, "general", coarseResult.confidence)
    }
}
```

## Embedding Extraction

Extract embeddings from images using PE Core models:

```swift
import CoreML
import Vision

class PECoreEmbeddingExtractor {
    private let model: VNCoreMLModel
    
    init(modelName: String = "pe_core_s16_384_fp16") throws {
        let mlModel = try MLModel(contentsOf: modelURL)
        self.model = try VNCoreMLModel(for: mlModel)
    }
    
    func extractEmbedding(from image: UIImage, completion: @escaping ([Float]?) -> Void) {
        guard let cgImage = image.cgImage else {
            completion(nil)
            return
        }
        
        let request = VNCoreMLRequest(model: model) { request, error in
            guard error == nil,
                  let results = request.results as? [VNCoreMLFeatureValueObservation],
                  let embedding = results.first?.featureValue.multiArrayValue else {
                completion(nil)
                return
            }
            
            // Convert MLMultiArray to [Float]
            let floatArray = (0..<embedding.count).map { 
                Float(embedding[$0].floatValue) 
            }
            
            // Normalize embedding
            let normalized = self.normalizeEmbedding(floatArray)
            completion(normalized)
        }
        
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        try? handler.perform([request])
    }
    
    private func normalizeEmbedding(_ embedding: [Float]) -> [Float] {
        let magnitude = sqrt(embedding.reduce(0) { $0 + $1 * $1 })
        return embedding.map { $0 / magnitude }
    }
}
```

## Similarity Metrics

### Cosine Similarity (Recommended)

Best for normalized embeddings:

```swift
func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
    guard a.count == b.count else { return 0 }
    
    let dotProduct = zip(a, b).reduce(0) { $0 + $1.0 * $1.1 }
    let magnitudeA = sqrt(a.reduce(0) { $0 + $1 * $1 })
    let magnitudeB = sqrt(b.reduce(0) { $0 + $1 * $1 })
    
    return dotProduct / (magnitudeA * magnitudeB)
}
```

### Euclidean Distance

For KNN and clustering:

```swift
func euclideanDistance(_ a: [Float], _ b: [Float]) -> Float {
    guard a.count == b.count else { return Float.infinity }
    
    return sqrt(zip(a, b).reduce(0) { $0 + pow($1.0 - $1.1, 2) })
}
```

## Creating Class Prototypes

### From Multiple Examples

Average embeddings from multiple examples for robust prototypes:

```swift
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
    
    // Divide by count
    let count = Float(embeddings.count)
    prototype = prototype.map { $0 / count }
    
    // Normalize
    let magnitude = sqrt(prototype.reduce(0) { $0 + $1 * $1 })
    return prototype.map { $0 / magnitude }
}
```

### From Text Descriptions

If you have the text encoder model:

```swift
func createPrototypeFromText(descriptions: [String]) -> [Float] {
    // This requires the text encoder part of PE Core
    // Usually done server-side and distributed to apps
    let textEmbeddings = descriptions.map { encodeText($0) }
    return createPrototype(from: textEmbeddings)
}
```

## Performance Optimization

### 1. Batch Processing

Process multiple images efficiently:

```swift
func classifyBatch(images: [UIImage], completion: @escaping ([(String, Float)]) -> Void) {
    let group = DispatchGroup()
    var results = [(String, Float)](repeating: ("unknown", 0), count: images.count)
    
    for (index, image) in images.enumerated() {
        group.enter()
        extractEmbedding(from: image) { embedding in
            if let embedding = embedding {
                results[index] = self.classify(imageEmbedding: embedding)
            }
            group.leave()
        }
    }
    
    group.notify(queue: .main) {
        completion(results)
    }
}
```

### 2. Embedding Cache

Cache frequently used embeddings:

```swift
class EmbeddingCache {
    private let cache = NSCache<NSString, NSArray>()
    
    func embedding(for imageKey: String) -> [Float]? {
        return cache.object(forKey: imageKey as NSString) as? [Float]
    }
    
    func store(embedding: [Float], for imageKey: String) {
        cache.setObject(embedding as NSArray, forKey: imageKey as NSString)
    }
}
```

### 3. Quantized Prototypes

Reduce memory usage with quantized prototypes:

```swift
struct QuantizedPrototype {
    let name: String
    let embedding: [Int8] // Quantized to 8-bit
    let scale: Float
    
    func similarity(with embedding: [Float]) -> Float {
        // Dequantize and compute similarity
        let dequantized = self.embedding.map { Float($0) * scale }
        return cosineSimilarity(embedding, dequantized)
    }
}
```

## Next Steps

- See [Model Management](model-management.md) for CDN-based model updates
- Check [Examples](examples/) for complete implementation samples
- Review [Best Practices](best-practices.md) for production deployment