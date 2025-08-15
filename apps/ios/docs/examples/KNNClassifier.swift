//
//  KNNClassifier.swift
//  PE Core iOS Integration Example
//
//  K-Nearest Neighbors implementation for embedding-based classification
//

import Foundation

/// K-Nearest Neighbors classifier for PE Core embeddings
class KNNClassifier {
    
    // MARK: - Types
    
    struct ReferenceEmbedding {
        let id: UUID
        let label: String
        let embedding: [Float]
        let metadata: [String: Any]?
        let timestamp: Date
        
        init(label: String, embedding: [Float], metadata: [String: Any]? = nil) {
            self.id = UUID()
            self.label = label
            self.embedding = embedding
            self.metadata = metadata
            self.timestamp = Date()
        }
    }
    
    struct ClassificationResult {
        let `class`: String
        let confidence: Float
        let neighbors: [Neighbor]
        
        struct Neighbor {
            let label: String
            let distance: Float
            let id: UUID
        }
    }
    
    // MARK: - Properties
    
    private var references: [ReferenceEmbedding] = []
    private let maxReferencesPerClass: Int
    private let distanceMetric: DistanceMetric
    
    enum DistanceMetric {
        case euclidean
        case cosine
        case manhattan
    }
    
    // MARK: - Initialization
    
    init(maxReferencesPerClass: Int = 50, distanceMetric: DistanceMetric = .cosine) {
        self.maxReferencesPerClass = maxReferencesPerClass
        self.distanceMetric = distanceMetric
    }
    
    // MARK: - Public Interface
    
    /// Add a reference embedding for training
    func addReference(label: String, embedding: [Float], metadata: [String: Any]? = nil) {
        let reference = ReferenceEmbedding(label: label, embedding: embedding, metadata: metadata)
        references.append(reference)
        
        // Limit references per class to prevent memory issues
        pruneReferencesIfNeeded(for: label)
    }
    
    /// Add multiple references at once
    func addReferences(label: String, embeddings: [[Float]], metadata: [String: Any]? = nil) {
        for embedding in embeddings {
            addReference(label: label, embedding: embedding, metadata: metadata)
        }
    }
    
    /// Classify an embedding using KNN
    func classify(imageEmbedding: [Float], k: Int = 5) -> ClassificationResult {
        guard !references.isEmpty else {
            return ClassificationResult(class: "unknown", confidence: 0, neighbors: [])
        }
        
        // Calculate distances to all references
        let distances = references.map { reference in
            let distance = calculateDistance(imageEmbedding, reference.embedding)
            return (reference: reference, distance: distance)
        }
        
        // Get k nearest neighbors
        let kNearest = distances
            .sorted { $0.distance < $1.distance }
            .prefix(k)
        
        // Convert to neighbors for result
        let neighbors = kNearest.map { 
            ClassificationResult.Neighbor(
                label: $0.reference.label, 
                distance: $0.distance,
                id: $0.reference.id
            )
        }
        
        // Vote on class with distance weighting
        let prediction = weightedVote(neighbors: Array(kNearest))
        
        return ClassificationResult(
            class: prediction.class,
            confidence: prediction.confidence,
            neighbors: neighbors
        )
    }
    
    /// Remove all references for a specific class
    func removeClass(label: String) {
        references.removeAll { $0.label == label }
    }
    
    /// Remove a specific reference by ID
    func removeReference(id: UUID) {
        references.removeAll { $0.id == id }
    }
    
    /// Get all available classes
    func getClasses() -> [String] {
        return Array(Set(references.map { $0.label })).sorted()
    }
    
    /// Get statistics about the classifier
    func getStatistics() -> KNNStatistics {
        let classCounts = Dictionary(grouping: references, by: { $0.label })
            .mapValues { $0.count }
        
        return KNNStatistics(
            totalReferences: references.count,
            uniqueClasses: classCounts.count,
            classCounts: classCounts,
            distanceMetric: distanceMetric
        )
    }
    
    /// Clear all references
    func clear() {
        references.removeAll()
    }
    
    /// Save references to disk
    func save(to url: URL) throws {
        let data = try JSONEncoder().encode(references.map(ReferenceData.init))
        try data.write(to: url)
    }
    
    /// Load references from disk
    func load(from url: URL) throws {
        let data = try Data(contentsOf: url)
        let referenceData = try JSONDecoder().decode([ReferenceData].self, from: data)
        references = referenceData.map(ReferenceEmbedding.init)
    }
}

// MARK: - Private Implementation

private extension KNNClassifier {
    
    func calculateDistance(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count else { return Float.infinity }
        
        switch distanceMetric {
        case .euclidean:
            return euclideanDistance(a, b)
        case .cosine:
            return 1.0 - cosineSimilarity(a, b) // Convert similarity to distance
        case .manhattan:
            return manhattanDistance(a, b)
        }
    }
    
    func euclideanDistance(_ a: [Float], _ b: [Float]) -> Float {
        return sqrt(zip(a, b).reduce(0) { $0 + pow($1.0 - $1.1, 2) })
    }
    
    func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        let dotProduct = zip(a, b).reduce(0) { $0 + $1.0 * $1.1 }
        let magnitudeA = sqrt(a.reduce(0) { $0 + $1 * $1 })
        let magnitudeB = sqrt(b.reduce(0) { $0 + $1 * $1 })
        
        guard magnitudeA > 0 && magnitudeB > 0 else { return 0 }
        return dotProduct / (magnitudeA * magnitudeB)
    }
    
    func manhattanDistance(_ a: [Float], _ b: [Float]) -> Float {
        return zip(a, b).reduce(0) { $0 + abs($1.0 - $1.1) }
    }
    
    func weightedVote(neighbors: [(reference: ReferenceEmbedding, distance: Float)]) -> (class: String, confidence: Float) {
        guard !neighbors.isEmpty else { return ("unknown", 0) }
        
        // Group neighbors by class
        let classGroups = Dictionary(grouping: neighbors, by: { $0.reference.label })
        
        // Calculate weighted votes (inverse distance weighting)
        var classScores: [String: Float] = [:]
        
        for (className, classNeighbors) in classGroups {
            let weights = classNeighbors.map { neighbor in
                // Add small epsilon to avoid division by zero
                1.0 / (neighbor.distance + 0.001)
            }
            classScores[className] = weights.reduce(0, +)
        }
        
        // Find class with highest score
        guard let bestClass = classScores.max(by: { $0.value < $1.value }) else {
            return ("unknown", 0)
        }
        
        // Calculate confidence as ratio of best score to total score
        let totalScore = classScores.values.reduce(0, +)
        let confidence = totalScore > 0 ? bestClass.value / totalScore : 0
        
        return (bestClass.key, confidence)
    }
    
    func pruneReferencesIfNeeded(for label: String) {
        let classReferences = references.filter { $0.label == label }
        
        if classReferences.count > maxReferencesPerClass {
            // Remove oldest references for this class
            let sortedByDate = classReferences.sorted { $0.timestamp < $1.timestamp }
            let toRemove = sortedByDate.prefix(classReferences.count - maxReferencesPerClass)
            
            for reference in toRemove {
                references.removeAll { $0.id == reference.id }
            }
        }
    }
}

// MARK: - Supporting Types

struct KNNStatistics {
    let totalReferences: Int
    let uniqueClasses: Int
    let classCounts: [String: Int]
    let distanceMetric: KNNClassifier.DistanceMetric
}

// MARK: - Persistence

private struct ReferenceData: Codable {
    let id: UUID
    let label: String
    let embedding: [Float]
    let timestamp: Date
    
    init(from reference: KNNClassifier.ReferenceEmbedding) {
        self.id = reference.id
        self.label = reference.label
        self.embedding = reference.embedding
        self.timestamp = reference.timestamp
    }
}

private extension KNNClassifier.ReferenceEmbedding {
    init(from data: ReferenceData) {
        self.id = data.id
        self.label = data.label
        self.embedding = data.embedding
        self.metadata = nil
        self.timestamp = data.timestamp
    }
}

// MARK: - Advanced Features

extension KNNClassifier {
    
    /// Find similar references to a given embedding
    func findSimilar(to embedding: [Float], count: Int = 10) -> [ReferenceEmbedding] {
        let distances = references.map { reference in
            (reference: reference, distance: calculateDistance(embedding, reference.embedding))
        }
        
        return distances
            .sorted { $0.distance < $1.distance }
            .prefix(count)
            .map { $0.reference }
    }
    
    /// Analyze class separability
    func analyzeClassSeparability() -> [String: Float] {
        let classes = getClasses()
        var separability: [String: Float] = [:]
        
        for className in classes {
            let classRefs = references.filter { $0.label == className }
            let otherRefs = references.filter { $0.label != className }
            
            guard !classRefs.isEmpty && !otherRefs.isEmpty else { continue }
            
            // Calculate average intra-class distance
            var intraDistances: [Float] = []
            for i in 0..<classRefs.count {
                for j in (i+1)..<classRefs.count {
                    let distance = calculateDistance(classRefs[i].embedding, classRefs[j].embedding)
                    intraDistances.append(distance)
                }
            }
            
            // Calculate average inter-class distance
            var interDistances: [Float] = []
            for classRef in classRefs {
                for otherRef in otherRefs.prefix(min(50, otherRefs.count)) { // Sample to avoid O(n²)
                    let distance = calculateDistance(classRef.embedding, otherRef.embedding)
                    interDistances.append(distance)
                }
            }
            
            let avgIntra = intraDistances.isEmpty ? 0 : intraDistances.reduce(0, +) / Float(intraDistances.count)
            let avgInter = interDistances.isEmpty ? 0 : interDistances.reduce(0, +) / Float(interDistances.count)
            
            // Separability ratio (higher is better)
            separability[className] = avgInter / max(avgIntra, 0.001)
        }
        
        return separability
    }
    
    /// Remove outlier references
    func removeOutliers(threshold: Float = 2.0) {
        let classes = getClasses()
        
        for className in classes {
            let classRefs = references.filter { $0.label == className }
            guard classRefs.count > 3 else { continue } // Need minimum samples
            
            // Calculate centroid
            let dimensions = classRefs[0].embedding.count
            var centroid = [Float](repeating: 0, count: dimensions)
            
            for ref in classRefs {
                for i in 0..<dimensions {
                    centroid[i] += ref.embedding[i]
                }
            }
            centroid = centroid.map { $0 / Float(classRefs.count) }
            
            // Calculate distances to centroid
            let distances = classRefs.map { ref in
                (ref: ref, distance: calculateDistance(ref.embedding, centroid))
            }
            
            // Calculate mean and standard deviation
            let meanDistance = distances.reduce(0) { $0 + $1.distance } / Float(distances.count)
            let variance = distances.reduce(0) { $0 + pow($1.distance - meanDistance, 2) } / Float(distances.count)
            let stdDev = sqrt(variance)
            
            // Remove outliers beyond threshold
            let outliers = distances.filter { $0.distance > meanDistance + threshold * stdDev }
            for outlier in outliers {
                removeReference(id: outlier.ref.id)
            }
        }
    }
}