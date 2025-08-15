# Best Practices for PE Core iOS Implementation

This document outlines best practices for deploying PE Core models in production iOS applications, covering performance optimization, user experience, and operational considerations.

## Model Selection and Performance

### 1. Choose the Right Model Size

Select models based on your app's requirements and device constraints:

```swift
func selectOptimalModel() -> String {
    let device = UIDevice.current
    let availableMemory = ProcessInfo.processInfo.physicalMemory
    
    // High-end devices (iPhone 15 Pro, iPad Pro)
    if device.hasNeuralEngine && availableMemory > 6_000_000_000 {
        return "pe-core-s16-384"  // Best balance
    }
    
    // Mid-range devices (iPhone 12+, iPad Air)
    else if availableMemory > 3_000_000_000 {
        return "pe-core-t16-384"  // Fast and efficient
    }
    
    // Older devices
    else {
        return "pe-core-t16-224"  // Minimal footprint
    }
}
```

**Performance Targets:**
- Real-time apps: < 50ms per inference
- Interactive apps: < 200ms per inference
- Batch processing: Optimize for throughput

### 2. Optimize Compute Units

Configure models for optimal hardware utilization:

```swift
func configureOptimalComputeUnits() -> MLComputeUnits {
    let device = UIDevice.current
    
    // Use Neural Engine when available
    if device.hasNeuralEngine {
        return .cpuAndNeuralEngine
    }
    
    // Fall back to GPU for older devices
    if device.supportsMetalPerformanceShaders {
        return .cpuAndGPU
    }
    
    return .cpuOnly
}
```

### 3. Memory Management

Implement proper memory management for sustained performance:

```swift
class MemoryAwareClassifier {
    private var classifier: PECoreClassifier?
    private let memoryPressureObserver = MemoryPressureObserver()
    
    init() {
        memoryPressureObserver.onMemoryWarning = { [weak self] in
            self?.handleMemoryPressure()
        }
    }
    
    private func handleMemoryPressure() {
        // Release non-essential caches
        embeddingCache.removeAll()
        
        // Consider switching to smaller model
        if currentModel == "pe-core-s16-384" {
            loadModel("pe-core-t16-384")
        }
    }
}
```

## User Experience Guidelines

### 1. Progressive Loading

Implement smooth app startup with progressive model loading:

```swift
class AppStartupManager {
    func startApp() {
        // Phase 1: Show UI immediately with placeholder
        showMainInterface()
        
        // Phase 2: Load basic model from bundle
        loadBundleModel { success in
            if success {
                self.enableBasicFeatures()
                
                // Phase 3: Check for updates in background
                self.checkForModelUpdates()
            }
        }
    }
    
    private func checkForModelUpdates() {
        modelManager.checkForUpdates { hasUpdates in
            if hasUpdates {
                // Show optional update prompt
                self.showUpdatePrompt()
            }
        }
    }
}
```

### 2. Feedback and Error Handling

Provide clear feedback for all states:

```swift
enum ClassificationState {
    case idle
    case processing
    case completed(ClassificationResult)
    case error(ClassificationError)
    case lowConfidence(String, Float)
}

func updateUI(for state: ClassificationState) {
    switch state {
    case .idle:
        statusLabel.text = "Ready to classify"
        activityIndicator.stopAnimating()
        
    case .processing:
        statusLabel.text = "Analyzing image..."
        activityIndicator.startAnimating()
        
    case .completed(let result):
        statusLabel.text = "\(result.className.capitalized)"
        showConfidence(result.confidence)
        
    case .error(let error):
        showErrorMessage(for: error)
        
    case .lowConfidence(let className, let confidence):
        statusLabel.text = "Possibly \(className)?"
        showConfidenceWarning(confidence)
    }
}
```

### 3. Offline-First Design

Design for offline operation with graceful degradation:

```swift
class OfflineFirstClassifier {
    private let offlineCapabilities: OfflineCapabilities
    
    func classify(image: UIImage, completion: @escaping (ClassificationResult) -> Void) {
        // Always try local classification first
        classifyLocally(image: image) { localResult in
            if localResult.confidence > 0.8 {
                completion(localResult)
            } else {
                // Only use online enhancement if needed and available
                self.enhanceWithOnlineData(localResult, completion: completion)
            }
        }
    }
    
    private func enhanceWithOnlineData(_ localResult: ClassificationResult, 
                                     completion: @escaping (ClassificationResult) -> Void) {
        guard NetworkMonitor.isConnected else {
            completion(localResult)
            return
        }
        
        // Fetch additional context or refined classifications
        onlineClassificationService.enhance(localResult) { enhancedResult in
            completion(enhancedResult ?? localResult)
        }
    }
}
```

## Classification Strategy

### 1. Confidence Thresholds

Implement adaptive confidence thresholds:

```swift
struct AdaptiveThresholds {
    static func threshold(for className: String, userHistory: UserHistory) -> Float {
        // Higher threshold for critical classifications
        let criticalClasses = ["medical", "security", "financial"]
        if criticalClasses.contains(className.lowercased()) {
            return 0.9
        }
        
        // Adjust based on user's previous interactions
        let userAccuracy = userHistory.accuracyFor(className: className)
        if userAccuracy > 0.95 {
            return 0.7  // User rarely corrects this class
        } else if userAccuracy < 0.8 {
            return 0.85 // User often corrects this class
        }
        
        return 0.75 // Default threshold
    }
}
```

### 2. Hierarchical Classification

Implement coarse-to-fine classification for better UX:

```swift
class HierarchicalClassificationUI {
    func performClassification(image: UIImage) {
        // Step 1: Quick coarse classification
        coarseClassifier.classify(image: image) { coarseResult in
            self.showCoarseResult(coarseResult)
            
            // Step 2: Detailed classification within category
            self.fineClassifier.classify(image: image, 
                                       category: coarseResult.className) { fineResult in
                self.showFinalResult(fineResult)
            }
        }
    }
    
    private func showCoarseResult(_ result: ClassificationResult) {
        // Show general category immediately
        categoryLabel.text = result.className
        categoryLabel.alpha = 0.7 // Indicate provisional
    }
    
    private func showFinalResult(_ result: ClassificationResult) {
        // Show specific classification
        UIView.animate(withDuration: 0.3) {
            self.categoryLabel.alpha = 1.0
            self.detailLabel.text = result.className
        }
    }
}
```

### 3. User Feedback Integration

Learn from user corrections:

```swift
class LearningClassifier {
    private let feedbackCollector = UserFeedbackCollector()
    
    func classify(image: UIImage, completion: @escaping (ClassificationResult) -> Void) {
        baseClassifier.classify(image: image) { result in
            // Present result with feedback option
            self.presentResultWithFeedback(result) { userCorrection in
                if let correction = userCorrection {
                    self.handleUserCorrection(image: image, 
                                            prediction: result.className, 
                                            correction: correction)
                }
                completion(result)
            }
        }
    }
    
    private func handleUserCorrection(image: UIImage, 
                                    prediction: String, 
                                    correction: String) {
        // Add to local training data
        let embedding = embeddingExtractor.extractEmbedding(from: image)
        localLearningData.add(embedding: embedding, label: correction)
        
        // Optionally contribute to global model improvement
        if userConsent.allowsDataSharing {
            feedbackCollector.submitCorrection(
                prediction: prediction,
                correction: correction,
                embedding: embedding
            )
        }
    }
}
```

## Performance Optimization

### 1. Caching Strategies

Implement intelligent caching:

```swift
class IntelligentCache {
    private let embeddingCache = LRUCache<String, [Float]>(capacity: 1000)
    private let resultCache = LRUCache<String, ClassificationResult>(capacity: 500)
    
    func classify(image: UIImage, completion: @escaping (ClassificationResult) -> Void) {
        let imageHash = image.hash
        
        // Check result cache first
        if let cachedResult = resultCache.get(imageHash) {
            completion(cachedResult)
            return
        }
        
        // Check embedding cache
        if let cachedEmbedding = embeddingCache.get(imageHash) {
            let result = classifyEmbedding(cachedEmbedding)
            resultCache.set(imageHash, result)
            completion(result)
            return
        }
        
        // Compute new embedding
        embeddingExtractor.extract(from: image) { embedding in
            self.embeddingCache.set(imageHash, embedding)
            let result = self.classifyEmbedding(embedding)
            self.resultCache.set(imageHash, result)
            completion(result)
        }
    }
}
```

### 2. Batch Processing

Optimize for batch operations:

```swift
class BatchProcessor {
    func processBatch(images: [UIImage], 
                     batchSize: Int = 8,
                     completion: @escaping ([ClassificationResult]) -> Void) {
        var allResults: [ClassificationResult] = []
        let chunks = images.chunked(into: batchSize)
        
        let group = DispatchGroup()
        
        for chunk in chunks {
            group.enter()
            
            // Process chunk in parallel
            processChunk(chunk) { chunkResults in
                allResults.append(contentsOf: chunkResults)
                group.leave()
            }
        }
        
        group.notify(queue: .main) {
            // Sort results to maintain original order
            let sortedResults = self.sortResultsByOriginalOrder(allResults, images: images)
            completion(sortedResults)
        }
    }
}
```

### 3. Background Processing

Implement background model updates:

```swift
class BackgroundModelManager {
    func scheduleBackgroundUpdate() {
        let identifier = "com.app.model-update"
        
        let request = BGAppRefreshTaskRequest(identifier: identifier)
        request.earliestBeginDate = Date(timeIntervalSinceNow: 24 * 60 * 60) // 24 hours
        
        try? BGTaskScheduler.shared.submit(request)
    }
    
    func handleBackgroundUpdate(task: BGAppRefreshTask) {
        let operation = ModelUpdateOperation()
        
        task.expirationHandler = {
            operation.cancel()
        }
        
        operation.completionBlock = {
            task.setTaskCompleted(success: !operation.isCancelled)
        }
        
        backgroundQueue.addOperation(operation)
    }
}
```

## Security and Privacy

### 1. Local Processing Priority

Always prioritize on-device processing:

```swift
class PrivacyAwareClassifier {
    private let localClassifier: PECoreClassifier
    private let privacySettings: PrivacySettings
    
    func classify(image: UIImage, completion: @escaping (ClassificationResult) -> Void) {
        // Always process locally first
        localClassifier.classify(image: image) { localResult in
            
            // Only consider cloud enhancement if explicitly allowed
            if self.privacySettings.allowsCloudEnhancement && 
               localResult.confidence < 0.7 {
                self.enhanceWithCloud(localResult, completion: completion)
            } else {
                completion(localResult)
            }
        }
    }
    
    private func enhanceWithCloud(_ localResult: ClassificationResult,
                                completion: @escaping (ClassificationResult) -> Void) {
        // Send only embeddings, never raw images
        let embedding = embeddingExtractor.extract(from: image)
        cloudService.classify(embedding: embedding) { cloudResult in
            // Combine results intelligently
            let finalResult = self.combineResults(local: localResult, cloud: cloudResult)
            completion(finalResult)
        }
    }
}
```

### 2. Secure Model Updates

Implement secure model distribution:

```swift
class SecureModelDownloader {
    private let certificatePinner = CertificatePinner()
    
    func downloadModel(info: ModelInfo, completion: @escaping (Result<URL, Error>) -> Void) {
        // Verify SSL certificate
        let session = URLSession(configuration: .default, 
                               delegate: certificatePinner, 
                               delegateQueue: nil)
        
        session.downloadTask(with: info.url) { tempURL, response, error in
            guard let tempURL = tempURL else {
                completion(.failure(error ?? ModelError.downloadFailed))
                return
            }
            
            // Verify model signature
            guard self.verifyModelSignature(at: tempURL, info: info) else {
                completion(.failure(ModelError.invalidSignature))
                return
            }
            
            completion(.success(tempURL))
        }.resume()
    }
    
    private func verifyModelSignature(at url: URL, info: ModelInfo) -> Bool {
        // Implement cryptographic signature verification
        return CryptographicSignatureVerifier.verify(file: url, 
                                                    signature: info.signature,
                                                    publicKey: trustedPublicKey)
    }
}
```

## Monitoring and Analytics

### 1. Performance Metrics

Track key performance indicators:

```swift
class PerformanceMonitor {
    private let analytics = Analytics()
    
    func trackClassification(duration: TimeInterval, 
                           modelName: String, 
                           confidence: Float,
                           deviceInfo: DeviceInfo) {
        analytics.track("classification_performance", parameters: [
            "duration_ms": duration * 1000,
            "model_name": modelName,
            "confidence": confidence,
            "device_model": deviceInfo.model,
            "ios_version": deviceInfo.osVersion,
            "available_memory": deviceInfo.availableMemory
        ])
    }
    
    func trackModelUpdate(modelName: String, 
                         downloadTime: TimeInterval,
                         success: Bool) {
        analytics.track("model_update", parameters: [
            "model_name": modelName,
            "download_time_s": downloadTime,
            "success": success,
            "network_type": NetworkInfo.currentType
        ])
    }
}
```

### 2. Error Tracking

Implement comprehensive error tracking:

```swift
class ErrorTracker {
    func trackClassificationError(_ error: ClassificationError, 
                                context: [String: Any]) {
        let errorInfo: [String: Any] = [
            "error_type": String(describing: error),
            "context": context,
            "timestamp": Date().timeIntervalSince1970,
            "app_version": AppInfo.version,
            "model_version": ModelInfo.currentVersion
        ]
        
        // Log locally for debugging
        Logger.error("Classification failed", metadata: errorInfo)
        
        // Send to crash reporting service
        CrashReporter.recordError(error, metadata: errorInfo)
    }
}
```

## Testing Strategies

### 1. Unit Testing

Test classification components:

```swift
class ClassificationTests: XCTestCase {
    func testBasicClassification() {
        let classifier = PECoreClassifier()
        let testImage = TestImages.dog
        
        let expectation = XCTestExpectation(description: "Classification completes")
        
        classifier.classify(image: testImage) { result in
            XCTAssertEqual(result.className, "dog")
            XCTAssertGreaterThan(result.confidence, 0.8)
            expectation.fulfill()
        }
        
        wait(for: [expectation], timeout: 5.0)
    }
    
    func testModelFallback() {
        // Test fallback to bundle model when CDN fails
        let mockModelManager = MockModelManager()
        mockModelManager.simulateNetworkFailure = true
        
        let classifier = PECoreClassifier(modelManager: mockModelManager)
        
        // Should still work with bundle model
        XCTAssertNotNil(classifier.classify(image: TestImages.cat))
    }
}
```

### 2. Integration Testing

Test end-to-end workflows:

```swift
class IntegrationTests: XCTestCase {
    func testModelUpdateWorkflow() {
        let modelManager = PECoreModelManager.shared
        
        // Test complete update cycle
        let updateExpectation = XCTestExpectation(description: "Model update completes")
        
        modelManager.downloadUpdates(
            progress: { _, _ in },
            completion: { result in
                switch result {
                case .success:
                    updateExpectation.fulfill()
                case .failure(let error):
                    XCTFail("Update failed: \(error)")
                }
            }
        )
        
        wait(for: [updateExpectation], timeout: 30.0)
    }
}
```

## Deployment Checklist

### Pre-Release
- [ ] Test on multiple device types and iOS versions
- [ ] Verify offline functionality
- [ ] Test model fallback mechanisms
- [ ] Validate performance under memory pressure
- [ ] Test background updates
- [ ] Verify security measures

### Launch
- [ ] Monitor classification accuracy
- [ ] Track performance metrics
- [ ] Monitor error rates
- [ ] Watch model download success rates
- [ ] Track user satisfaction

### Post-Launch
- [ ] Analyze classification patterns
- [ ] Identify improvement opportunities
- [ ] Plan model updates based on usage data
- [ ] Optimize for new device capabilities

This comprehensive guide should help you build robust, performant, and user-friendly iOS applications with PE Core models.