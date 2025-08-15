# Model Management and CDN Deployment

This document covers how to manage PE Core models in iOS apps with CDN-based updates and local caching strategies.

## Overview

The model management system provides:

- **CDN-first loading**: Download latest models from CDN
- **Bundle fallback**: Ship baseline models with the app
- **Automatic updates**: Check for new model versions
- **Storage optimization**: Manage local model cache
- **Version control**: Track model versions and compatibility

## Model Distribution Strategy

### 1. Three-Tier Architecture

```
CDN (Primary) → Local Cache → App Bundle (Fallback)
```

1. **CDN**: Latest models with version metadata
2. **Local Cache**: Downloaded models for offline use
3. **App Bundle**: Baseline models shipped with app

### 2. Model Versioning

```json
{
  "models": {
    "pe-core-s16-384": {
      "version": "1.2.0",
      "url": "https://cdn.example.com/models/pe-core-s16-384-v1.2.0.mlpackage",
      "checksum": "sha256:abc123...",
      "size": 48000000,
      "minAppVersion": "2.0.0",
      "features": ["classification", "embedding"],
      "hardware": {
        "neuralEngine": true,
        "gpu": true,
        "cpu": true
      }
    }
  },
  "prototypes": {
    "version": "2.1.0",
    "url": "https://cdn.example.com/prototypes/v2.1.0.json",
    "checksum": "sha256:def456...",
    "classes": 150
  }
}
```

## Implementation

### Core Model Manager

```swift
import Foundation
import CoreML

class PECoreModelManager {
    static let shared = PECoreModelManager()
    
    private let cdnBaseURL = "https://cdn.example.com"
    private let cacheDirectory: URL
    private let bundleDirectory: URL
    
    struct ModelInfo {
        let name: String
        let version: String
        let url: URL
        let checksum: String
        let size: Int64
        let localPath: URL?
    }
    
    private init() {
        // Setup cache directory
        let documents = FileManager.default.urls(for: .documentDirectory, 
                                                in: .userDomainMask).first!
        cacheDirectory = documents.appendingPathComponent("models")
        
        // Setup bundle directory
        bundleDirectory = Bundle.main.bundleURL.appendingPathComponent("models")
        
        createCacheDirectoryIfNeeded()
    }
    
    // MARK: - Public Interface
    
    func loadModel(name: String, completion: @escaping (Result<MLModel, Error>) -> Void) {
        // 1. Try to load from local cache
        if let cachedModel = loadCachedModel(name: name) {
            completion(.success(cachedModel))
            return
        }
        
        // 2. Try to load from app bundle
        if let bundledModel = loadBundledModel(name: name) {
            completion(.success(bundledModel))
            
            // Asynchronously check for updates
            checkForModelUpdate(name: name)
            return
        }
        
        // 3. Download from CDN
        downloadModel(name: name) { result in
            switch result {
            case .success(let model):
                completion(.success(model))
            case .failure(let error):
                completion(.failure(error))
            }
        }
    }
    
    func checkForUpdates(completion: @escaping (Bool) -> Void) {
        fetchModelManifest { manifest in
            guard let manifest = manifest else {
                completion(false)
                return
            }
            
            let hasUpdates = self.compareVersions(manifest: manifest)
            completion(hasUpdates)
        }
    }
    
    func downloadUpdates(progress: @escaping (String, Float) -> Void, 
                        completion: @escaping (Result<Void, Error>) -> Void) {
        fetchModelManifest { manifest in
            guard let manifest = manifest else {
                completion(.failure(ModelError.manifestUnavailable))
                return
            }
            
            let updatesNeeded = self.getUpdatesNeeded(manifest: manifest)
            self.downloadModels(updatesNeeded, progress: progress, completion: completion)
        }
    }
}

// MARK: - Private Implementation

private extension PECoreModelManager {
    
    func loadCachedModel(name: String) -> MLModel? {
        let modelPath = cacheDirectory.appendingPathComponent("\(name).mlpackage")
        
        guard FileManager.default.fileExists(atPath: modelPath.path),
              isModelVersionCurrent(at: modelPath, name: name) else {
            return nil
        }
        
        return try? MLModel(contentsOf: modelPath)
    }
    
    func loadBundledModel(name: String) -> MLModel? {
        let modelPath = bundleDirectory.appendingPathComponent("\(name).mlpackage")
        
        guard FileManager.default.fileExists(atPath: modelPath.path) else {
            return nil
        }
        
        return try? MLModel(contentsOf: modelPath)
    }
    
    func downloadModel(name: String, completion: @escaping (Result<MLModel, Error>) -> Void) {
        fetchModelManifest { manifest in
            guard let manifest = manifest,
                  let modelInfo = manifest.models[name] else {
                completion(.failure(ModelError.modelNotFound))
                return
            }
            
            self.downloadModelFile(modelInfo: modelInfo) { result in
                switch result {
                case .success(let localURL):
                    do {
                        let model = try MLModel(contentsOf: localURL)
                        completion(.success(model))
                    } catch {
                        completion(.failure(error))
                    }
                case .failure(let error):
                    completion(.failure(error))
                }
            }
        }
    }
    
    func downloadModelFile(modelInfo: ModelInfo, 
                          completion: @escaping (Result<URL, Error>) -> Void) {
        let localURL = cacheDirectory.appendingPathComponent("\(modelInfo.name).mlpackage")
        
        // Create download task
        let task = URLSession.shared.downloadTask(with: modelInfo.url) { tempURL, response, error in
            if let error = error {
                completion(.failure(error))
                return
            }
            
            guard let tempURL = tempURL else {
                completion(.failure(ModelError.downloadFailed))
                return
            }
            
            do {
                // Verify checksum
                guard self.verifyChecksum(at: tempURL, expected: modelInfo.checksum) else {
                    completion(.failure(ModelError.checksumMismatch))
                    return
                }
                
                // Move to cache directory
                if FileManager.default.fileExists(atPath: localURL.path) {
                    try FileManager.default.removeItem(at: localURL)
                }
                try FileManager.default.moveItem(at: tempURL, to: localURL)
                
                // Store version metadata
                self.storeModelMetadata(modelInfo, at: localURL)
                
                completion(.success(localURL))
            } catch {
                completion(.failure(error))
            }
        }
        
        task.resume()
    }
    
    func fetchModelManifest(completion: @escaping (ModelManifest?) -> Void) {
        let manifestURL = URL(string: "\(cdnBaseURL)/manifest.json")!
        
        URLSession.shared.dataTask(with: manifestURL) { data, response, error in
            guard let data = data,
                  let manifest = try? JSONDecoder().decode(ModelManifest.self, from: data) else {
                completion(nil)
                return
            }
            
            completion(manifest)
        }.resume()
    }
    
    func verifyChecksum(at url: URL, expected: String) -> Bool {
        guard let data = try? Data(contentsOf: url) else { return false }
        
        let computed = data.sha256
        return computed == expected
    }
    
    func storeModelMetadata(_ modelInfo: ModelInfo, at url: URL) {
        let metadataURL = url.appendingPathExtension("metadata")
        let metadata = [
            "version": modelInfo.version,
            "checksum": modelInfo.checksum,
            "downloadDate": ISO8601DateFormatter().string(from: Date())
        ]
        
        if let data = try? JSONSerialization.data(withJSONObject: metadata) {
            try? data.write(to: metadataURL)
        }
    }
    
    func isModelVersionCurrent(at url: URL, name: String) -> Bool {
        let metadataURL = url.appendingPathExtension("metadata")
        
        guard let data = try? Data(contentsOf: metadataURL),
              let metadata = try? JSONSerialization.jsonObject(with: data) as? [String: String],
              let version = metadata["version"] else {
            return false
        }
        
        // Check against current version (could be fetched from manifest)
        return isVersionCurrent(version, for: name)
    }
    
    func createCacheDirectoryIfNeeded() {
        try? FileManager.default.createDirectory(at: cacheDirectory, 
                                                withIntermediateDirectories: true)
    }
}
```

### Model Manifest Structure

```swift
struct ModelManifest: Codable {
    struct Model: Codable {
        let version: String
        let url: String
        let checksum: String
        let size: Int64
        let minAppVersion: String
        let features: [String]
        let hardware: HardwareSupport
    }
    
    struct HardwareSupport: Codable {
        let neuralEngine: Bool
        let gpu: Bool
        let cpu: Bool
    }
    
    struct Prototypes: Codable {
        let version: String
        let url: String
        let checksum: String
        let classes: Int
    }
    
    let models: [String: Model]
    let prototypes: Prototypes
    let minAppVersion: String
}
```

### Cache Management

```swift
extension PECoreModelManager {
    
    func cleanupOldModels() {
        let modelFiles = try? FileManager.default.contentsOfDirectory(at: cacheDirectory, 
                                                                     includingPropertiesForKeys: [.contentModificationDateKey])
        
        // Sort by modification date
        let sortedFiles = modelFiles?.sorted { url1, url2 in
            let date1 = try? url1.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate
            let date2 = try? url2.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate
            return (date1 ?? Date.distantPast) < (date2 ?? Date.distantPast)
        }
        
        // Keep only recent models, remove old ones
        let maxModels = 3
        if let files = sortedFiles, files.count > maxModels {
            for i in 0..<(files.count - maxModels) {
                try? FileManager.default.removeItem(at: files[i])
            }
        }
    }
    
    func getCacheSize() -> Int64 {
        guard let files = try? FileManager.default.contentsOfDirectory(at: cacheDirectory, 
                                                                      includingPropertiesForKeys: [.fileSizeKey]) else {
            return 0
        }
        
        return files.reduce(0) { total, url in
            let size = try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize
            return total + Int64(size ?? 0)
        }
    }
    
    func clearCache() {
        try? FileManager.default.removeItem(at: cacheDirectory)
        createCacheDirectoryIfNeeded()
    }
}
```

## App Bundle Strategy

### Shipping Baseline Models

Include essential models in your app bundle:

```
MyApp.app/
├── models/
│   ├── pe-core-s16-384.mlpackage    # Primary model
│   └── pe-core-t16-384.mlpackage    # Fallback for older devices
├── prototypes/
│   └── baseline-classes.json        # Basic class set
└── Info.plist
```

### Bundle Selection Logic

```swift
class BundleModelSelector {
    static func selectOptimalModel() -> String {
        let device = UIDevice.current
        
        // Check device capabilities
        if device.hasNeuralEngine && device.totalMemory > 4_000_000_000 {
            return "pe-core-s16-384"  // Best model for capable devices
        } else if device.totalMemory > 2_000_000_000 {
            return "pe-core-t16-384"  // Smaller model for limited devices
        } else {
            return "pe-core-t16-224"  // Minimal model for old devices
        }
    }
}
```

## CDN Configuration

### CloudFront Distribution

```yaml
# Example CDN configuration
Distribution:
  Origins:
    - DomainName: models.yourapp.com
      Id: models-origin
      CustomOriginConfig:
        HTTPPort: 443
        OriginProtocolPolicy: https-only
  
  DefaultCacheBehavior:
    TargetOriginId: models-origin
    ViewerProtocolPolicy: redirect-to-https
    CachePolicyId: optimized-caching
    TTL:
      DefaultTTL: 86400  # 24 hours
      MaxTTL: 31536000   # 1 year
  
  Comment: "PE Core models distribution"
```

### Model Organization

```
CDN Structure:
models.yourapp.com/
├── manifest.json                    # Model catalog
├── models/
│   ├── pe-core-s16-384-v1.2.0.mlpackage
│   ├── pe-core-t16-384-v1.2.0.mlpackage
│   └── checksums/
├── prototypes/
│   ├── v2.1.0.json                 # Classification prototypes
│   └── checksums/
└── metadata/
    └── compatibility.json          # App version compatibility
```

## Error Handling and Fallbacks

```swift
enum ModelError: Error {
    case modelNotFound
    case downloadFailed
    case checksumMismatch
    case incompatibleVersion
    case storageExceeded
    case manifestUnavailable
}

extension PECoreModelManager {
    
    func handleModelLoadError(_ error: Error, for modelName: String) -> MLModel? {
        switch error {
        case ModelError.downloadFailed:
            // Try bundle fallback
            return loadBundledModel(name: modelName)
            
        case ModelError.checksumMismatch:
            // Re-download model
            downloadModel(name: modelName) { _ in }
            return loadBundledModel(name: modelName)
            
        case ModelError.incompatibleVersion:
            // Use older compatible version
            return findCompatibleModel(for: modelName)
            
        default:
            // Log error and use bundle fallback
            logError(error, context: "Model loading for \(modelName)")
            return loadBundledModel(name: modelName)
        }
    }
}
```

## Best Practices

### 1. Progressive Updates

- Ship with baseline models in bundle
- Download updates in background
- Gracefully fallback on errors

### 2. Storage Management

- Set reasonable cache limits
- Clean up old model versions
- Monitor disk usage

### 3. Network Efficiency

- Use delta updates when possible
- Compress models appropriately
- Implement retry logic with backoff

### 4. User Experience

- Show download progress for large models
- Allow offline operation with cached models
- Provide settings for data usage control

### 5. Security

- Verify model checksums
- Use HTTPS for all downloads
- Implement certificate pinning

### 6. Monitoring

- Track model download success rates
- Monitor cache hit ratios
- Log model performance metrics

## Next Steps

- See [Classification](classification.md) for using loaded models
- Check [Examples](examples/) for integration samples
- Review [Best Practices](best-practices.md) for production tips