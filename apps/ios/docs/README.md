# PE Core iOS Integration Documentation

This documentation covers the implementation of Perception Encoder (PE) Core models in iOS applications, with a focus on dynamic classification and efficient model deployment.

## Overview

PE Core models are vision transformers that output feature embeddings rather than fixed classification scores. This architecture enables:

- Dynamic classification without app updates
- On-device privacy-preserving inference
- Flexible class definitions via embeddings
- Efficient model updates via CDN

## Documentation Structure

1. **[Classification Implementation](classification.md)** - How to implement dynamic classification with PE Core
2. **[Model Management](model-management.md)** - CDN-based model deployment and updates
3. **[Code Examples](examples/)** - Swift implementation examples
4. **[Best Practices](best-practices.md)** - Performance optimization and deployment strategies

## Quick Start

```swift
// Initialize classifier with CDN model updates
let classifier = PECoreClassifier()

// Classify an image
classifier.classify(image: uiImage) { className, confidence in
    print("Predicted: \(className) with confidence: \(confidence)")
}

// Update classification prototypes without app update
classifier.updatePrototypes(from: "https://cdn.example.com/prototypes.json")
```

## Model Selection Guide

| Model | Size | Accuracy | Latency | Use Case |
|-------|------|----------|---------|----------|
| PE-Core-T16-384 | 12MB | 62.1% | 11-12ms | Real-time camera |
| PE-Core-S16-384 | 48MB | 72.7% | 11-12ms | **Recommended** |
| PE-Core-B16-224 | 180MB | 78.4% | 20-25ms | High accuracy |

## Key Features

- **Embedding-based classification**: Compare image embeddings with class prototypes
- **Dynamic updates**: Add/modify classes without App Store updates
- **Offline support**: Cache models and prototypes locally
- **Privacy-first**: All inference happens on-device
- **User customization**: Allow users to create custom classes

## Architecture Benefits

1. **No retraining required** - Add new classes by providing example embeddings
2. **Composable classifiers** - Combine multiple classifiers for hierarchical classification
3. **Fine-grained distinctions** - Distinguish between very similar classes
4. **Multi-modal capability** - Use text descriptions to define classes