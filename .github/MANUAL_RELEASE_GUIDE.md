# Manual Release Guide for PE-Core-S16-384 Models

This guide explains how to manually trigger the GitHub Actions workflow to create PE-Core-S16-384 model releases.

## 🚀 How to Trigger a Manual Release

### **Method 1: GitHub Web Interface (Recommended)**

1. **Navigate to Actions Tab**
   - Go to your repository on GitHub
   - Click the **"Actions"** tab
   - Find **"Release PE-Core-S16-384 Models"** workflow

2. **Trigger Workflow**
   - Click **"Run workflow"** button (on the right side)
   - Select the branch (usually `main`)
   - Configure the parameters (see options below)
   - Click **"Run workflow"**

### **Method 2: GitHub CLI**

```bash
# Install GitHub CLI if not already installed
# https://cli.github.com/

# Trigger a release with all formats
gh workflow run "Release PE-Core-S16-384 Models" \
  -f model_version="v1.0.0" \
  -f include_formats="coreml,onnx,executorch" \
  -f create_release="true" \
  -f run_benchmarks="true" \
  -f release_type="release"

# Quick CoreML-only release for testing
gh workflow run "Release PE-Core-S16-384 Models" \
  -f model_version="v1.0.0-test" \
  -f include_formats="coreml" \
  -f create_release="false" \
  -f run_benchmarks="false" \
  -f release_type="draft"
```

### **Method 3: REST API**

```bash
curl -X POST \
  -H "Accept: application/vnd.github.v3+json" \
  -H "Authorization: token YOUR_TOKEN" \
  https://api.github.com/repos/OWNER/REPO/actions/workflows/release-s16-models.yml/dispatches \
  -d '{
    "ref": "main",
    "inputs": {
      "model_version": "v1.0.0",
      "include_formats": "coreml,onnx,executorch",
      "create_release": "true",
      "run_benchmarks": "true",
      "release_type": "release"
    }
  }'
```

## ⚙️ Configuration Options

### **Model Version** (Required)
- **Format**: `v1.0.0`, `v1.1.0-beta`, `v2.0.0-rc1`
- **Examples**:
  - `v1.0.0` - Stable release
  - `v1.1.0-beta` - Beta version
  - `v2.0.0-rc1` - Release candidate
  - `v1.0.1-hotfix` - Hotfix release

### **Include Formats** (Required)
Choose which model formats to convert and include:

| Option | Description | Use Case |
|--------|-------------|----------|
| `coreml` | CoreML only | iOS-only deployment |
| `onnx` | ONNX only | Cross-platform testing |
| `executorch` | ExecuTorch only | PyTorch mobile only |
| `coreml,onnx` | CoreML + ONNX | iOS + cross-platform |
| `coreml,executorch` | CoreML + ExecuTorch | iOS native + PyTorch mobile |
| `onnx,executorch` | ONNX + ExecuTorch | No iOS native |
| `coreml,onnx,executorch` | All formats | Complete release (default) |

### **Create Release** (Optional, default: true)
- **`true`**: Create GitHub release with packages
- **`false`**: Only build and upload artifacts (for testing)

### **Run Benchmarks** (Optional, default: true)
- **`true`**: Run performance benchmarks (takes longer)
- **`false`**: Skip benchmarks for faster builds

### **Release Type** (Optional, default: release)
- **`release`**: Public stable release
- **`prerelease`**: Pre-release (beta, rc, etc.)
- **`draft`**: Draft release (not publicly visible)

## 📋 Common Use Cases

### **1. Full Production Release**
```yaml
Model Version: v1.0.0
Include Formats: coreml,onnx,executorch
Create Release: true
Run Benchmarks: true
Release Type: release
```

### **2. Beta Testing Release**
```yaml
Model Version: v1.1.0-beta
Include Formats: coreml,onnx
Create Release: true
Run Benchmarks: true
Release Type: prerelease
```

### **3. Quick CoreML Test**
```yaml
Model Version: v1.0.0-test
Include Formats: coreml
Create Release: false
Run Benchmarks: false
Release Type: draft
```

### **4. Development Build**
```yaml
Model Version: v1.0.0-dev
Include Formats: onnx
Create Release: false
Run Benchmarks: false
Release Type: draft
```

### **5. Platform-Specific Release**
```yaml
# iOS-focused release
Model Version: v1.0.0-ios
Include Formats: coreml,executorch
Create Release: true
Run Benchmarks: true
Release Type: release
```

## 📦 What Gets Created

### **Artifacts (Always Created)**
- `pe-core-s16-384-models-{version}.zip` - Build artifacts
- Individual format packages (based on selection)
- `release-manifest.json` - Complete metadata

### **GitHub Release (If enabled)**
- Release with comprehensive description
- All model packages attached
- SHA256 checksums included
- Usage examples and documentation links

### **Benchmarks (If enabled)**
- Performance analysis across all formats
- Real-time capability assessment (30/60 FPS)
- Platform-specific latency estimates

## ⏱️ Expected Timing

| Configuration | Duration | Notes |
|---------------|----------|--------|
| CoreML only, no benchmarks | ~10-15 min | Fastest option |
| ONNX only, no benchmarks | ~8-12 min | Quick cross-platform |
| All formats, no benchmarks | ~15-20 min | Complete without testing |
| All formats, with benchmarks | ~25-35 min | Full validation |

## 🔍 Monitoring Progress

### **Via GitHub Interface**
1. Go to **Actions** tab
2. Click on your running workflow
3. Monitor real-time logs for each step

### **Via GitHub CLI**
```bash
# List recent workflow runs
gh run list --workflow="release-s16-models.yml"

# Watch specific run
gh run watch RUN_ID

# View logs
gh run view RUN_ID --log
```

### **Key Progress Indicators**
- ✅ **Dependencies installed** - Setup complete
- ✅ **Models converted** - Core conversion done
- ✅ **Checksums generated** - Security validation
- ✅ **Packages created** - Distribution ready
- ✅ **Release published** - Public availability
- ✅ **Benchmarks completed** - Performance validated

## 🐛 Troubleshooting

### **Common Issues**

#### **"Model loading failed"**
- Check if PE-Core-S16-384 is available
- Verify network connectivity to HuggingFace
- Try with a different format

#### **"CoreML conversion failed"**
- Normal on some PyTorch versions
- ONNX/ExecuTorch will still work
- Check conversion logs for details

#### **"Release already exists"**
- Use a different version number
- Delete existing release if needed
- Use draft mode for testing

#### **"Benchmarks timed out"**
- Disable benchmarks for faster builds
- Use smaller format subset
- Try again (sometimes transient)

### **Getting Help**
- Check workflow logs in GitHub Actions
- Look for error messages in specific steps
- File an issue with logs if needed

## 🎯 Best Practices

### **Version Numbering**
- Use semantic versioning: `vMAJOR.MINOR.PATCH`
- Add suffixes for pre-releases: `-beta`, `-rc1`, `-alpha`
- Use descriptive suffixes: `-ios`, `-android`, `-test`

### **Format Selection**
- **Development**: Start with single format
- **Testing**: Use `coreml,onnx` for broad coverage
- **Production**: Use all formats for complete release

### **Release Strategy**
- **Draft** for initial testing
- **Prerelease** for beta testing
- **Release** for production deployment

### **Performance Validation**
- Always run benchmarks for production releases
- Skip benchmarks for quick development builds
- Monitor benchmark trends across versions

This manual trigger system gives you complete control over when and how PE-Core-S16-384 models are released, with flexibility for different use cases from development to production.