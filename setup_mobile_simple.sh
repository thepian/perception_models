#!/bin/bash
# Simple Mobile Environment Setup with UV
# Uses available PyTorch versions + compatible ExecuTorch

set -e

echo "🚀 Setting up Simple Mobile Environment with UV"
echo ""

# Create mobile directory
mkdir -p mobile_deployment
cd mobile_deployment

# Initialize UV project
echo "📦 Initializing UV project..."
uv init --python 3.11

# Install PyTorch first (latest stable)
echo "🔧 Installing PyTorch (latest stable)..."
uv add torch torchvision torchaudio

# Install core dependencies
echo "📦 Installing core dependencies..."
uv add numpy scipy pillow einops requests tqdm

# Install ONNX tools
echo "🔧 Installing ONNX tools..."
uv add onnx onnxruntime

# Try to install ExecuTorch (fallback to older version if needed)
echo "🔧 Installing ExecuTorch..."
uv add executorch || {
    echo "⚠️  ExecuTorch latest failed, trying older version..."
    uv add "executorch>=0.4.0,<0.7.0" || {
        echo "⚠️  ExecuTorch not available, will use ONNX-only approach"
    }
}

# Install development tools
echo "🛠️  Installing development tools..."
uv add --dev pytest black isort jupyter matplotlib

# Copy PE Core source
echo "📋 Copying PE Core source..."
cp -r ../perception_models .

# Create simple test script
cat > test_environment.py << 'EOF'
#!/usr/bin/env python3
"""Test mobile environment setup."""

def test_imports():
    """Test that all required packages can be imported."""
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch: {e}")
        return False
    
    try:
        import torchvision
        print(f"✅ TorchVision: {torchvision.__version__}")
    except ImportError as e:
        print(f"❌ TorchVision: {e}")
    
    try:
        import onnx
        print(f"✅ ONNX: {onnx.__version__}")
    except ImportError as e:
        print(f"❌ ONNX: {e}")
    
    try:
        import executorch
        print("✅ ExecuTorch: Available")
        return True
    except ImportError:
        print("⚠️  ExecuTorch: Not available (will use ONNX-only)")
        return False

def test_pe_core():
    """Test PE Core model loading."""
    try:
        from perception_models.models.vision_transformer import VisionTransformer
        print("✅ PE Core models: Available")
        return True
    except ImportError as e:
        print(f"❌ PE Core models: {e}")
        return False

def main():
    print("🔍 Testing Mobile Environment Setup")
    print("=" * 50)
    
    imports_ok = test_imports()
    pe_core_ok = test_pe_core()
    
    print("=" * 50)
    if imports_ok and pe_core_ok:
        print("✅ Mobile environment setup successful!")
        print("\n📋 Next steps:")
        print("   1. Export PE Core to ONNX in main environment")
        print("   2. Convert ONNX to mobile format here")
        print("   3. Test mobile inference performance")
    else:
        print("❌ Mobile environment setup incomplete")
        print("   Some components missing but basic functionality available")
    
    return imports_ok and pe_core_ok

if __name__ == "__main__":
    main()
EOF

# Create activation script
cat > activate.sh << 'EOF'
#!/bin/bash
echo "🚀 Mobile Environment Activated"
echo "   Location: $(pwd)"
echo "   Python: $(uv run python --version)"
echo ""
echo "📋 Available commands:"
echo "   uv run python test_environment.py  # Test setup"
echo "   uv run python -c 'import torch; print(torch.__version__)'  # Check PyTorch"
echo ""
echo "💡 Use 'uv run python <script>' to run commands"
EOF

chmod +x activate.sh test_environment.py

echo ""
echo "✅ Simple mobile environment setup complete!"
echo ""
echo "📋 Test the setup:"
echo "   cd mobile_deployment"
echo "   uv run python test_environment.py"
echo ""
echo "🔧 Environment location: $(pwd)"
