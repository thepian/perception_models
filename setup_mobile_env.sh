#!/bin/bash
# Setup Mobile Deployment Environment with UV
# PyTorch 2.8 + ExecuTorch 0.7.0 for PE Core Mobile Deployment

set -e  # Exit on any error

echo "🚀 Setting up PE Core Mobile Deployment Environment"
echo "   PyTorch 2.8 + ExecuTorch 0.7.0 using UV"
echo ""

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "❌ UV not found. Please install UV first."
    echo "   Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Create mobile deployment directory
MOBILE_DIR="mobile_deployment"
if [ -d "$MOBILE_DIR" ]; then
    echo "⚠️  Mobile deployment directory already exists."
    read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing existing directory..."
        rm -rf "$MOBILE_DIR"
    else
        echo "✅ Using existing directory."
        cd "$MOBILE_DIR"
        echo "📁 Current directory: $(pwd)"
        exit 0
    fi
fi

echo "📁 Creating mobile deployment directory..."
mkdir -p "$MOBILE_DIR"
cd "$MOBILE_DIR"

echo "📋 Copying mobile configuration..."
cp ../mobile_pyproject.toml pyproject.toml

echo "📦 Initializing UV project with Python 3.11..."
uv init --python 3.11

echo "🔧 Installing dependencies with UV..."
uv sync --extra executorch --extra ios

# Copy PE Core source code
echo "📦 Copying PE Core source code..."
cp -r ../perception_models .

# Verify PyTorch installation
echo "🔍 Verifying PyTorch 2.8 installation..."
uv run python -c "import torch; print(f'PyTorch version: {torch.__version__}')" || {
    echo "⚠️  PyTorch 2.8 not available, falling back to latest stable..."
    uv add "torch>=2.5.0" "torchvision>=0.20.0" "torchaudio>=2.5.0"
}

# Verify ExecuTorch installation
echo "🔍 Verifying ExecuTorch 0.7.0 installation..."
uv run python -c "import executorch; print('ExecuTorch imported successfully')" || {
    echo "⚠️  ExecuTorch import failed, attempting manual installation..."
    uv add "executorch>=0.4.0"
}

# Create mobile deployment directories
echo "📁 Creating mobile deployment directories..."
mkdir -p mobile_models
mkdir -p mobile_benchmarks
mkdir -p mobile_apps
mkdir -p mobile_bridge

# Create environment activation script
echo "📝 Creating activation script..."
cat > activate_mobile_env.sh << 'EOF'
#!/bin/bash
# Activate PE Core Mobile Environment (UV-based)
cd "$(dirname "$0")"

echo "🚀 PE Core Mobile Environment Activated (UV)"
echo "   Location: $(pwd)"
echo "   PyTorch: $(uv run python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
echo "   ExecuTorch: $(uv run python -c 'import executorch; print("Available")' 2>/dev/null || echo 'Not available')"
echo ""
echo "📋 Available commands:"
echo "   uv run python perception_models/tools/convert_executorch.py --help"
echo "   uv run python perception_models/tools/test_mobile_inference.py --help"
echo "   uv run python perception_models/tools/onnx_to_executorch.py --help"
echo ""
echo "💡 Use 'uv run python <script>' to run commands in this environment"
EOF

chmod +x activate_mobile_env.sh

echo ""
echo "✅ Mobile deployment environment setup complete!"
echo ""
echo "📋 Next steps:"
echo "   1. Activate environment: cd mobile_deployment && source activate_mobile_env.sh"
echo "   2. Convert PE Core models: uv run python perception_models/tools/convert_executorch.py --model PE-Core-T16-384"
echo "   3. Test mobile inference: uv run python perception_models/tools/test_mobile_inference.py --model mobile_models/pe_core_t16_384.pte"
echo ""
echo "🔧 Environment details:"
echo "   Type: UV-managed Python environment"
echo "   Python: 3.11"
echo "   PyTorch: 2.8+ (or latest compatible)"
echo "   ExecuTorch: 0.7+ (or latest compatible)"
echo "   Location: $(pwd)"
echo ""
echo "💡 To use: cd mobile_deployment && uv run python <script>"
