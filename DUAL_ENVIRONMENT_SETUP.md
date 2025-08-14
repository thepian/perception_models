# Dual Environment Setup Guide
## PyTorch 2.2.x (Franca-compatible) + PyTorch 2.8 (ExecuTorch 0.7.0)

This guide sets up a dual environment strategy for PE Core mobile deployment while maintaining Franca compatibility.

---

## 🎯 Strategy Overview

**Environment 1: Core Development (Current)**
- PyTorch 2.2.x (Franca-compatible)
- PE Core model development and testing
- Export to ONNX format

**Environment 2: Mobile Deployment (New)**
- PyTorch 2.8 + ExecuTorch 0.7.0
- ONNX → ExecuTorch conversion
- Mobile optimization and deployment

---

## 🚀 Quick Setup

### **Step 1: Create Mobile Environment**

```bash
cd /Volumes/Projects/Evidently/perception_models

# Make setup script executable
chmod +x setup_mobile_env.sh

# Run setup (takes 5-10 minutes)
./setup_mobile_env.sh
```

### **Step 2: Activate Mobile Environment**

```bash
# Option A: Use activation script
source activate_mobile_env.sh

# Option B: Manual activation
conda activate pe-core-mobile
```

### **Step 3: Verify Installation**

```bash
# Check PyTorch 2.8
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Check ExecuTorch 0.7.0
python -c "import executorch; print('ExecuTorch: Available')"

# Check CUDA (if available)
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 🔄 Dual Environment Workflow

### **Phase 1: Core Development (Current Environment)**

```bash
# Stay in current environment (PyTorch 2.2.x)
cd /Volumes/Projects/Evidently/perception_models

# Develop and test PE Core models
uv run python perception_models/tools/convert.py --model T16 --format onnx

# Export to ONNX for mobile bridge
uv run python perception_models/tools/convert.py --model T16 --format onnx --output mobile_bridge/
```

### **Phase 2: Mobile Deployment (Mobile Environment)**

```bash
# Switch to mobile environment
conda activate pe-core-mobile

# Convert ONNX to ExecuTorch
python perception_models/tools/onnx_to_executorch.py \
    --input mobile_bridge/pe_core_t16.onnx \
    --output mobile_models/pe_core_t16.pte \
    --backend xnnpack

# Test mobile inference
python perception_models/tools/test_mobile_inference.py \
    --model mobile_models/pe_core_t16.pte \
    --benchmark --iterations 1000
```

---

## 📋 Environment Details

### **Core Environment (Current)**
```yaml
Name: perception-models (uv managed)
Python: 3.13.3
PyTorch: 2.2.0-2.3.0
Purpose: PE Core development, Franca compatibility
Location: .venv/
```

### **Mobile Environment (New)**
```yaml
Name: pe-core-mobile
Python: 3.11
PyTorch: 2.8.0
ExecuTorch: 0.7.0
Purpose: Mobile deployment, ExecuTorch conversion
Location: $(conda info --base)/envs/pe-core-mobile
```

---

## 🛠 Available Tools

### **Core Environment Tools**
```bash
# PE Core model conversion (current)
uv run python perception_models/tools/convert.py --help

# ONNX export for mobile bridge
uv run python perception_models/tools/convert.py --model T16 --format onnx
```

### **Mobile Environment Tools**
```bash
# ONNX to ExecuTorch conversion
python perception_models/tools/onnx_to_executorch.py --help

# Mobile inference testing
python perception_models/tools/test_mobile_inference.py --help

# ExecuTorch model conversion (direct)
python perception_models/tools/convert_executorch.py --help
```

---

## 🎯 Conversion Pipeline

### **Complete Workflow Example**

```bash
# 1. Core Environment: PE Core → ONNX
uv run python perception_models/tools/convert.py \
    --model PE-Core-T16-384 \
    --format onnx \
    --output mobile_bridge/

# 2. Switch to Mobile Environment
conda activate pe-core-mobile

# 3. Mobile Environment: ONNX → ExecuTorch
python perception_models/tools/onnx_to_executorch.py \
    --input mobile_bridge/pe_core_t16_384.onnx \
    --output mobile_models/pe_core_t16_384.pte \
    --backend xnnpack \
    --quantization fp16

# 4. Test Mobile Performance
python perception_models/tools/test_mobile_inference.py \
    --model mobile_models/pe_core_t16_384.pte \
    --benchmark \
    --iterations 1000 \
    --accuracy-test \
    --reference-model PE-Core-T16-384
```

---

## 📊 Expected Performance

### **Target Metrics**
- **Inference Time**: 5-15ms (vs 133ms current)
- **Frame Rate**: 30+ FPS sustained
- **Memory Usage**: < 100MB total
- **Accuracy**: > 99% similarity to reference

### **Benchmark Commands**
```bash
# Quick performance test
python perception_models/tools/test_mobile_inference.py \
    --model mobile_models/pe_core_t16_384.pte \
    --benchmark --iterations 100

# Comprehensive accuracy test
python perception_models/tools/test_mobile_inference.py \
    --model mobile_models/pe_core_t16_384.pte \
    --accuracy-test \
    --reference-model PE-Core-T16-384
```

---

## 🔧 Troubleshooting

### **Common Issues**

**1. Environment Activation Fails**
```bash
# Ensure conda is properly initialized
conda init
source ~/.bashrc  # or ~/.zshrc
```

**2. ExecuTorch Import Error**
```bash
# Reinstall ExecuTorch in mobile environment
conda activate pe-core-mobile
pip install executorch==0.7.0 --force-reinstall
```

**3. ONNX Conversion Issues**
```bash
# Check ONNX model validity
python -c "import onnx; onnx.checker.check_model(onnx.load('model.onnx'))"
```

**4. PyTorch Version Conflicts**
```bash
# Verify correct environment
python -c "import torch; print(torch.__version__)"
# Should show 2.8.0 in mobile environment
```

### **Environment Switching**

```bash
# Switch to core environment
conda deactivate
cd /Volumes/Projects/Evidently/perception_models
# Use uv run for commands

# Switch to mobile environment
conda activate pe-core-mobile
# Use python directly for commands
```

---

## 📋 Next Steps

### **Immediate Actions**
1. ✅ **Run setup script**: `./setup_mobile_env.sh`
2. 📋 **Test environment**: Verify PyTorch 2.8 + ExecuTorch 0.7.0
3. 📋 **Convert first model**: PE-Core-T16-384 → ONNX → ExecuTorch
4. 📋 **Benchmark performance**: Validate 5-15ms target

### **Validation Phase**
1. 📋 **Performance testing**: Compare with current approach
2. 📋 **Accuracy validation**: Ensure model quality preservation
3. 📋 **Mobile integration**: Test on actual iOS devices
4. 📋 **Stability assessment**: Long-term testing

### **Success Criteria**
- ✅ **Environment setup**: Both environments working
- 📋 **Conversion pipeline**: ONNX → ExecuTorch working
- 📋 **Performance target**: < 15ms inference achieved
- 📋 **Accuracy target**: > 99% similarity maintained

---

## 💡 Tips

**Environment Management**:
- Use `conda env list` to see all environments
- Use `which python` to verify active environment
- Keep environments separate to avoid conflicts

**Development Workflow**:
- Develop in core environment (Franca-compatible)
- Deploy in mobile environment (ExecuTorch)
- Use ONNX as bridge format between environments

**Performance Optimization**:
- Start with XNNPACK backend (most compatible)
- Try FP16 quantization for speed/size balance
- Test on actual mobile devices for realistic benchmarks

This dual environment approach provides the best of both worlds: Franca compatibility and cutting-edge mobile performance!
