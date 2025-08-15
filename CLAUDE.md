# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is Meta Research's Perception Models repository, featuring state-of-the-art vision encoders (PE) and vision-language models (PLM) built on PyTorch. The project supports mobile deployment via CoreML and ExecuTorch.

## Development Commands

### Setup
```bash
# Create conda environment (Python 3.12)
conda create --name perception_models python=3.12
conda activate perception_models

# Install PyTorch 2.5.1 with CUDA
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 xformers --index-url https://download.pytorch.org/whl/cu124

# Install for development
pip install -e ".[dev]"

# Mobile deployment setup
./setup_mobile_simple.sh
```

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=core --cov=apps --cov-report=term-missing

# Run specific test types
pytest -m "not slow"  # Skip slow tests
pytest -m gpu         # GPU tests only
pytest -m coreml      # CoreML tests only

# Run single test file
pytest core/tests/dataloader_test.py
```

### Code Quality
```bash
# Format code
black core apps perception_models
isort core apps perception_models

# Lint
ruff check core apps perception_models

# Type check
mypy core apps perception_models
```

### Build
```bash
# Build Python wheel
python -m build
```

## Architecture Overview

### Core Structure
- `core/`: Shared components for all perception models
  - `vision_encoder/`: PE (Perception Encoder) implementations with Vision Transformer
  - `transformer.py`: Base transformer with flexible attention (SDPA, xFormers, FlexAttention)
  - `vision_projector/`: Vision-to-language projection layers
  - `data/`: JSONL-based data loading with multi-process support
  - `transforms/`: Image/video preprocessing pipelines

### Applications
- `apps/pe/`: Standalone Perception Encoder models (Core, Lang, Spatial variants)
- `apps/plm/`: Perception Language Models (1B, 3B, 8B) combining PE with LLMs
- `apps/detection/`: Object detection using PE backbones (DETA, Detectron2)

### Mobile Deployment
- `perception_models/tools/`: Conversion utilities for mobile deployment
- `coreml_models/`: Pre-converted CoreML models for iOS
- `apps/ios/docs/`: Complete iOS integration documentation and examples
- `apps/ios/docs/`: Mobile deployment guides and recommendations
- Supports CoreML, ONNX, and ExecuTorch (PyTorch 2.8)

### Key Patterns
- **Config-based model creation**: Models defined via `PEConfig` dataclasses
- **Modular transforms**: Composable image/video preprocessing
- **From-config loading**: `model = CLIP.from_config("PE-Core-T16-384", pretrained=True)`
- **HuggingFace integration**: Models hosted on HF hub (facebook/perception-encoder-*)

### Model Variants
- PE Core: CLIP-style vision-language models (T16 to G14 sizes)
- PE Lang: LLM-aligned encoders for multimodal tasks
- PE Spatial: Dense prediction optimized encoders
- PLM: Full multimodal language models

### Important Notes
- Primary platform is macOS/Darwin with mobile deployment focus
- Supports PyTorch 2.8+ with ExecuTorch for mobile deployment
- **Recommended mobile model**: PE-Core-S16-384 (optimal balance: 25ms latency, 72.7% accuracy)
- Models use attention pooling or CLS tokens for global features
- RoPE (Rotary Position Embeddings) available for 2D vision tasks
- See `docs/MOBILE_DEPLOYMENT.md` for mobile deployment guide
- See `apps/ios/docs/` for complete iOS integration examples