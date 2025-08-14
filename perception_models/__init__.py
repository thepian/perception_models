"""
Perception Models - Facebook's Perception Encoder with Mobile Deployment

A modern Python package for deploying Facebook's Perception Encoder models
on mobile devices, with special focus on iOS CoreML deployment.

Key Features:
- PE Core: CLIP-style zero-shot classification
- PE Lang: LLM-aligned multimodal encoder  
- PE Spatial: Dense prediction for object detection
- CoreML export for iOS deployment
- Mobile optimization and quantization
- Real-time inference benchmarking
"""

__version__ = "0.1.0"
__author__ = "Meta Research"
__email__ = "research@meta.com"

# Core imports
from core.vision_encoder.pe import VisionTransformer
from core.vision_encoder.config import PE_VISION_CONFIG

# Mobile deployment utilities
try:
    from .tools.convert import CoreMLConverter
    MOBILE_TOOLS_AVAILABLE = True
except ImportError:
    MOBILE_TOOLS_AVAILABLE = False

__all__ = [
    "VisionTransformer",
    "PE_VISION_CONFIG",
    "__version__",
]

if MOBILE_TOOLS_AVAILABLE:
    __all__.extend(["CoreMLConverter"])


def get_available_models():
    """Get list of available PE models."""
    return list(PE_VISION_CONFIG.keys())


def get_mobile_optimized_models():
    """Get list of models optimized for mobile deployment."""
    mobile_models = []
    for model_name in PE_VISION_CONFIG.keys():
        # Consider Tiny and Small models as mobile-optimized
        if any(size in model_name for size in ["T16", "S16"]):
            mobile_models.append(model_name)
    return mobile_models


def load_model(model_name: str, pretrained: bool = True):
    """
    Load a PE model by name.
    
    Args:
        model_name: Name of the model (e.g., 'PE-Core-T16-384')
        pretrained: Whether to load pretrained weights
        
    Returns:
        VisionTransformer model
    """
    return VisionTransformer.from_config(model_name, pretrained=pretrained)


# Version info
def get_version_info():
    """Get detailed version information."""
    import torch
    import sys
    
    info = {
        "perception_models": __version__,
        "python": sys.version,
        "torch": torch.__version__,
        "mobile_tools": MOBILE_TOOLS_AVAILABLE,
    }
    
    try:
        import coremltools
        info["coremltools"] = coremltools.__version__
    except ImportError:
        info["coremltools"] = "Not installed"
    
    return info
