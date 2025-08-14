#!/usr/bin/env python3
"""
CoreML Conversion Tool for PE Core Models

Converts Facebook's Perception Encoder Core models to CoreML format
optimized for iOS deployment with live camera classification.

Features:
- PE Core model conversion with proper input/output handling
- Multiple precision options (FP32, FP16, INT8)
- iOS-optimized model variants (Tiny, Small, Base)
- Batch conversion for multiple models
- Validation and performance testing
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import warnings

import torch
import numpy as np
from PIL import Image

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.vision_encoder.pe import VisionTransformer

try:
    import coremltools as ct
    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False
    print("❌ CoreML Tools not available. Install with: uv sync --extra coreml")

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


class CoreMLConverter:
    """Converts PE Core models to CoreML format for iOS deployment."""
    
    def __init__(self):
        self.mobile_optimized_models = [
            "PE-Core-T16-384",  # Tiny - most mobile-friendly
            "PE-Core-S16-384",  # Small - good mobile candidate
            "PE-Core-B16-224",  # Base - may work on newer devices
        ]
        
        self.precision_options = {
            "fp32": ct.precision.FLOAT32,
            "fp16": ct.precision.FLOAT16,
        }
        
        self.compute_units = {
            "all": ct.ComputeUnit.ALL,
            "cpu": ct.ComputeUnit.CPU_ONLY,
            "neural_engine": ct.ComputeUnit.CPU_AND_NE,
            "gpu": ct.ComputeUnit.CPU_AND_GPU,
        }
    
    def load_pe_core_model(self, model_name: str) -> torch.nn.Module:
        """Load PE Core model for conversion."""
        print(f"📦 Loading {model_name}...")
        
        try:
            # Load PE Core model
            model = VisionTransformer.from_config(model_name, pretrained=True)
            
            # Ensure model is on CPU for CoreML conversion
            model = model.cpu()
            model.eval()
            
            # Disable gradients
            for param in model.parameters():
                param.requires_grad = False
            
            print(f"✅ {model_name} loaded successfully")
            return model
            
        except Exception as e:
            print(f"❌ Failed to load {model_name}: {e}")
            return None
    
    def get_model_input_size(self, model_name: str) -> int:
        """Get the expected input size for a PE Core model."""
        # Extract input size from model name
        if "224" in model_name:
            return 224
        elif "336" in model_name:
            return 336
        elif "384" in model_name:
            return 384
        elif "448" in model_name:
            return 448
        else:
            return 224  # Default fallback
    
    def create_example_input(self, model_name: str) -> torch.Tensor:
        """Create example input tensor for the model."""
        input_size = self.get_model_input_size(model_name)
        return torch.randn(1, 3, input_size, input_size).cpu()
    
    def create_mobile_wrapper(self, model: torch.nn.Module, model_name: str) -> torch.nn.Module:
        """Create a mobile-friendly wrapper that eliminates dynamic behavior."""

        class MobilePEWrapper(torch.nn.Module):
            def __init__(self, pe_model, input_size):
                super().__init__()
                self.pe_model = pe_model
                self.input_size = input_size

                # Pre-compute fixed values to avoid dynamic operations
                self.register_buffer('dummy_input', torch.randn(1, 3, input_size, input_size))

                # Set model to eval mode and disable dynamic behavior
                self.pe_model.eval()

            def forward(self, x):
                # Ensure input is the expected size
                if x.shape != self.dummy_input.shape:
                    x = torch.nn.functional.interpolate(x, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)

                # Forward through PE model
                features = self.pe_model(x)

                # Ensure output is 2D for mobile deployment
                if len(features.shape) > 2:
                    features = features.view(features.shape[0], -1)

                return features

        input_size = self.get_model_input_size(model_name)
        wrapper = MobilePEWrapper(model, input_size)
        return wrapper

    def convert_to_coreml(self, model: torch.nn.Module, model_name: str,
                         precision: str = "fp16", compute_unit: str = "all",
                         output_dir: str = "coreml_models") -> Optional[Path]:
        """Convert PyTorch model to CoreML format."""
        if not COREML_AVAILABLE:
            print("❌ CoreML Tools not available")
            return None

        print(f"🔄 Converting {model_name} to CoreML ({precision})...")

        try:
            # Create mobile wrapper to eliminate dynamic behavior
            print("   Creating mobile wrapper...")
            mobile_model = self.create_mobile_wrapper(model, model_name)

            # Create example input
            example_input = self.create_example_input(model_name)
            input_size = example_input.shape[-1]

            print(f"   Input shape: {example_input.shape}")
            print(f"   Precision: {precision}")
            print(f"   Compute unit: {compute_unit}")

            # Test the wrapper first
            print("   Testing mobile wrapper...")
            with torch.no_grad():
                test_output = mobile_model(example_input)
                print(f"   Wrapper output shape: {test_output.shape}")

            # Convert via ONNX for better compatibility
            print("   Converting via ONNX...")
            onnx_path = Path(output_dir) / f"{model_name.lower().replace('-', '_')}_temp.onnx"
            onnx_path.parent.mkdir(exist_ok=True)

            # Export to ONNX
            torch.onnx.export(
                mobile_model,
                example_input,
                str(onnx_path),
                export_params=True,
                opset_version=17,
                do_constant_folding=True,
                input_names=['image'],
                output_names=['features'],
                dynamic_axes={
                    'image': {0: 'batch_size'},
                    'features': {0: 'batch_size'}
                }
            )

            # Load ONNX model and convert to CoreML
            print("   Loading ONNX model...")
            import onnx
            onnx_model = onnx.load(str(onnx_path))

            print("   Converting ONNX to CoreML...")
            coreml_model = ct.convert(
                onnx_model,
                inputs=[ct.TensorType(
                    shape=example_input.shape,
                    name="image"
                )],
                outputs=[ct.TensorType(name="features")],
                compute_precision=self.precision_options[precision],
                compute_units=self.compute_units[compute_unit],
                convert_to="mlprogram"  # Use ML Program format for iOS 15+
            )

            # Clean up temporary ONNX file
            onnx_path.unlink()

            # Add metadata
            coreml_model.short_description = f"PE Core {model_name} for live camera classification"
            coreml_model.author = "Meta Research"
            coreml_model.license = "MIT"
            coreml_model.version = "1.0"

            # Add input/output descriptions
            coreml_model.input_description["image"] = f"Input image ({input_size}x{input_size} RGB)"
            coreml_model.output_description["features"] = "Image features for classification"

            # Save model
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)

            model_filename = f"{model_name.lower().replace('-', '_')}_{precision}.mlpackage"
            model_path = output_path / model_filename

            print(f"   Saving to: {model_path}")
            coreml_model.save(str(model_path))

            print(f"✅ CoreML conversion successful: {model_path}")
            return model_path

        except Exception as e:
            print(f"❌ CoreML conversion failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def validate_coreml_model(self, model_path: Path, original_model: torch.nn.Module,
                             model_name: str) -> Dict[str, Any]:
        """Validate CoreML model against original PyTorch model."""
        if not COREML_AVAILABLE:
            return {'error': 'CoreML Tools not available'}
        
        print(f"🔍 Validating CoreML model: {model_path.name}")
        
        try:
            # Load CoreML model
            coreml_model = ct.models.MLModel(str(model_path))
            
            # Create test input
            example_input = self.create_example_input(model_name)
            test_image = example_input.numpy()
            
            # Get PyTorch output
            with torch.no_grad():
                pytorch_output = original_model(example_input).numpy()
            
            # Get CoreML output
            coreml_input = {"image": test_image}
            coreml_output = coreml_model.predict(coreml_input)
            coreml_features = coreml_output["features"]
            
            # Compare outputs
            mse = np.mean((pytorch_output - coreml_features) ** 2)
            max_diff = np.max(np.abs(pytorch_output - coreml_features))
            
            # Get model size
            model_size_mb = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file()) / (1024 * 1024)
            
            validation_result = {
                'model_path': str(model_path),
                'model_size_mb': model_size_mb,
                'mse': float(mse),
                'max_difference': float(max_diff),
                'pytorch_shape': pytorch_output.shape,
                'coreml_shape': coreml_features.shape,
                'validation_passed': mse < 1e-4 and max_diff < 1e-3
            }
            
            print(f"   Model size: {model_size_mb:.1f}MB")
            print(f"   MSE: {mse:.2e}")
            print(f"   Max diff: {max_diff:.2e}")
            print(f"   Validation: {'✅ PASSED' if validation_result['validation_passed'] else '❌ FAILED'}")
            
            return validation_result
            
        except Exception as e:
            print(f"   ❌ Validation failed: {e}")
            return {'error': str(e)}
    
    def convert_mobile_optimized_models(self, precision: str = "fp16", 
                                      compute_unit: str = "all",
                                      output_dir: str = "coreml_models") -> Dict[str, Any]:
        """Convert all mobile-optimized PE Core models."""
        print("🚀 PE CORE COREML CONVERSION")
        print("=" * 60)
        print("Converting mobile-optimized PE Core models:")
        for model_name in self.mobile_optimized_models:
            print(f"- {model_name}")
        print("=" * 60)
        
        results = {
            'conversion_info': {
                'precision': precision,
                'compute_unit': compute_unit,
                'output_directory': output_dir
            },
            'models': [],
            'summary': {}
        }
        
        successful_conversions = 0
        
        for model_name in self.mobile_optimized_models:
            print(f"\n{'='*20} {model_name} {'='*20}")
            
            # Load model
            model = self.load_pe_core_model(model_name)
            if model is None:
                continue
            
            # Convert to CoreML
            model_path = self.convert_to_coreml(
                model, model_name, precision, compute_unit, output_dir
            )
            
            if model_path is None:
                continue
            
            # Validate conversion
            validation_result = self.validate_coreml_model(model_path, model, model_name)
            
            # Store results
            model_result = {
                'model_name': model_name,
                'conversion_successful': True,
                'model_path': str(model_path),
                'validation': validation_result
            }
            
            results['models'].append(model_result)
            
            if validation_result.get('validation_passed', False):
                successful_conversions += 1
                print(f"   🎉 {model_name} conversion completed successfully!")
            else:
                print(f"   ⚠️  {model_name} converted but validation failed")
        
        # Generate summary
        results['summary'] = {
            'total_models': len(self.mobile_optimized_models),
            'successful_conversions': successful_conversions,
            'conversion_rate': successful_conversions / len(self.mobile_optimized_models) * 100,
            'output_directory': output_dir
        }
        
        return results


def main():
    parser = argparse.ArgumentParser(description="PE Core CoreML Conversion Tool")
    parser.add_argument("--model", choices=["T16", "S16", "B16", "all"], default="all",
                        help="Specific model to convert (default: all mobile-optimized)")
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp16",
                        help="Model precision (default: fp16)")
    parser.add_argument("--compute-unit", choices=["all", "cpu", "neural_engine", "gpu"], 
                        default="all", help="Target compute unit (default: all)")
    parser.add_argument("--output-dir", default="coreml_models",
                        help="Output directory for CoreML models")
    parser.add_argument("--validate", action="store_true",
                        help="Validate converted models")
    
    args = parser.parse_args()
    
    if not COREML_AVAILABLE:
        print("❌ CoreML Tools not available. Install with: uv sync --extra coreml")
        return 1
    
    # Check if we're on macOS
    import platform
    if sys.platform != 'darwin':
        print("⚠️  CoreML conversion works best on macOS")
    
    # Initialize converter
    converter = CoreMLConverter()
    
    if args.model != "all":
        # Convert specific model
        model_name = f"PE-Core-{args.model}-384" if args.model in ["T16", "S16"] else f"PE-Core-{args.model}-224"
        converter.mobile_optimized_models = [model_name]
    
    # Run conversion
    results = converter.convert_mobile_optimized_models(
        args.precision, args.compute_unit, args.output_dir
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 COREML CONVERSION SUMMARY")
    print("=" * 60)
    
    summary = results['summary']
    print(f"Models processed: {summary['total_models']}")
    print(f"Successful conversions: {summary['successful_conversions']}")
    print(f"Success rate: {summary['conversion_rate']:.1f}%")
    print(f"Output directory: {summary['output_directory']}")
    
    if summary['successful_conversions'] > 0:
        print(f"\n🎉 CoreML models ready for iOS deployment!")
        print(f"📱 Use these models in your iOS app for live camera classification")
    else:
        print(f"\n❌ No successful conversions")
    
    return 0 if summary['successful_conversions'] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
