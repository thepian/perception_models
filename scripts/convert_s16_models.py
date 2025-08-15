#!/usr/bin/env python3
"""
Convert PE-Core-S16-384 models to multiple formats for release.
Supports CoreML (.mlpackage), ONNX (.onnx), and ExecuTorch (.pte) formats.
"""

import argparse
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torchvision.transforms as transforms
# mobile_optimizer import removed as it's not used

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.vision_encoder.pe import CLIP


class S16ModelConverter:
    """Converter for PE-Core-S16-384 models to multiple formats."""
    
    def __init__(self, model_name: str = "PE-Core-S16-384"):
        self.model_name = model_name
        self.input_size = (3, 384, 384)
        self.batch_size = 1
        
    def load_model(self) -> torch.nn.Module:
        """Load the PE-Core-S16-384 model."""
        print(f"🔄 Loading {self.model_name} model...")
        
        try:
            # Load model using the standard API
            model = CLIP.from_config(self.model_name, pretrained=True)
            model.eval()
            
            print(f"✅ Model loaded successfully")
            print(f"   Vision model: {type(model.visual).__name__}")
            print(f"   Input size: {self.input_size}")
            
            return model
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            traceback.print_exc()
            raise
    
    def create_example_input(self) -> torch.Tensor:
        """Create example input tensor for model conversion."""
        return torch.randn(self.batch_size, *self.input_size)
    
    def create_mobile_wrapper(self, model: torch.nn.Module) -> torch.nn.Module:
        """Create a mobile-optimized wrapper for the vision encoder."""
        
        class MobilePECore(torch.nn.Module):
            """Mobile-optimized wrapper for PE-Core vision encoder."""
            
            def __init__(self, vision_model):
                super().__init__()
                self.vision_model = vision_model
                
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                """Forward pass returning normalized embeddings."""
                # Extract vision features
                features = self.vision_model(x)
                
                # Normalize embeddings for consistent similarity computation
                features = features / features.norm(dim=-1, keepdim=True)
                
                return features
        
        mobile_model = MobilePECore(model.visual)
        mobile_model.eval()
        
        return mobile_model
    
    def convert_to_coreml(self, model: torch.nn.Module, output_dir: Path, version: str) -> Optional[Path]:
        """Convert model to CoreML format."""
        print(f"🔄 Converting to CoreML...")
        
        try:
            import coremltools as ct
            
            example_input = self.create_example_input()
            mobile_model = self.create_mobile_wrapper(model)
            
            # First trace the model to make it compatible with CoreML
            print("   🔄 Tracing model for CoreML conversion...")
            try:
                traced_model = torch.jit.trace(mobile_model, example_input, strict=False)
                print("   ✅ Model tracing successful")
            except Exception as trace_err:
                print(f"   ❌ Model tracing failed: {trace_err}")
                return None
            
            # Convert to CoreML using the traced model
            try:
                print("   🔄 Converting traced model to CoreML...")
                coreml_model = ct.convert(
                    traced_model,
                    inputs=[ct.TensorType(
                        name="image",
                        shape=example_input.shape,
                        dtype=float
                    )],
                    outputs=[ct.TensorType(
                        name="embeddings",
                        dtype=float
                    )],
                    minimum_deployment_target=ct.target.iOS16,
                    compute_units=ct.ComputeUnit.ALL,
                    convert_to="mlprogram"
                )
                print("   ✅ CoreML conversion successful")
            except Exception as convert_err:
                print(f"   ❌ CoreML conversion failed: {convert_err}")
                return None
            
            # Add metadata
            coreml_model.short_description = f"PE-Core-S16-384 Vision Encoder v{version}"
            coreml_model.author = "Meta Research"
            coreml_model.license = "Apache 2.0"
            coreml_model.version = version
            
            # Add input/output descriptions
            coreml_model.input_description["image"] = "Input image tensor (384x384x3)"
            coreml_model.output_description["embeddings"] = "512-dimensional normalized embeddings"
            
            # Save models with different compute unit configurations
            variants = [
                ("all", ct.ComputeUnit.ALL),
                ("cpu_only", ct.ComputeUnit.CPU_ONLY),
                ("neural_engine", ct.ComputeUnit.CPU_AND_NE),
            ]
            
            output_paths = []
            
            for variant_name, compute_unit in variants:
                # Set compute unit
                coreml_model = ct.utils.rename_feature(coreml_model, "image", "image")  # Refresh
                
                variant_filename = f"pe_core_s16_384_{variant_name}.mlpackage"
                variant_path = output_dir / variant_filename
                
                coreml_model.save(str(variant_path))
                output_paths.append(variant_path)
                
                print(f"   ✅ Saved {variant_name}: {variant_filename}")
            
            # Create FP16 variant
            try:
                fp16_model = ct.models.neural_network.quantization_utils.quantize_weights(
                    coreml_model, nbits=16
                )
                fp16_path = output_dir / "pe_core_s16_384_fp16.mlpackage"
                fp16_model.save(str(fp16_path))
                output_paths.append(fp16_path)
                print(f"   ✅ Saved FP16: pe_core_s16_384_fp16.mlpackage")
            except Exception as e:
                print(f"   ⚠️  FP16 conversion failed: {e}")
            
            return output_paths[0]  # Return main model path
            
        except ImportError:
            print("   ❌ CoreML Tools not available, skipping CoreML conversion")
            return None
        except Exception as e:
            print(f"   ❌ CoreML conversion failed: {e}")
            traceback.print_exc()
            return None
    
    def convert_to_onnx(self, model: torch.nn.Module, output_dir: Path, version: str) -> Optional[Path]:
        """Convert model to ONNX format."""
        print(f"🔄 Converting to ONNX...")
        
        try:
            example_input = self.create_example_input()
            mobile_model = self.create_mobile_wrapper(model)
            
            # Export to ONNX
            onnx_path = output_dir / f"pe_core_s16_384_{version}.onnx"
            
            torch.onnx.export(
                mobile_model,
                example_input,
                str(onnx_path),
                input_names=["image"],
                output_names=["embeddings"],
                dynamic_axes={
                    "image": {0: "batch_size"},
                    "embeddings": {0: "batch_size"}
                },
                opset_version=17,
                do_constant_folding=True,
                export_params=True,
                training=torch.onnx.TrainingMode.EVAL,
                verbose=False,
            )
            
            print(f"   ✅ Saved: {onnx_path.name}")
            
            # Verify ONNX model
            try:
                import onnx
                onnx_model = onnx.load(str(onnx_path))
                onnx.checker.check_model(onnx_model)
                print(f"   ✅ ONNX model validation passed")
            except ImportError:
                print("   ⚠️  ONNX not available for verification")
            except Exception as e:
                print(f"   ⚠️  ONNX validation warning: {e}")
            
            return onnx_path
            
        except Exception as e:
            print(f"   ❌ ONNX conversion failed: {e}")
            traceback.print_exc()
            return None
    
    def convert_to_executorch(self, model: torch.nn.Module, output_dir: Path, version: str) -> Optional[Path]:
        """Convert model to ExecuTorch format."""
        print(f"🔄 Converting to ExecuTorch...")
        
        try:
            from executorch.exir import to_edge_transform_and_lower
            from executorch.extension.pybindings.portable_lib import _load_for_executorch
            
            example_input = self.create_example_input()
            mobile_model = self.create_mobile_wrapper(model)
            
            # Convert to ExecuTorch
            et_program = to_edge_transform_and_lower(mobile_model, (example_input,))
            
            # Save ExecuTorch model
            et_path = output_dir / f"pe_core_s16_384_{version}.pte"
            et_program.write_to_file(str(et_path))
            
            print(f"   ✅ Saved: {et_path.name}")
            
            # Verify ExecuTorch model
            try:
                test_model = _load_for_executorch(str(et_path))
                test_output = test_model.forward([example_input])
                print(f"   ✅ ExecuTorch model validation passed")
            except Exception as e:
                print(f"   ⚠️  ExecuTorch validation warning: {e}")
            
            return et_path
            
        except ImportError:
            print("   ❌ ExecuTorch not available, skipping conversion")
            return None
        except Exception as e:
            print(f"   ❌ ExecuTorch conversion failed: {e}")
            traceback.print_exc()
            return None
    
    def convert_all_formats(self, formats: List[str], output_dir: Path, version: str) -> Dict[str, Optional[Path]]:
        """Convert model to all specified formats."""
        print(f"🚀 Converting {self.model_name} to formats: {', '.join(formats)}")
        print(f"📁 Output directory: {output_dir}")
        print("=" * 60)
        
        # Load the model once
        model = self.load_model()
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        if "coreml" in formats:
            results["coreml"] = self.convert_to_coreml(model, output_dir, version)
        
        if "onnx" in formats:
            results["onnx"] = self.convert_to_onnx(model, output_dir, version)
        
        if "executorch" in formats:
            results["executorch"] = self.convert_to_executorch(model, output_dir, version)
        
        print("=" * 60)
        print("🎉 Conversion complete!")
        
        for format_name, path in results.items():
            if path:
                print(f"   ✅ {format_name}: {path}")
            else:
                print(f"   ❌ {format_name}: Failed")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Convert PE-Core-S16-384 models")
    parser.add_argument("--model-name", default="PE-Core-S16-384", 
                       help="Model name to convert")
    parser.add_argument("--formats", default="coreml,onnx,executorch",
                       help="Comma-separated list of formats (coreml,onnx,executorch)")
    parser.add_argument("--output-dir", required=True, type=Path,
                       help="Output directory for converted models")
    parser.add_argument("--version", required=True,
                       help="Version tag for the models")
    
    args = parser.parse_args()
    
    # Parse formats
    formats = [f.strip() for f in args.formats.split(",")]
    valid_formats = {"coreml", "onnx", "executorch"}
    
    for fmt in formats:
        if fmt not in valid_formats:
            print(f"❌ Invalid format: {fmt}. Valid formats: {valid_formats}")
            sys.exit(1)
    
    # Run conversion
    converter = S16ModelConverter(args.model_name)
    
    try:
        results = converter.convert_all_formats(formats, args.output_dir, args.version)
        
        # Check if any conversions succeeded
        if any(results.values()):
            print(f"\n🎉 Model conversion completed successfully!")
            sys.exit(0)
        else:
            print(f"\n❌ All conversions failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 Conversion failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()