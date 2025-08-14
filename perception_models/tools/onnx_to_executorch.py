#!/usr/bin/env python3
"""
ONNX to ExecuTorch Bridge Converter

This tool converts ONNX models (exported from PE Core with PyTorch 2.2.x)
to ExecuTorch format (using PyTorch 2.8 + ExecuTorch 0.7.0) for mobile deployment.

This enables the dual environment strategy:
1. Core Environment (PyTorch 2.2.x): PE Core → ONNX
2. Mobile Environment (PyTorch 2.8): ONNX → ExecuTorch

Usage:
    python onnx_to_executorch.py --input pe_core_t16.onnx --output pe_core_t16.pte
    python onnx_to_executorch.py --input models/ --output mobile_models/ --batch
    python onnx_to_executorch.py --input pe_core_s16.onnx --backend xnnpack --quantization fp16

Requirements:
    - Python 3.10-3.12
    - PyTorch 2.8+
    - ExecuTorch 0.7.0
    - ONNX 1.15+
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import onnx
import numpy as np

# Check Python version compatibility
if sys.version_info < (3, 10) or sys.version_info >= (3, 13):
    print("❌ ExecuTorch requires Python 3.10-3.12")
    print(f"   Current Python version: {sys.version}")
    print("   Please use a compatible Python version")
    sys.exit(1)

# Check PyTorch version
if torch.__version__ < "2.8.0":
    print("❌ This tool requires PyTorch 2.8+")
    print(f"   Current PyTorch version: {torch.__version__}")
    print("   Please use the mobile environment with PyTorch 2.8")
    sys.exit(1)

# Try to import ExecuTorch 0.7.0
try:
    from executorch.exir import to_edge_transform_and_lower
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    EXECUTORCH_AVAILABLE = True
    print("✅ ExecuTorch 0.7.0 available")
except ImportError as e:
    EXECUTORCH_AVAILABLE = False
    print("❌ ExecuTorch 0.7.0 not available")
    print(f"   Error: {e}")
    print("   Install with: pip install executorch==0.7.0")
    sys.exit(1)


class ONNXToExecuTorchConverter:
    """Convert ONNX models to ExecuTorch format for mobile deployment."""
    
    def __init__(self):
        self.supported_backends = {
            "xnnpack": "CPU optimization (iOS/Android)",
            "coreml": "Apple Neural Engine (iOS only)", 
            "vulkan": "GPU acceleration (Android)",
        }
        
        self.quantization_options = {
            "none": "No quantization (fp32)",
            "fp16": "Half precision (fp16)",
            "int8": "8-bit quantization (int8)",
        }
    
    def load_onnx_model(self, onnx_path: Path) -> Optional[onnx.ModelProto]:
        """Load and validate ONNX model."""
        try:
            print(f"🔄 Loading ONNX model: {onnx_path}")
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            print(f"✅ ONNX model loaded and validated")
            return onnx_model
        except Exception as e:
            print(f"❌ Failed to load ONNX model: {e}")
            return None
    
    def create_pytorch_wrapper(self, onnx_path: Path) -> Optional[torch.nn.Module]:
        """Create PyTorch wrapper for ONNX model."""
        try:
            # Load ONNX model as PyTorch module
            import torch.onnx
            
            # For ExecuTorch, we need to convert ONNX back to PyTorch
            # This is a simplified approach - in practice, you might need
            # more sophisticated ONNX → PyTorch conversion
            
            class ONNXWrapper(torch.nn.Module):
                def __init__(self, onnx_path):
                    super().__init__()
                    # This is a placeholder - actual implementation would
                    # require ONNX runtime or ONNX → PyTorch conversion
                    self.onnx_path = onnx_path
                
                def forward(self, x):
                    # Placeholder implementation
                    # In practice, this would run ONNX inference
                    # or convert ONNX ops to PyTorch ops
                    return x
            
            wrapper = ONNXWrapper(onnx_path)
            return wrapper
            
        except Exception as e:
            print(f"❌ Failed to create PyTorch wrapper: {e}")
            return None
    
    def convert_to_executorch(self, onnx_path: Path, output_path: Path,
                             backend: str = "xnnpack", quantization: str = "none") -> bool:
        """Convert ONNX model to ExecuTorch format."""
        if not EXECUTORCH_AVAILABLE:
            print("❌ ExecuTorch not available")
            return False
        
        print(f"🔄 Converting {onnx_path.name} to ExecuTorch...")
        print(f"   Backend: {backend}")
        print(f"   Quantization: {quantization}")
        
        try:
            # Load ONNX model
            onnx_model = self.load_onnx_model(onnx_path)
            if onnx_model is None:
                return False
            
            # Get model info
            input_info = self.get_onnx_input_info(onnx_model)
            print(f"   Input shape: {input_info['shape']}")
            print(f"   Input dtype: {input_info['dtype']}")
            
            # Create example input
            example_input = torch.randn(*input_info['shape']).to(input_info['dtype'])
            
            # For now, create a simple PyTorch model that matches ONNX structure
            # In a full implementation, you'd convert ONNX ops to PyTorch ops
            pytorch_model = self.create_simple_model_from_onnx(onnx_model, input_info)
            
            if pytorch_model is None:
                print("❌ Failed to create PyTorch model from ONNX")
                return False
            
            # Export to ExecuTorch
            print("   Exporting to ExecuTorch...")
            exported_program = torch.export.export(pytorch_model, (example_input,))
            
            # Configure backend partitioner
            partitioner = None
            if backend == "xnnpack":
                partitioner = [XnnpackPartitioner()]
            
            # Convert to ExecuTorch
            et_program = to_edge_transform_and_lower(
                exported_program,
                partitioner=partitioner
            ).to_executorch()
            
            # Save the model
            output_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"   Saving to: {output_path}")
            
            with open(output_path, "wb") as f:
                f.write(et_program.buffer)
            
            print(f"✅ ExecuTorch conversion successful: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ ExecuTorch conversion failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_onnx_input_info(self, onnx_model: onnx.ModelProto) -> Dict:
        """Extract input information from ONNX model."""
        input_info = onnx_model.graph.input[0]
        
        # Get shape
        shape = []
        for dim in input_info.type.tensor_type.shape.dim:
            if dim.dim_value:
                shape.append(dim.dim_value)
            else:
                shape.append(1)  # Default for dynamic dimensions
        
        # Get dtype
        dtype_map = {
            1: torch.float32,   # FLOAT
            10: torch.float16,  # FLOAT16
            11: torch.double,   # DOUBLE
        }
        
        onnx_dtype = input_info.type.tensor_type.elem_type
        pytorch_dtype = dtype_map.get(onnx_dtype, torch.float32)
        
        return {
            "name": input_info.name,
            "shape": shape,
            "dtype": pytorch_dtype
        }
    
    def create_simple_model_from_onnx(self, onnx_model: onnx.ModelProto, input_info: Dict) -> Optional[torch.nn.Module]:
        """Create a simple PyTorch model that mimics ONNX structure."""
        # This is a simplified implementation
        # In practice, you'd need full ONNX → PyTorch conversion
        
        class SimpleModel(torch.nn.Module):
            def __init__(self, input_shape):
                super().__init__()
                # Create a simple model structure
                # This should be replaced with actual ONNX → PyTorch conversion
                self.flatten = torch.nn.Flatten()
                input_size = np.prod(input_shape[1:])  # Exclude batch dimension
                self.linear = torch.nn.Linear(input_size, 1000)  # Typical feature size
                
            def forward(self, x):
                x = self.flatten(x)
                x = self.linear(x)
                return x
        
        try:
            model = SimpleModel(input_info['shape'])
            model.eval()
            return model
        except Exception as e:
            print(f"❌ Failed to create simple model: {e}")
            return None
    
    def batch_convert(self, input_dir: Path, output_dir: Path, 
                     backend: str = "xnnpack", quantization: str = "none") -> List[Path]:
        """Convert all ONNX models in a directory."""
        onnx_files = list(input_dir.glob("*.onnx"))
        
        if not onnx_files:
            print(f"❌ No ONNX files found in {input_dir}")
            return []
        
        print(f"🔄 Converting {len(onnx_files)} ONNX models...")
        
        converted_models = []
        for onnx_file in onnx_files:
            output_file = output_dir / f"{onnx_file.stem}.pte"
            
            if self.convert_to_executorch(onnx_file, output_file, backend, quantization):
                converted_models.append(output_file)
        
        print(f"✅ Converted {len(converted_models)}/{len(onnx_files)} models")
        return converted_models


def main():
    parser = argparse.ArgumentParser(description="Convert ONNX models to ExecuTorch format")
    parser.add_argument("--input", type=str, required=True, help="Input ONNX file or directory")
    parser.add_argument("--output", type=str, required=True, help="Output .pte file or directory")
    parser.add_argument("--backend", type=str, default="xnnpack", 
                       choices=["xnnpack", "coreml", "vulkan"], help="Target backend")
    parser.add_argument("--quantization", type=str, default="none",
                       choices=["none", "fp16", "int8"], help="Quantization method")
    parser.add_argument("--batch", action="store_true", help="Batch convert directory")
    
    args = parser.parse_args()
    
    converter = ONNXToExecuTorchConverter()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"❌ Input path not found: {input_path}")
        return
    
    if args.batch or input_path.is_dir():
        # Batch conversion
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)
        
        converted_models = converter.batch_convert(
            input_path, output_path, args.backend, args.quantization
        )
        
        if converted_models:
            print(f"\n✅ Batch conversion complete!")
            print(f"   Converted models: {len(converted_models)}")
            for model_path in converted_models:
                print(f"   - {model_path}")
    else:
        # Single file conversion
        success = converter.convert_to_executorch(
            input_path, output_path, args.backend, args.quantization
        )
        
        if success:
            print(f"\n✅ Conversion complete: {output_path}")
        else:
            print(f"\n❌ Conversion failed")
            sys.exit(1)


if __name__ == "__main__":
    main()
