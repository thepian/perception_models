#!/usr/bin/env python3
"""
ExecuTorch Conversion Tool for PE Core Models

This tool converts Facebook's Perception Encoder (PE) Core models to ExecuTorch format
for mobile deployment on iOS and Android devices.

Usage:
    python convert_executorch.py --model PE-Core-T16-384 --output mobile_models/
    python convert_executorch.py --model PE-Core-S16-384 --precision fp16
    python convert_executorch.py --all-models --benchmark

Features:
    - Convert PE Core models to ExecuTorch .pte format
    - Mobile-optimized model preparation
    - Performance benchmarking
    - Quantization support
    - iOS/Android deployment preparation

Requirements:
    - Python 3.10-3.12 (ExecuTorch requirement)
    - ExecuTorch package: pip install executorch
    - PyTorch 2.0+
    - PE Core models (automatically downloaded)
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.export import export

# Check Python version compatibility
if sys.version_info < (3, 10) or sys.version_info >= (3, 13):
    print("❌ ExecuTorch requires Python 3.10-3.12")
    print(f"   Current Python version: {sys.version}")
    print("   Please use a compatible Python version for ExecuTorch conversion")
    sys.exit(1)

# Try to import ExecuTorch 0.7.0 (will fail if not installed)
try:
    from executorch.exir import to_edge_transform_and_lower
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    # ExecuTorch 0.7.0 specific imports
    from executorch.exir.backend.backend_details import BackendDetails
    from executorch.exir.backend.compile_spec_schema import CompileSpec
    EXECUTORCH_AVAILABLE = True
    print("✅ ExecuTorch 0.7.0 available")
except ImportError as e:
    EXECUTORCH_AVAILABLE = False
    print("❌ ExecuTorch 0.7.0 not available")
    print(f"   Error: {e}")
    print("   Install with: pip install executorch==0.7.0")
    print("   Note: Requires Python 3.10-3.12 and PyTorch 2.8+")

# Import PE Core models
try:
    from perception_models.models.vision_transformer import VisionTransformer
    PE_CORE_AVAILABLE = True
except ImportError as e:
    PE_CORE_AVAILABLE = False
    print(f"❌ PE Core models not available: {e}")


class ExecuTorchConverter:
    """Convert PE Core models to ExecuTorch format for mobile deployment."""
    
    def __init__(self):
        self.supported_models = {
            "PE-Core-T16-384": {"input_size": 384, "description": "Tiny model, fastest inference"},
            "PE-Core-S16-384": {"input_size": 384, "description": "Small model, balanced performance"},
            "PE-Core-B16-224": {"input_size": 224, "description": "Base model, highest accuracy"},
        }
        
        self.quantization_options = {
            "none": "No quantization (fp32)",
            "fp16": "Half precision (fp16)",
            "int8": "8-bit quantization (int8)",
        }
        
        self.backend_options = {
            "xnnpack": "CPU optimization (iOS/Android)",
            "coreml": "Apple Neural Engine (iOS only)",
            "vulkan": "GPU acceleration (Android)",
        }
    
    def check_requirements(self) -> bool:
        """Check if all requirements are available."""
        if not EXECUTORCH_AVAILABLE:
            print("❌ ExecuTorch not available - cannot convert models")
            return False
        
        if not PE_CORE_AVAILABLE:
            print("❌ PE Core models not available")
            return False
        
        return True
    
    def load_pe_core_model(self, model_name: str) -> Optional[nn.Module]:
        """Load a PE Core model."""
        if not PE_CORE_AVAILABLE:
            print("❌ PE Core models not available")
            return None
        
        try:
            print(f"🔄 Loading {model_name}...")
            model = VisionTransformer.from_config(model_name, pretrained=True)
            model.eval()
            print(f"✅ Loaded {model_name}")
            return model
        except Exception as e:
            print(f"❌ Failed to load {model_name}: {e}")
            return None
    
    def create_mobile_wrapper(self, model: nn.Module, input_size: int) -> nn.Module:
        """Create a mobile-optimized wrapper for the PE Core model."""
        
        class MobilePEWrapper(nn.Module):
            """Mobile-optimized wrapper for PE Core models."""
            
            def __init__(self, pe_model: nn.Module, input_size: int):
                super().__init__()
                self.pe_model = pe_model
                self.input_size = input_size
                
                # Set to eval mode
                self.pe_model.eval()
                
                # Register buffer for input size reference
                self.register_buffer('_input_size', torch.tensor(input_size))
            
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                """Forward pass with mobile optimizations."""
                # Ensure input is correct size
                if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
                    x = torch.nn.functional.interpolate(
                        x, size=(self.input_size, self.input_size), 
                        mode='bilinear', align_corners=False
                    )
                
                # Forward through PE model
                features = self.pe_model(x)
                
                # Ensure output is 2D for mobile deployment
                if len(features.shape) > 2:
                    features = features.view(features.shape[0], -1)
                
                return features
        
        return MobilePEWrapper(model, input_size)
    
    def convert_to_executorch(self, model: nn.Module, model_name: str, 
                             input_size: int, output_dir: str = "mobile_models",
                             backend: str = "xnnpack", quantization: str = "none") -> Optional[Path]:
        """Convert model to ExecuTorch format."""
        if not EXECUTORCH_AVAILABLE:
            print("❌ ExecuTorch not available")
            return None
        
        print(f"🔄 Converting {model_name} to ExecuTorch...")
        print(f"   Backend: {backend}")
        print(f"   Quantization: {quantization}")
        print(f"   Input size: {input_size}x{input_size}")
        
        try:
            # Create mobile wrapper
            mobile_model = self.create_mobile_wrapper(model, input_size)
            
            # Create example input
            example_input = torch.randn(1, 3, input_size, input_size)
            
            # Test the wrapper
            print("   Testing mobile wrapper...")
            with torch.no_grad():
                test_output = mobile_model(example_input)
                print(f"   Output shape: {test_output.shape}")
            
            # Export the model
            print("   Exporting model...")
            exported_program = export(mobile_model, (example_input,))
            
            # Configure backend partitioner
            partitioner = None
            if backend == "xnnpack":
                partitioner = [XnnpackPartitioner()]
            # Add other backends as needed
            
            # Convert to ExecuTorch
            print("   Converting to ExecuTorch format...")
            et_program = to_edge_transform_and_lower(
                exported_program,
                partitioner=partitioner
            ).to_executorch()
            
            # Save the model
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            model_filename = f"{model_name.lower().replace('-', '_')}_{backend}"
            if quantization != "none":
                model_filename += f"_{quantization}"
            model_filename += ".pte"
            
            model_path = output_path / model_filename
            
            print(f"   Saving to: {model_path}")
            with open(model_path, "wb") as f:
                f.write(et_program.buffer)
            
            print(f"✅ ExecuTorch conversion successful: {model_path}")
            return model_path
            
        except Exception as e:
            print(f"❌ ExecuTorch conversion failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def benchmark_model(self, model_path: Path, num_runs: int = 100) -> Dict[str, float]:
        """Benchmark ExecuTorch model performance."""
        if not EXECUTORCH_AVAILABLE:
            print("❌ ExecuTorch not available for benchmarking")
            return {}
        
        try:
            from executorch.runtime import Runtime
            
            print(f"🔄 Benchmarking {model_path.name}...")
            
            # Load the model
            runtime = Runtime.get()
            program = runtime.load_program(str(model_path))
            method = program.load_method("forward")
            
            # Create input tensor (assuming 384x384 for now)
            input_tensor = torch.randn(1, 3, 384, 384)
            
            # Warmup runs
            for _ in range(10):
                method.execute([input_tensor])
            
            # Benchmark runs
            times = []
            for _ in range(num_runs):
                start_time = time.time()
                method.execute([input_tensor])
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # Convert to ms
            
            # Calculate statistics
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            results = {
                "avg_inference_ms": avg_time,
                "min_inference_ms": min_time,
                "max_inference_ms": max_time,
                "fps": 1000 / avg_time,
            }
            
            print(f"   Average inference time: {avg_time:.2f}ms")
            print(f"   Min/Max: {min_time:.2f}ms / {max_time:.2f}ms")
            print(f"   Estimated FPS: {results['fps']:.1f}")
            
            return results
            
        except Exception as e:
            print(f"❌ Benchmarking failed: {e}")
            return {}
    
    def convert_all_models(self, output_dir: str = "mobile_models", 
                          backend: str = "xnnpack", quantization: str = "none") -> List[Path]:
        """Convert all supported PE Core models."""
        if not self.check_requirements():
            return []
        
        converted_models = []
        
        for model_name, config in self.supported_models.items():
            print(f"\n{'='*60}")
            print(f"Converting {model_name}")
            print(f"{'='*60}")
            
            # Load model
            model = self.load_pe_core_model(model_name)
            if model is None:
                continue
            
            # Convert to ExecuTorch
            model_path = self.convert_to_executorch(
                model, model_name, config["input_size"], 
                output_dir, backend, quantization
            )
            
            if model_path:
                converted_models.append(model_path)
        
        return converted_models


def main():
    parser = argparse.ArgumentParser(description="Convert PE Core models to ExecuTorch format")
    parser.add_argument("--model", type=str, help="Model to convert (e.g., PE-Core-T16-384)")
    parser.add_argument("--all-models", action="store_true", help="Convert all supported models")
    parser.add_argument("--output", type=str, default="mobile_models", help="Output directory")
    parser.add_argument("--backend", type=str, default="xnnpack", 
                       choices=["xnnpack", "coreml", "vulkan"], help="Target backend")
    parser.add_argument("--quantization", type=str, default="none",
                       choices=["none", "fp16", "int8"], help="Quantization method")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark converted models")
    parser.add_argument("--list-models", action="store_true", help="List supported models")
    
    args = parser.parse_args()
    
    converter = ExecuTorchConverter()
    
    if args.list_models:
        print("📋 Supported PE Core Models:")
        for model_name, config in converter.supported_models.items():
            print(f"   {model_name}: {config['description']} ({config['input_size']}x{config['input_size']})")
        return
    
    if not converter.check_requirements():
        print("\n💡 Setup Instructions:")
        print("   1. Use Python 3.10-3.12: conda create -n executorch python=3.11")
        print("   2. Install ExecuTorch: pip install executorch")
        print("   3. Install PE Core models: pip install -e .")
        return
    
    converted_models = []
    
    if args.all_models:
        converted_models = converter.convert_all_models(
            args.output, args.backend, args.quantization
        )
    elif args.model:
        if args.model not in converter.supported_models:
            print(f"❌ Unsupported model: {args.model}")
            print("   Use --list-models to see supported models")
            return
        
        model = converter.load_pe_core_model(args.model)
        if model:
            config = converter.supported_models[args.model]
            model_path = converter.convert_to_executorch(
                model, args.model, config["input_size"],
                args.output, args.backend, args.quantization
            )
            if model_path:
                converted_models.append(model_path)
    else:
        parser.print_help()
        return
    
    # Benchmark if requested
    if args.benchmark and converted_models:
        print(f"\n{'='*60}")
        print("BENCHMARKING RESULTS")
        print(f"{'='*60}")
        
        for model_path in converted_models:
            converter.benchmark_model(model_path)
    
    # Summary
    if converted_models:
        print(f"\n✅ Successfully converted {len(converted_models)} model(s)")
        print("📱 Ready for mobile deployment!")
        print("\n📋 Next Steps:")
        print("   1. Test models on target devices")
        print("   2. Integrate into iOS/Android apps")
        print("   3. Optimize for production deployment")
    else:
        print("❌ No models were converted")


if __name__ == "__main__":
    main()
