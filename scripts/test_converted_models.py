#!/usr/bin/env python3
"""
Test converted PE-Core-S16-384 model files to ensure they work correctly.
"""

import argparse
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch


class ModelTester:
    """Test converted model files."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.input_shape = (1, 3, 384, 384)
        self.expected_output_shape = (1, 512)
        
    def create_test_input(self) -> np.ndarray:
        """Create test input data."""
        # Create normalized test input
        test_input = np.random.randn(*self.input_shape).astype(np.float32)
        return test_input
    
    def test_coreml_models(self, models_dir: Path) -> Dict[str, bool]:
        """Test CoreML model files."""
        print("🧪 Testing CoreML models...")
        results = {}
        
        try:
            import coremltools as ct
            
            # Find CoreML models
            coreml_files = list(models_dir.glob("*.mlpackage"))
            
            for model_path in coreml_files:
                model_name = model_path.name
                print(f"  Testing {model_name}...")
                
                try:
                    # Load model
                    model = ct.models.MLModel(str(model_path))
                    
                    # Get model info
                    spec = model.get_spec()
                    print(f"    Model loaded successfully")
                    
                    # Create test input
                    test_input = self.create_test_input()
                    input_dict = {"image": test_input}
                    
                    # Run prediction
                    output = model.predict(input_dict)
                    
                    # Validate output
                    if "embeddings" in output:
                        embeddings = output["embeddings"]
                        print(f"    Output shape: {embeddings.shape}")
                        print(f"    Output range: [{embeddings.min():.4f}, {embeddings.max():.4f}]")
                        
                        # Check output shape
                        if embeddings.shape == self.expected_output_shape:
                            print(f"    ✅ Shape validation passed")
                        else:
                            print(f"    ❌ Shape mismatch: expected {self.expected_output_shape}, got {embeddings.shape}")
                            results[model_name] = False
                            continue
                        
                        # Check for NaN/Inf
                        if np.isfinite(embeddings).all():
                            print(f"    ✅ Output validation passed")
                        else:
                            print(f"    ❌ Invalid values in output")
                            results[model_name] = False
                            continue
                        
                        results[model_name] = True
                        print(f"    ✅ {model_name} test passed")
                    else:
                        print(f"    ❌ Expected 'embeddings' output not found")
                        results[model_name] = False
                
                except Exception as e:
                    print(f"    ❌ Error testing {model_name}: {e}")
                    results[model_name] = False
        
        except ImportError:
            print("  ⚠️  CoreML Tools not available, skipping CoreML tests")
        
        return results
    
    def test_onnx_models(self, models_dir: Path) -> Dict[str, bool]:
        """Test ONNX model files."""
        print("🧪 Testing ONNX models...")
        results = {}
        
        try:
            import onnxruntime as ort
            
            # Find ONNX models
            onnx_files = list(models_dir.glob("*.onnx"))
            
            for model_path in onnx_files:
                model_name = model_path.name
                print(f"  Testing {model_name}...")
                
                try:
                    # Create session
                    session = ort.InferenceSession(str(model_path))
                    
                    print(f"    Model loaded successfully")
                    
                    # Get input/output info
                    input_info = session.get_inputs()[0]
                    output_info = session.get_outputs()[0]
                    
                    print(f"    Input: {input_info.name} {input_info.shape}")
                    print(f"    Output: {output_info.name} {output_info.shape}")
                    
                    # Create test input
                    test_input = self.create_test_input()
                    
                    # Run inference
                    outputs = session.run([output_info.name], {input_info.name: test_input})
                    embeddings = outputs[0]
                    
                    print(f"    Output shape: {embeddings.shape}")
                    print(f"    Output range: [{embeddings.min():.4f}, {embeddings.max():.4f}]")
                    
                    # Validate output
                    if embeddings.shape == self.expected_output_shape:
                        print(f"    ✅ Shape validation passed")
                    else:
                        print(f"    ❌ Shape mismatch: expected {self.expected_output_shape}, got {embeddings.shape}")
                        results[model_name] = False
                        continue
                    
                    # Check for NaN/Inf
                    if np.isfinite(embeddings).all():
                        print(f"    ✅ Output validation passed")
                    else:
                        print(f"    ❌ Invalid values in output")
                        results[model_name] = False
                        continue
                    
                    results[model_name] = True
                    print(f"    ✅ {model_name} test passed")
                
                except Exception as e:
                    print(f"    ❌ Error testing {model_name}: {e}")
                    results[model_name] = False
        
        except ImportError:
            print("  ⚠️  ONNX Runtime not available, skipping ONNX tests")
        
        return results
    
    def test_executorch_models(self, models_dir: Path) -> Dict[str, bool]:
        """Test ExecuTorch model files."""
        print("🧪 Testing ExecuTorch models...")
        results = {}
        
        try:
            from executorch.extension.pybindings.portable_lib import _load_for_executorch
            
            # Find ExecuTorch models
            et_files = list(models_dir.glob("*.pte"))
            
            for model_path in et_files:
                model_name = model_path.name
                print(f"  Testing {model_name}...")
                
                try:
                    # Load model
                    model = _load_for_executorch(str(model_path))
                    
                    print(f"    Model loaded successfully")
                    
                    # Create test input
                    test_input_np = self.create_test_input()
                    test_input_tensor = torch.from_numpy(test_input_np)
                    
                    # Run inference
                    outputs = model.forward([test_input_tensor])
                    embeddings = outputs[0].numpy()
                    
                    print(f"    Output shape: {embeddings.shape}")
                    print(f"    Output range: [{embeddings.min():.4f}, {embeddings.max():.4f}]")
                    
                    # Validate output
                    if embeddings.shape == self.expected_output_shape:
                        print(f"    ✅ Shape validation passed")
                    else:
                        print(f"    ❌ Shape mismatch: expected {self.expected_output_shape}, got {embeddings.shape}")
                        results[model_name] = False
                        continue
                    
                    # Check for NaN/Inf
                    if np.isfinite(embeddings).all():
                        print(f"    ✅ Output validation passed")
                    else:
                        print(f"    ❌ Invalid values in output")
                        results[model_name] = False
                        continue
                    
                    results[model_name] = True
                    print(f"    ✅ {model_name} test passed")
                
                except Exception as e:
                    print(f"    ❌ Error testing {model_name}: {e}")
                    results[model_name] = False
        
        except ImportError:
            print("  ⚠️  ExecuTorch not available, skipping ExecuTorch tests")
        
        return results
    
    def test_all_models(self, models_dir: Path) -> bool:
        """Test all model formats."""
        print(f"🔍 Testing converted models in {models_dir}")
        print("=" * 60)
        
        all_results = {}
        
        # Test each format
        all_results.update(self.test_coreml_models(models_dir))
        all_results.update(self.test_onnx_models(models_dir))
        all_results.update(self.test_executorch_models(models_dir))
        
        print("=" * 60)
        print("📊 Test Results Summary:")
        
        if not all_results:
            print("  ⚠️  No models found to test")
            return False
        
        passed = 0
        failed = 0
        
        for model_name, result in all_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {status}: {model_name}")
            
            if result:
                passed += 1
            else:
                failed += 1
        
        print(f"\nTotal: {len(all_results)} models")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        
        success = failed == 0
        if success:
            print("\n🎉 All model tests passed!")
        else:
            print(f"\n❌ {failed} model test(s) failed")
        
        return success


def main():
    parser = argparse.ArgumentParser(description="Test converted model files")
    parser.add_argument("--models-dir", required=True, type=Path,
                       help="Directory containing converted models")
    parser.add_argument("--model-name", default="PE-Core-S16-384",
                       help="Model name being tested")
    
    args = parser.parse_args()
    
    if not args.models_dir.exists():
        print(f"❌ Models directory not found: {args.models_dir}")
        sys.exit(1)
    
    tester = ModelTester(args.model_name)
    
    try:
        success = tester.test_all_models(args.models_dir)
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"💥 Testing failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()