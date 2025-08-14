#!/usr/bin/env python3
"""
Mobile Inference Testing Tool for PE Core Models

This tool tests mobile-optimized PE Core models to validate performance
and accuracy before deployment to iOS/Android devices.

Usage:
    python test_mobile_inference.py --model pe_core_t16_384.pte
    python test_mobile_inference.py --benchmark --iterations 1000
    python test_mobile_inference.py --accuracy-test --reference-model PE-Core-T16-384

Features:
    - Test ExecuTorch .pte models
    - Performance benchmarking
    - Accuracy validation against reference models
    - Memory usage analysis
    - Mobile deployment readiness assessment
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np

# Check for ExecuTorch availability
try:
    from executorch.runtime import Runtime
    EXECUTORCH_AVAILABLE = True
except ImportError:
    EXECUTORCH_AVAILABLE = False
    print("❌ ExecuTorch runtime not available")
    print("   Install with: pip install executorch")

# Check for PE Core models
try:
    from perception_models.models.vision_transformer import VisionTransformer
    PE_CORE_AVAILABLE = True
except ImportError:
    PE_CORE_AVAILABLE = False
    print("❌ PE Core models not available")


class MobileInferenceTester:
    """Test mobile-optimized PE Core models."""
    
    def __init__(self):
        self.supported_models = {
            "pe_core_t16_384": {"input_size": 384, "reference": "PE-Core-T16-384"},
            "pe_core_s16_384": {"input_size": 384, "reference": "PE-Core-S16-384"},
            "pe_core_b16_224": {"input_size": 224, "reference": "PE-Core-B16-224"},
        }
    
    def load_executorch_model(self, model_path: Path) -> Optional[object]:
        """Load an ExecuTorch .pte model."""
        if not EXECUTORCH_AVAILABLE:
            print("❌ ExecuTorch runtime not available")
            return None
        
        try:
            print(f"🔄 Loading ExecuTorch model: {model_path}")
            runtime = Runtime.get()
            program = runtime.load_program(str(model_path))
            method = program.load_method("forward")
            print(f"✅ Loaded ExecuTorch model successfully")
            return method
        except Exception as e:
            print(f"❌ Failed to load ExecuTorch model: {e}")
            return None
    
    def load_reference_model(self, model_name: str) -> Optional[torch.nn.Module]:
        """Load reference PE Core model for comparison."""
        if not PE_CORE_AVAILABLE:
            print("❌ PE Core models not available")
            return None
        
        try:
            print(f"🔄 Loading reference model: {model_name}")
            model = VisionTransformer.from_config(model_name, pretrained=True)
            model.eval()
            print(f"✅ Loaded reference model successfully")
            return model
        except Exception as e:
            print(f"❌ Failed to load reference model: {e}")
            return None
    
    def create_test_inputs(self, input_size: int, batch_size: int = 1, 
                          num_samples: int = 10) -> List[torch.Tensor]:
        """Create test input tensors."""
        inputs = []
        for i in range(num_samples):
            # Create diverse test inputs
            if i == 0:
                # Random noise
                tensor = torch.randn(batch_size, 3, input_size, input_size)
            elif i == 1:
                # All zeros
                tensor = torch.zeros(batch_size, 3, input_size, input_size)
            elif i == 2:
                # All ones
                tensor = torch.ones(batch_size, 3, input_size, input_size)
            else:
                # Random with different distributions
                tensor = torch.randn(batch_size, 3, input_size, input_size) * 0.5 + 0.5
                tensor = torch.clamp(tensor, 0, 1)
            
            inputs.append(tensor)
        
        return inputs
    
    def benchmark_model(self, model_method: object, input_tensor: torch.Tensor, 
                       num_iterations: int = 100) -> Dict[str, float]:
        """Benchmark model inference performance."""
        print(f"🔄 Benchmarking model ({num_iterations} iterations)...")
        
        # Warmup runs
        for _ in range(10):
            try:
                model_method.execute([input_tensor])
            except Exception as e:
                print(f"❌ Warmup failed: {e}")
                return {}
        
        # Benchmark runs
        times = []
        for i in range(num_iterations):
            start_time = time.perf_counter()
            try:
                output = model_method.execute([input_tensor])
            except Exception as e:
                print(f"❌ Inference failed at iteration {i}: {e}")
                return {}
            end_time = time.perf_counter()
            
            inference_time = (end_time - start_time) * 1000  # Convert to ms
            times.append(inference_time)
            
            if (i + 1) % 50 == 0:
                print(f"   Completed {i + 1}/{num_iterations} iterations...")
        
        # Calculate statistics
        times = np.array(times)
        results = {
            "avg_inference_ms": float(np.mean(times)),
            "min_inference_ms": float(np.min(times)),
            "max_inference_ms": float(np.max(times)),
            "std_inference_ms": float(np.std(times)),
            "p50_inference_ms": float(np.percentile(times, 50)),
            "p95_inference_ms": float(np.percentile(times, 95)),
            "p99_inference_ms": float(np.percentile(times, 99)),
            "fps": 1000 / float(np.mean(times)),
        }
        
        return results
    
    def test_accuracy(self, mobile_method: object, reference_model: torch.nn.Module,
                     test_inputs: List[torch.Tensor], tolerance: float = 1e-3) -> Dict[str, float]:
        """Test accuracy of mobile model against reference."""
        print(f"🔄 Testing accuracy against reference model...")
        
        similarities = []
        max_diffs = []
        
        for i, input_tensor in enumerate(test_inputs):
            try:
                # Mobile model inference
                mobile_output = mobile_method.execute([input_tensor])
                mobile_features = mobile_output[0]
                
                # Reference model inference
                with torch.no_grad():
                    ref_features = reference_model(input_tensor)
                
                # Flatten outputs for comparison
                if len(mobile_features.shape) > 2:
                    mobile_features = mobile_features.view(mobile_features.shape[0], -1)
                if len(ref_features.shape) > 2:
                    ref_features = ref_features.view(ref_features.shape[0], -1)
                
                # Calculate similarity
                cosine_sim = F.cosine_similarity(mobile_features, ref_features, dim=1)
                similarities.append(float(cosine_sim.mean()))
                
                # Calculate max difference
                max_diff = float(torch.max(torch.abs(mobile_features - ref_features)))
                max_diffs.append(max_diff)
                
                print(f"   Sample {i+1}: Cosine similarity = {cosine_sim.mean():.4f}, Max diff = {max_diff:.6f}")
                
            except Exception as e:
                print(f"❌ Accuracy test failed for sample {i+1}: {e}")
                continue
        
        if not similarities:
            return {}
        
        results = {
            "avg_cosine_similarity": float(np.mean(similarities)),
            "min_cosine_similarity": float(np.min(similarities)),
            "avg_max_difference": float(np.mean(max_diffs)),
            "max_difference": float(np.max(max_diffs)),
            "accuracy_pass": float(np.mean(similarities)) > 0.99,  # 99% similarity threshold
        }
        
        return results
    
    def assess_mobile_readiness(self, benchmark_results: Dict[str, float]) -> Dict[str, bool]:
        """Assess if model is ready for mobile deployment."""
        if not benchmark_results:
            return {"mobile_ready": False}
        
        avg_time = benchmark_results.get("avg_inference_ms", float('inf'))
        fps = benchmark_results.get("fps", 0)
        
        assessments = {
            "real_time_capable": avg_time < 33.3,  # 30 FPS
            "interactive_capable": avg_time < 100,  # 10 FPS
            "production_ready": avg_time < 50 and fps > 20,  # 20+ FPS
            "mobile_optimized": avg_time < 20,  # Very fast
        }
        
        assessments["mobile_ready"] = assessments["real_time_capable"]
        
        return assessments
    
    def print_results(self, benchmark_results: Dict[str, float], 
                     accuracy_results: Dict[str, float],
                     readiness_results: Dict[str, bool]):
        """Print comprehensive test results."""
        print(f"\n{'='*60}")
        print("MOBILE INFERENCE TEST RESULTS")
        print(f"{'='*60}")
        
        if benchmark_results:
            print("\n📊 Performance Benchmarks:")
            print(f"   Average inference time: {benchmark_results['avg_inference_ms']:.2f}ms")
            print(f"   Min/Max inference time: {benchmark_results['min_inference_ms']:.2f}ms / {benchmark_results['max_inference_ms']:.2f}ms")
            print(f"   P95/P99 inference time: {benchmark_results['p95_inference_ms']:.2f}ms / {benchmark_results['p99_inference_ms']:.2f}ms")
            print(f"   Estimated FPS: {benchmark_results['fps']:.1f}")
            print(f"   Standard deviation: {benchmark_results['std_inference_ms']:.2f}ms")
        
        if accuracy_results:
            print("\n🎯 Accuracy Results:")
            print(f"   Average cosine similarity: {accuracy_results['avg_cosine_similarity']:.4f}")
            print(f"   Min cosine similarity: {accuracy_results['min_cosine_similarity']:.4f}")
            print(f"   Average max difference: {accuracy_results['avg_max_difference']:.6f}")
            print(f"   Max difference: {accuracy_results['max_difference']:.6f}")
            print(f"   Accuracy test: {'✅ PASS' if accuracy_results['accuracy_pass'] else '❌ FAIL'}")
        
        if readiness_results:
            print("\n📱 Mobile Deployment Readiness:")
            print(f"   Real-time capable (30+ FPS): {'✅ YES' if readiness_results['real_time_capable'] else '❌ NO'}")
            print(f"   Interactive capable (10+ FPS): {'✅ YES' if readiness_results['interactive_capable'] else '❌ NO'}")
            print(f"   Production ready (20+ FPS): {'✅ YES' if readiness_results['production_ready'] else '❌ NO'}")
            print(f"   Mobile optimized (<20ms): {'✅ YES' if readiness_results['mobile_optimized'] else '❌ NO'}")
            print(f"   Overall mobile ready: {'✅ YES' if readiness_results['mobile_ready'] else '❌ NO'}")
        
        print(f"\n{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Test mobile PE Core model inference")
    parser.add_argument("--model", type=str, required=True, help="Path to .pte model file")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--accuracy-test", action="store_true", help="Test accuracy against reference")
    parser.add_argument("--reference-model", type=str, help="Reference model name for accuracy test")
    parser.add_argument("--iterations", type=int, default=100, help="Number of benchmark iterations")
    parser.add_argument("--input-size", type=int, help="Input size (auto-detected if not specified)")
    
    args = parser.parse_args()
    
    tester = MobileInferenceTester()
    
    # Load mobile model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        return
    
    mobile_method = tester.load_executorch_model(model_path)
    if not mobile_method:
        return
    
    # Determine input size
    input_size = args.input_size
    if not input_size:
        # Try to auto-detect from filename
        for model_key, config in tester.supported_models.items():
            if model_key in model_path.name:
                input_size = config["input_size"]
                break
        
        if not input_size:
            input_size = 384  # Default
            print(f"⚠️  Using default input size: {input_size}")
    
    print(f"📋 Using input size: {input_size}x{input_size}")
    
    # Create test inputs
    test_inputs = tester.create_test_inputs(input_size)
    
    # Run benchmark
    benchmark_results = {}
    if args.benchmark:
        benchmark_results = tester.benchmark_model(mobile_method, test_inputs[0], args.iterations)
    
    # Run accuracy test
    accuracy_results = {}
    if args.accuracy_test:
        if not args.reference_model:
            print("❌ Reference model required for accuracy test (use --reference-model)")
            return
        
        reference_model = tester.load_reference_model(args.reference_model)
        if reference_model:
            accuracy_results = tester.test_accuracy(mobile_method, reference_model, test_inputs)
    
    # Assess mobile readiness
    readiness_results = tester.assess_mobile_readiness(benchmark_results)
    
    # Print results
    tester.print_results(benchmark_results, accuracy_results, readiness_results)


if __name__ == "__main__":
    main()
