#!/usr/bin/env python3
"""
Benchmark PE-Core-S16-384 converted models for performance analysis.
"""

import argparse
import json
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch


class ModelBenchmark:
    """Benchmark converted model files."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.input_shape = (1, 3, 384, 384)
        self.warmup_runs = 5
        self.benchmark_runs = 20
        
    def create_test_input(self) -> np.ndarray:
        """Create test input data."""
        return np.random.randn(*self.input_shape).astype(np.float32)
    
    def benchmark_coreml(self, models_dir: Path) -> Dict[str, Any]:
        """Benchmark CoreML models."""
        print("⚡ Benchmarking CoreML models...")
        results = {}
        
        try:
            import coremltools as ct
            
            coreml_files = list(models_dir.glob("*.mlpackage"))
            
            for model_path in coreml_files:
                model_name = model_path.name
                print(f"  Benchmarking {model_name}...")
                
                try:
                    # Load model
                    model = ct.models.MLModel(str(model_path))
                    test_input = {"image": self.create_test_input()}
                    
                    # Warmup
                    for _ in range(self.warmup_runs):
                        _ = model.predict(test_input)
                    
                    # Benchmark
                    times = []
                    for _ in range(self.benchmark_runs):
                        start_time = time.perf_counter()
                        _ = model.predict(test_input)
                        end_time = time.perf_counter()
                        times.append((end_time - start_time) * 1000)  # Convert to ms
                    
                    # Calculate statistics
                    times = np.array(times)
                    result = {
                        "mean_ms": float(np.mean(times)),
                        "std_ms": float(np.std(times)),
                        "min_ms": float(np.min(times)),
                        "max_ms": float(np.max(times)),
                        "p50_ms": float(np.percentile(times, 50)),
                        "p95_ms": float(np.percentile(times, 95)),
                        "p99_ms": float(np.percentile(times, 99)),
                        "iterations": len(times)
                    }
                    
                    results[model_name] = result
                    print(f"    Mean: {result['mean_ms']:.2f}ms ± {result['std_ms']:.2f}ms")
                    print(f"    P95: {result['p95_ms']:.2f}ms")
                    
                except Exception as e:
                    print(f"    ❌ Error benchmarking {model_name}: {e}")
                    results[model_name] = {"error": str(e)}
        
        except ImportError:
            print("  ⚠️  CoreML Tools not available")
        
        return results
    
    def benchmark_onnx(self, models_dir: Path) -> Dict[str, Any]:
        """Benchmark ONNX models."""
        print("⚡ Benchmarking ONNX models...")
        results = {}
        
        try:
            import onnxruntime as ort
            
            # Test different providers
            providers = []
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers.append("CUDAExecutionProvider")
            if "CPUExecutionProvider" in ort.get_available_providers():
                providers.append("CPUExecutionProvider")
            
            onnx_files = list(models_dir.glob("*.onnx"))
            
            for model_path in onnx_files:
                model_name = model_path.name
                
                for provider in providers:
                    variant_name = f"{model_name}_{provider.lower().replace('executionprovider', '')}"
                    print(f"  Benchmarking {variant_name}...")
                    
                    try:
                        # Create session with specific provider
                        session = ort.InferenceSession(str(model_path), providers=[provider])
                        
                        input_info = session.get_inputs()[0]
                        output_info = session.get_outputs()[0]
                        test_input = {input_info.name: self.create_test_input()}
                        
                        # Warmup
                        for _ in range(self.warmup_runs):
                            _ = session.run([output_info.name], test_input)
                        
                        # Benchmark
                        times = []
                        for _ in range(self.benchmark_runs):
                            start_time = time.perf_counter()
                            _ = session.run([output_info.name], test_input)
                            end_time = time.perf_counter()
                            times.append((end_time - start_time) * 1000)
                        
                        # Calculate statistics
                        times = np.array(times)
                        result = {
                            "mean_ms": float(np.mean(times)),
                            "std_ms": float(np.std(times)),
                            "min_ms": float(np.min(times)),
                            "max_ms": float(np.max(times)),
                            "p50_ms": float(np.percentile(times, 50)),
                            "p95_ms": float(np.percentile(times, 95)),
                            "p99_ms": float(np.percentile(times, 99)),
                            "provider": provider,
                            "iterations": len(times)
                        }
                        
                        results[variant_name] = result
                        print(f"    Mean: {result['mean_ms']:.2f}ms ± {result['std_ms']:.2f}ms")
                        print(f"    P95: {result['p95_ms']:.2f}ms")
                        
                    except Exception as e:
                        print(f"    ❌ Error with {provider}: {e}")
                        results[variant_name] = {"error": str(e), "provider": provider}
        
        except ImportError:
            print("  ⚠️  ONNX Runtime not available")
        
        return results
    
    def benchmark_executorch(self, models_dir: Path) -> Dict[str, Any]:
        """Benchmark ExecuTorch models."""
        print("⚡ Benchmarking ExecuTorch models...")
        results = {}
        
        try:
            from executorch.extension.pybindings.portable_lib import _load_for_executorch
            
            et_files = list(models_dir.glob("*.pte"))
            
            for model_path in et_files:
                model_name = model_path.name
                print(f"  Benchmarking {model_name}...")
                
                try:
                    # Load model
                    model = _load_for_executorch(str(model_path))
                    
                    test_input_np = self.create_test_input()
                    test_input = [torch.from_numpy(test_input_np)]
                    
                    # Warmup
                    for _ in range(self.warmup_runs):
                        _ = model.forward(test_input)
                    
                    # Benchmark
                    times = []
                    for _ in range(self.benchmark_runs):
                        start_time = time.perf_counter()
                        _ = model.forward(test_input)
                        end_time = time.perf_counter()
                        times.append((end_time - start_time) * 1000)
                    
                    # Calculate statistics
                    times = np.array(times)
                    result = {
                        "mean_ms": float(np.mean(times)),
                        "std_ms": float(np.std(times)),
                        "min_ms": float(np.min(times)),
                        "max_ms": float(np.max(times)),
                        "p50_ms": float(np.percentile(times, 50)),
                        "p95_ms": float(np.percentile(times, 95)),
                        "p99_ms": float(np.percentile(times, 99)),
                        "iterations": len(times)
                    }
                    
                    results[model_name] = result
                    print(f"    Mean: {result['mean_ms']:.2f}ms ± {result['std_ms']:.2f}ms")
                    print(f"    P95: {result['p95_ms']:.2f}ms")
                    
                except Exception as e:
                    print(f"    ❌ Error benchmarking {model_name}: {e}")
                    results[model_name] = {"error": str(e)}
        
        except ImportError:
            print("  ⚠️  ExecuTorch not available")
        
        return results
    
    def run_benchmark(self, models_dir: Path, iterations: int) -> Dict[str, Any]:
        """Run complete benchmark suite."""
        self.benchmark_runs = iterations
        
        print(f"🚀 Benchmarking {self.model_name} models")
        print(f"📁 Models directory: {models_dir}")
        print(f"🔢 Iterations: warmup={self.warmup_runs}, benchmark={self.benchmark_runs}")
        print("=" * 60)
        
        results = {
            "benchmark": {
                "model_name": self.model_name,
                "input_shape": self.input_shape,
                "warmup_runs": self.warmup_runs,
                "benchmark_runs": self.benchmark_runs,
                "timestamp": time.time()
            },
            "system": {
                "platform": platform.platform(),
                "processor": platform.processor(),
                "python_version": platform.python_version(),
                "pytorch_version": torch.__version__
            },
            "results": {}
        }
        
        # Run benchmarks
        coreml_results = self.benchmark_coreml(models_dir)
        onnx_results = self.benchmark_onnx(models_dir)
        executorch_results = self.benchmark_executorch(models_dir)
        
        results["results"]["coreml"] = coreml_results
        results["results"]["onnx"] = onnx_results
        results["results"]["executorch"] = executorch_results
        
        # Summary
        print("=" * 60)
        print("📊 Benchmark Summary:")
        
        all_valid_results = []
        for format_name, format_results in results["results"].items():
            for model_name, model_result in format_results.items():
                if "error" not in model_result and "mean_ms" in model_result:
                    all_valid_results.append((f"{format_name}/{model_name}", model_result["mean_ms"]))
        
        if all_valid_results:
            # Sort by performance
            all_valid_results.sort(key=lambda x: x[1])
            
            print("\nPerformance ranking (fastest to slowest):")
            for i, (name, mean_ms) in enumerate(all_valid_results, 1):
                print(f"  {i:2d}. {name:40s} {mean_ms:6.2f}ms")
            
            # Best performer
            best_name, best_time = all_valid_results[0]
            print(f"\n🏆 Best performer: {best_name} ({best_time:.2f}ms)")
            
            # Real-time capability analysis
            fps_30_threshold = 33.33  # ms for 30 FPS
            fps_60_threshold = 16.67  # ms for 60 FPS
            
            realtime_30fps = [name for name, ms in all_valid_results if ms < fps_30_threshold]
            realtime_60fps = [name for name, ms in all_valid_results if ms < fps_60_threshold]
            
            print(f"\n📱 Real-time capability:")
            print(f"   30 FPS capable: {len(realtime_30fps)}/{len(all_valid_results)} models")
            print(f"   60 FPS capable: {len(realtime_60fps)}/{len(all_valid_results)} models")
            
            if realtime_30fps:
                print(f"   Recommended for 30 FPS: {realtime_30fps[0]}")
            if realtime_60fps:
                print(f"   Recommended for 60 FPS: {realtime_60fps[0]}")
        
        else:
            print("  ⚠️  No valid benchmark results")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark converted models")
    parser.add_argument("--models-dir", required=True, type=Path,
                       help="Directory containing converted models")
    parser.add_argument("--model-name", default="PE-Core-S16-384",
                       help="Model name being benchmarked")
    parser.add_argument("--iterations", type=int, default=20,
                       help="Number of benchmark iterations")
    parser.add_argument("--report-file", type=Path,
                       help="Output file for benchmark report")
    
    args = parser.parse_args()
    
    if not args.models_dir.exists():
        print(f"❌ Models directory not found: {args.models_dir}")
        sys.exit(1)
    
    benchmark = ModelBenchmark(args.model_name)
    
    try:
        results = benchmark.run_benchmark(args.models_dir, args.iterations)
        
        if args.report_file:
            args.report_file.parent.mkdir(parents=True, exist_ok=True)
            with open(args.report_file, 'w') as f:
                json.dump(results, f, indent=2, sort_keys=True)
            print(f"\n📄 Benchmark report saved: {args.report_file}")
        
        # Check if we have any successful results
        has_results = any(
            "error" not in model_result and "mean_ms" in model_result
            for format_results in results["results"].values()
            for model_result in format_results.values()
        )
        
        sys.exit(0 if has_results else 1)
        
    except Exception as e:
        print(f"💥 Benchmark failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()