#!/usr/bin/env python3
"""
PE Spatial Mobile Performance Test

Tests Facebook's Perception Encoder (PE) Spatial models for mobile deployment:
- Benchmarks different PE Spatial model sizes (T16, S16, B16)
- Measures inference speed and memory usage
- Evaluates mobile deployment feasibility
- Compares with RF-DETR baseline performance
"""

import os
import sys
import time
import torch
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from core.vision_encoder.pe import VisionTransformer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


class PESpatialMobileBenchmark:
    """Benchmarks PE Spatial models for mobile deployment."""
    
    def __init__(self):
        self.results = {}
        self.available_models = [
            "PE-Spatial-T16-512",  # Tiny - most mobile-friendly
            "PE-Spatial-S16-512",  # Small - good mobile candidate
            "PE-Spatial-B16-512",  # Base - may be too large for mobile
        ]
        
    def load_pe_spatial_model(self, model_name: str) -> torch.nn.Module:
        """Load PE Spatial model."""
        print(f"📦 Loading {model_name}...")
        
        try:
            # Load PE Spatial model
            model = VisionTransformer.from_config(model_name, pretrained=True)
            model.eval()
            
            # Move to CPU for mobile simulation
            model = model.cpu()
            
            # Disable gradients
            for param in model.parameters():
                param.requires_grad = False
            
            print(f"✅ {model_name} loaded successfully")
            return model
            
        except Exception as e:
            print(f"❌ Failed to load {model_name}: {e}")
            return None
    
    def get_model_info(self, model: torch.nn.Module, model_name: str) -> Dict[str, Any]:
        """Get model information including size and parameters."""
        try:
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Estimate model size (rough approximation)
            param_size_mb = total_params * 4 / (1024 * 1024)  # Assuming FP32
            
            # Get model configuration
            config = getattr(model, 'config', {})
            
            info = {
                'model_name': model_name,
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'estimated_size_mb': param_size_mb,
                'config': config
            }
            
            print(f"   📊 Model Info:")
            print(f"      Total parameters: {total_params:,}")
            print(f"      Estimated size: {param_size_mb:.1f}MB")
            
            return info
            
        except Exception as e:
            print(f"   ❌ Failed to get model info: {e}")
            return {'error': str(e)}
    
    def benchmark_inference(self, model: torch.nn.Module, model_name: str, 
                          img_size: int = 512, num_runs: int = 100) -> Dict[str, Any]:
        """Benchmark model inference performance."""
        print(f"\n🔬 Benchmarking {model_name} inference...")
        
        try:
            # Create test input
            test_input = torch.randn(1, 3, img_size, img_size).cpu()
            
            print(f"   Input shape: {test_input.shape}")
            print(f"   Device: CPU (mobile simulation)")
            print(f"   Runs: {num_runs}")
            
            # Warmup
            print(f"   Warming up (10 runs)...")
            with torch.no_grad():
                for _ in range(10):
                    _ = model(test_input)
            
            # Benchmark
            print(f"   Running {num_runs} benchmark iterations...")
            times = []
            
            with torch.no_grad():
                for i in range(num_runs):
                    start_time = time.perf_counter()
                    output = model(test_input)
                    end_time = time.perf_counter()
                    
                    times.append(end_time - start_time)
                    
                    if (i + 1) % 20 == 0:
                        print(f"     Completed {i + 1}/{num_runs} runs")
            
            # Calculate statistics
            times = np.array(times)
            
            stats = {
                'model_name': model_name,
                'input_shape': list(test_input.shape),
                'num_runs': len(times),
                'mean_time_ms': float(np.mean(times) * 1000),
                'median_time_ms': float(np.median(times) * 1000),
                'min_time_ms': float(np.min(times) * 1000),
                'max_time_ms': float(np.max(times) * 1000),
                'p95_time_ms': float(np.percentile(times, 95) * 1000),
                'p99_time_ms': float(np.percentile(times, 99) * 1000),
                'fps': float(1.0 / np.mean(times)),
                'std_time_ms': float(np.std(times) * 1000),
                'output_shape': list(output.shape) if hasattr(output, 'shape') else 'complex'
            }
            
            # Print results
            print(f"   ✅ Results:")
            print(f"      Mean time: {stats['mean_time_ms']:.2f}ms")
            print(f"      Median time: {stats['median_time_ms']:.2f}ms")
            print(f"      Min time: {stats['min_time_ms']:.2f}ms")
            print(f"      P95 time: {stats['p95_time_ms']:.2f}ms")
            print(f"      FPS: {stats['fps']:.1f}")
            print(f"      Std dev: {stats['std_time_ms']:.2f}ms")
            print(f"      Output shape: {stats['output_shape']}")
            
            return stats
            
        except Exception as e:
            print(f"   ❌ Benchmark failed: {e}")
            return {'error': str(e)}
    
    def evaluate_mobile_readiness(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate mobile deployment readiness."""
        if 'error' in stats:
            return {'mobile_readiness': 'Error', 'assessment': stats['error']}
        
        mean_time = stats['mean_time_ms']
        
        if mean_time < 10:
            readiness = "🎉 Excellent"
            assessment = "Ready for real-time mobile deployment"
        elif mean_time < 50:
            readiness = "✅ Good"
            assessment = "Suitable for mobile deployment"
        elif mean_time < 100:
            readiness = "⚠️ Fair"
            assessment = "May work for some mobile use cases"
        elif mean_time < 200:
            readiness = "❌ Poor"
            assessment = "Too slow for most mobile applications"
        else:
            readiness = "❌ Very Poor"
            assessment = "Not suitable for mobile deployment"
        
        return {
            'mobile_readiness': readiness,
            'assessment': assessment,
            'target_real_time': '1-10ms',
            'current_performance': f"{mean_time:.2f}ms"
        }
    
    def run_comprehensive_benchmark(self, img_size: int = 512, num_runs: int = 100) -> Dict[str, Any]:
        """Run comprehensive PE Spatial mobile benchmark."""
        print("🚀 PE SPATIAL MOBILE PERFORMANCE BENCHMARK")
        print("=" * 60)
        print("Testing Facebook's Perception Encoder Spatial models:")
        print("- PE-Spatial-T16-512 (Tiny - most mobile-friendly)")
        print("- PE-Spatial-S16-512 (Small - good mobile candidate)")
        print("- PE-Spatial-B16-512 (Base - may be too large)")
        print("=" * 60)
        
        results = {
            'benchmark_info': {
                'img_size': img_size,
                'num_runs': num_runs,
                'device': 'CPU (mobile simulation)',
                'target_performance': '1-10ms for real-time mobile'
            },
            'models': [],
            'summary': {}
        }
        
        # Test each model
        for model_name in self.available_models:
            print(f"\n{'='*20} {model_name} {'='*20}")
            
            # Load model
            model = self.load_pe_spatial_model(model_name)
            if model is None:
                continue
            
            # Get model info
            model_info = self.get_model_info(model, model_name)
            
            # Benchmark performance
            perf_stats = self.benchmark_inference(model, model_name, img_size, num_runs)
            
            # Evaluate mobile readiness
            mobile_assessment = self.evaluate_mobile_readiness(perf_stats)
            
            # Combine results
            model_result = {
                'model_info': model_info,
                'performance': perf_stats,
                'mobile_assessment': mobile_assessment
            }
            
            results['models'].append(model_result)
            
            print(f"   🎯 Mobile Assessment: {mobile_assessment['mobile_readiness']}")
            print(f"      {mobile_assessment['assessment']}")
        
        # Generate summary
        results['summary'] = self.generate_summary(results['models'])
        
        return results
    
    def generate_summary(self, models: List[Dict]) -> Dict[str, Any]:
        """Generate benchmark summary."""
        if not models:
            return {'error': 'No successful model benchmarks'}
        
        # Find best performing model
        valid_models = [m for m in models if 'error' not in m['performance']]
        if not valid_models:
            return {'error': 'No valid performance results'}
        
        best_model = min(valid_models, key=lambda x: x['performance']['mean_time_ms'])
        
        # Compare with RF-DETR baseline (133ms from previous evaluation)
        rf_detr_baseline = 133.0  # ms
        
        summary = {
            'total_models_tested': len(models),
            'successful_benchmarks': len(valid_models),
            'best_performance': {
                'model': best_model['model_info']['model_name'],
                'mean_time_ms': best_model['performance']['mean_time_ms'],
                'fps': best_model['performance']['fps'],
                'size_mb': best_model['model_info']['estimated_size_mb'],
                'mobile_readiness': best_model['mobile_assessment']['mobile_readiness']
            },
            'rf_detr_comparison': {
                'rf_detr_time_ms': rf_detr_baseline,
                'best_pe_time_ms': best_model['performance']['mean_time_ms'],
                'speedup_vs_rf_detr': rf_detr_baseline / best_model['performance']['mean_time_ms'],
                'still_too_slow_for_mobile': best_model['performance']['mean_time_ms'] > 10
            }
        }
        
        # Overall mobile readiness assessment
        best_time = best_model['performance']['mean_time_ms']
        if best_time < 10:
            summary['overall_assessment'] = "🎉 PE Spatial achieves mobile-ready performance!"
        elif best_time < 50:
            summary['overall_assessment'] = "✅ PE Spatial shows promise for mobile deployment"
        elif best_time < rf_detr_baseline:
            summary['overall_assessment'] = f"⚠️ PE Spatial is faster than RF-DETR but still not mobile-ready"
        else:
            summary['overall_assessment'] = "❌ PE Spatial doesn't improve mobile performance vs RF-DETR"
        
        return summary


def main():
    parser = argparse.ArgumentParser(description="PE Spatial Mobile Performance Benchmark")
    parser.add_argument("--img-size", type=int, default=512,
                        help="Input image size (default: 512)")
    parser.add_argument("--num-runs", type=int, default=100,
                        help="Number of benchmark runs (default: 100)")
    parser.add_argument("--model", choices=["T16", "S16", "B16", "all"], default="all",
                        help="Specific model to test (default: all)")
    
    args = parser.parse_args()
    
    # Run benchmark
    benchmark = PESpatialMobileBenchmark()
    
    if args.model != "all":
        # Test specific model
        model_name = f"PE-Spatial-{args.model}-512"
        benchmark.available_models = [model_name]
    
    results = benchmark.run_comprehensive_benchmark(args.img_size, args.num_runs)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 PE SPATIAL MOBILE BENCHMARK SUMMARY")
    print("=" * 60)
    
    summary = results['summary']
    if 'error' in summary:
        print(f"❌ {summary['error']}")
        return 1
    
    print(f"Models tested: {summary['total_models_tested']}")
    print(f"Successful benchmarks: {summary['successful_benchmarks']}")
    
    best = summary['best_performance']
    print(f"\n🏆 Best Performance:")
    print(f"   Model: {best['model']}")
    print(f"   Mean time: {best['mean_time_ms']:.2f}ms")
    print(f"   FPS: {best['fps']:.1f}")
    print(f"   Model size: {best['size_mb']:.1f}MB")
    print(f"   Mobile readiness: {best['mobile_readiness']}")
    
    comp = summary['rf_detr_comparison']
    print(f"\n📊 vs RF-DETR Comparison:")
    print(f"   RF-DETR: {comp['rf_detr_time_ms']:.2f}ms")
    print(f"   Best PE: {comp['best_pe_time_ms']:.2f}ms")
    print(f"   Speedup: {comp['speedup_vs_rf_detr']:.2f}x")
    print(f"   Still too slow: {'Yes' if comp['still_too_slow_for_mobile'] else 'No'}")
    
    print(f"\n🎯 Overall Assessment:")
    print(f"   {summary['overall_assessment']}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
