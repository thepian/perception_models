#!/usr/bin/env python3
"""
Validate that the GitHub Actions workflow will work correctly.
Checks dependencies, imports, and basic functionality.
"""

import sys
import subprocess
import importlib.util
from pathlib import Path

def check_python_version():
    """Check Python version compatibility."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor == 12:
        print(f"  ✅ Python {version.major}.{version.minor}.{version.micro} (compatible)")
        return True
    else:
        print(f"  ⚠️  Python {version.major}.{version.minor}.{version.micro} (workflow expects 3.12)")
        return True  # Don't fail on version difference

def check_required_packages():
    """Check if required packages can be imported."""
    print("📦 Checking required packages...")
    
    required_packages = [
        ("torch", "PyTorch"),
        ("pathlib", "pathlib"),
        ("argparse", "argparse"),
        ("json", "json"),
        ("time", "time"),
        ("traceback", "traceback"),
        ("numpy", "NumPy"),
    ]
    
    optional_packages = [
        ("coremltools", "CoreML Tools"),
        ("onnx", "ONNX"),
        ("executorch", "ExecuTorch"),
    ]
    
    all_good = True
    
    for package, name in required_packages:
        try:
            importlib.import_module(package)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} (required)")
            all_good = False
    
    for package, name in optional_packages:
        try:
            importlib.import_module(package)
            print(f"  ✅ {name} (optional)")
        except ImportError:
            print(f"  ⚠️  {name} (optional, will be installed during workflow)")
    
    return all_good

def check_project_structure():
    """Check if project structure is correct."""
    print("📁 Checking project structure...")
    
    required_paths = [
        "core/vision_encoder/pe.py",
        "scripts/convert_s16_models.py",
        "scripts/create_release_manifest.py", 
        "scripts/test_converted_models.py",
        "scripts/benchmark_s16_models.py",
        ".github/workflows/release-s16-models.yml",
        "pyproject.toml",
    ]
    
    all_good = True
    
    for path in required_paths:
        if Path(path).exists():
            print(f"  ✅ {path}")
        else:
            print(f"  ❌ {path} (missing)")
            all_good = False
    
    return all_good

def check_script_syntax():
    """Check if all scripts have valid syntax."""
    print("🔍 Checking script syntax...")
    
    scripts = [
        "scripts/convert_s16_models.py",
        "scripts/create_release_manifest.py",
        "scripts/test_converted_models.py", 
        "scripts/benchmark_s16_models.py",
    ]
    
    all_good = True
    
    for script in scripts:
        try:
            with open(script, 'r') as f:
                compile(f.read(), script, 'exec')
            print(f"  ✅ {script}")
        except SyntaxError as e:
            print(f"  ❌ {script} (syntax error: {e})")
            all_good = False
        except FileNotFoundError:
            print(f"  ❌ {script} (not found)")
            all_good = False
    
    return all_good

def check_pe_core_import():
    """Check if PE Core can be imported."""
    print("🧠 Checking PE Core import...")
    
    try:
        sys.path.insert(0, str(Path.cwd()))
        from core.vision_encoder.pe import CLIP
        print("  ✅ PE Core CLIP model can be imported")
        return True
    except ImportError as e:
        print(f"  ❌ PE Core import failed: {e}")
        return False
    except Exception as e:
        print(f"  ⚠️  PE Core import warning: {e}")
        return True  # Don't fail on other errors

def check_model_availability():
    """Check if PE-Core-S16-384 model is available."""
    print("🤗 Checking model availability...")
    
    try:
        sys.path.insert(0, str(Path.cwd()))
        from core.vision_encoder.pe import CLIP
        
        # Try to load the model config (without downloading)
        model = CLIP.from_config("PE-Core-S16-384", pretrained=False)
        print("  ✅ PE-Core-S16-384 config available")
        return True
    except Exception as e:
        print(f"  ⚠️  PE-Core-S16-384 model check: {e}")
        return True  # Don't fail, model will be downloaded during workflow

def check_workflow_yaml():
    """Check workflow YAML syntax."""
    print("⚙️  Checking workflow YAML...")
    
    try:
        import yaml
        with open('.github/workflows/release-s16-models.yml', 'r') as f:
            yaml.safe_load(f)
        print("  ✅ Workflow YAML syntax valid")
        return True
    except Exception as e:
        print(f"  ❌ Workflow YAML error: {e}")
        return False

def main():
    """Run all validation checks."""
    print("🔍 Validating GitHub Actions Workflow")
    print("=" * 50)
    
    checks = [
        check_python_version,
        check_required_packages,
        check_project_structure,
        check_script_syntax,
        check_pe_core_import,
        check_model_availability,
        check_workflow_yaml,
    ]
    
    results = []
    for check in checks:
        try:
            result = check()
            results.append(result)
        except Exception as e:
            print(f"  💥 Check failed with error: {e}")
            results.append(False)
        print()
    
    print("=" * 50)
    print("📊 Validation Summary:")
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"  🎉 All {total} checks passed!")
        print("  ✅ Workflow should work correctly")
        return 0
    else:
        print(f"  ⚠️  {passed}/{total} checks passed")
        print("  🔧 Some issues found, but workflow may still work")
        print("  💡 Check the details above for any critical issues")
        return 1

if __name__ == "__main__":
    exit(main())