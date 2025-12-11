#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
环境检查脚本
检查所有必需的依赖是否已安装
"""
import sys

def check_package(package_name, import_name=None):
    """检查包是否已安装"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {package_name}: {version}")
        return True
    except ImportError:
        print(f"❌ {package_name}: NOT INSTALLED")
        return False

def check_cuda():
    """检查CUDA和GPU"""
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA available: Yes")
            print(f"✅ CUDA version: {torch.version.cuda}")
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
            print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            return True
        else:
            print(f"⚠️  CUDA available: No (CPU only)")
            return False
    except ImportError:
        print(f"❌ PyTorch: NOT INSTALLED")
        return False

def main():
    print("="*60)
    print("Environment Check")
    print("="*60)
    print()
    
    print("Python version:", sys.version)
    print()
    
    print("Core Dependencies:")
    print("-" * 60)
    packages = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("accelerate", "accelerate"),
        ("bitsandbytes", "bitsandbytes"),
        ("datasets", "datasets"),
        ("sacrebleu", "sacrebleu"),
        ("ijson", "ijson"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("matplotlib", "matplotlib"),
        ("tqdm", "tqdm"),
        ("peft", "peft"),
    ]
    
    all_ok = True
    for pkg_name, import_name in packages:
        if not check_package(pkg_name, import_name):
            all_ok = False
    
    print()
    print("CUDA and GPU:")
    print("-" * 60)
    cuda_ok = check_cuda()
    
    print()
    print("="*60)
    if all_ok and cuda_ok:
        print("✅ All dependencies are installed!")
        print("✅ Environment is ready for experiments!")
    elif all_ok:
        print("⚠️  All dependencies are installed, but CUDA is not available.")
        print("⚠️  Experiments will run on CPU (very slow).")
    else:
        print("❌ Some dependencies are missing!")
        print()
        print("To install missing packages, run:")
        print("  pip install -r requirements.txt")
        print()
        print("Or run the setup script:")
        print("  chmod +x setup_environment.sh")
        print("  ./setup_environment.sh")
    
    print("="*60)

if __name__ == "__main__":
    main()

