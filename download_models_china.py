#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型下载脚本 - 使用国内镜像源
支持ModelScope和HuggingFace镜像
"""
import os
import sys
from pathlib import Path

# 模型配置
MODELS = {
    "Qwen2.5-3B-Instruct": {
        "huggingface": "Qwen/Qwen2.5-3B-Instruct",
        "modelscope": "qwen/Qwen2.5-3B-Instruct",
        "output_dir": "./models/Qwen2.5-3B-Instruct"
    },
    "Llama3-8B-Instruct": {
        "huggingface": "meta-llama/Meta-Llama-3-8B-Instruct",
        "modelscope": "LLM-Research/Meta-Llama-3-8B-Instruct",
        "output_dir": "./models/Llama3-8B-Instruct"
    },
    "Gemma2-9B-Instruct": {
        "huggingface": "google/gemma-2-9b-it",
        "modelscope": "LLM-Research/gemma-2-9b-it",
        "output_dir": "./models/Gemma2-9B-Instruct"
    }
}

def download_with_modelscope(model_name, model_id, output_dir):
    """使用ModelScope下载模型（推荐方式）"""
    try:
        import sys
        # 检查Python版本（ModelScope需要Python 3.9+）
        if sys.version_info < (3, 9):
            print(f"⚠️  ModelScope requires Python 3.9+, but you have {sys.version_info.major}.{sys.version_info.minor}")
            return False
        
        from modelscope import snapshot_download
        print(f"📥 Downloading {model_name} from ModelScope...")
        print(f"   Model ID: {model_id}")
        # 新版本ModelScope不再支持local_dir_use_symlinks参数
        snapshot_download(
            model_id,
            cache_dir=None,
            local_dir=output_dir
        )
        print(f"✅ {model_name} downloaded successfully from ModelScope")
        return True
    except ImportError:
        print("⚠️  ModelScope not installed.")
        return False
    except Exception as e:
        print(f"⚠️  ModelScope download failed: {e}")
        return False

def download_with_git_clone(model_name, model_id, output_dir):
    """使用git clone从ModelScope下载模型（备选方案）"""
    try:
        import subprocess
        import shutil
        
        # 检查git是否安装
        try:
            subprocess.run(["git", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️  git not found. Please install git first.")
            return False
        
        # 检查git-lfs是否安装
        try:
            subprocess.run(["git", "lfs", "version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️  git-lfs not found. Installing...")
            print("   Please install git-lfs: https://git-lfs.github.com/")
            return False
        
        # 构建ModelScope git URL
        # ModelScope URL格式: https://www.modelscope.cn/{model_id}.git
        git_url = f"https://www.modelscope.cn/{model_id}.git"
        
        print(f"📥 Downloading {model_name} from ModelScope using git clone...")
        print(f"   URL: {git_url}")
        
        # 如果输出目录已存在，先删除
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        
        # 初始化git-lfs
        subprocess.run(["git", "lfs", "install"], check=True)
        
        # 克隆仓库
        subprocess.run(["git", "clone", git_url, output_dir], check=True)
        
        print(f"✅ {model_name} downloaded successfully using git clone")
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠️  git clone failed: {e}")
        return False
    except Exception as e:
        print(f"⚠️  git clone error: {e}")
        return False

def download_with_huggingface_mirror(model_name, model_id, output_dir):
    """使用HuggingFace镜像下载模型"""
    try:
        import os
        # 设置HuggingFace镜像
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        
        print(f"Downloading {model_name} from HuggingFace mirror (hf-mirror.com)...")
        
        # 使用huggingface_hub下载完整模型
        try:
            from huggingface_hub import snapshot_download
            print("  Downloading complete model...")
            # 新版本huggingface_hub不再支持local_dir_use_symlinks参数
            snapshot_download(
                repo_id=model_id,
                local_dir=output_dir,
                endpoint="https://hf-mirror.com"
            )
            print(f"✅ {model_name} downloaded successfully from HuggingFace mirror")
            return True
        except ImportError:
            print("  Installing huggingface_hub...")
            os.system("pip install huggingface_hub -i https://pypi.tuna.tsinghua.edu.cn/simple")
            from huggingface_hub import snapshot_download
            # 新版本huggingface_hub不再支持local_dir_use_symlinks参数
            snapshot_download(
                repo_id=model_id,
                local_dir=output_dir,
                endpoint="https://hf-mirror.com"
            )
            print(f"✅ {model_name} downloaded successfully from HuggingFace mirror")
            return True
    except Exception as e:
        print(f"❌ HuggingFace mirror download failed: {e}")
        return False

def download_model(model_name, use_modelscope=True):
    """下载模型"""
    if model_name not in MODELS:
        print(f"❌ Unknown model: {model_name}")
        print(f"Available models: {list(MODELS.keys())}")
        return False
    
    model_info = MODELS[model_name]
    output_dir = model_info["output_dir"]
    
    print("=" * 60)
    print(f"Downloading Model: {model_name}")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print()
    
    # 检查是否已存在
    if os.path.exists(output_dir) and os.listdir(output_dir):
        print(f"⚠️  Model directory already exists: {output_dir}")
        response = input("Delete existing model and re-download? (y/n): ")
        if response.lower() == 'y':
            import shutil
            shutil.rmtree(output_dir)
            print(f"✅ Deleted existing model directory")
        else:
            print("Skipping download")
            return True
    
    # 尝试下载（优先使用ModelScope，因为更稳定）
    success = False
    
    # 策略1: 优先使用ModelScope（推荐，国内访问快，无需认证）
    if use_modelscope:
        print("🔄 Strategy 1: Trying ModelScope (recommended, no authentication needed)...")
        success = download_with_modelscope(
            model_name,
            model_info["modelscope"],
            output_dir
        )
        print()
    
    # 策略2: 如果ModelScope失败，尝试git clone（对于Llama3等模型）
    if not success and use_modelscope:
        print("🔄 Strategy 2: Trying ModelScope git clone...")
        success = download_with_git_clone(
            model_name,
            model_info["modelscope"],
            output_dir
        )
        print()
    
    # 策略3: 如果都失败，尝试HuggingFace镜像
    if not success:
        print("🔄 Strategy 3: Trying HuggingFace mirror...")
        success = download_with_huggingface_mirror(
            model_name,
            model_info["huggingface"],
            output_dir
        )
        print()
    
    if success:
        print(f"✅ {model_name} download completed!")
        return True
    else:
        print(f"❌ {model_name} download failed!")
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Download models using Chinese mirrors")
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODELS.keys()) + ["all"],
        default="all",
        help="Model to download (default: all)"
    )
    parser.add_argument(
        "--use-modelscope",
        action="store_true",
        default=True,
        help="Use ModelScope (default: True)"
    )
    parser.add_argument(
        "--use-hf-mirror",
        action="store_true",
        help="Use HuggingFace mirror (hf-mirror.com)"
    )
    
    args = parser.parse_args()
    
    # 安装huggingface_hub（必须）
    try:
        import huggingface_hub
    except ImportError:
        print("Installing huggingface_hub...")
        os.system("pip install huggingface_hub -i https://pypi.tuna.tsinghua.edu.cn/simple")
    
    # 安装ModelScope（可选，需要Python 3.9+）
    if args.use_modelscope:
        import sys
        if sys.version_info >= (3, 9):
            try:
                import modelscope
            except ImportError:
                print("Installing ModelScope (requires Python 3.9+)...")
                os.system("pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple")
        else:
            print(f"⚠️  ModelScope requires Python 3.9+, but you have {sys.version_info.major}.{sys.version_info.minor}")
            print("   Will use HuggingFace mirror instead")
    
    # 下载模型
    if args.model == "all":
        print("Downloading all models...")
        print()
        for model_name in MODELS.keys():
            download_model(model_name, use_modelscope=args.use_modelscope)
            print()
    else:
        download_model(args.model, use_modelscope=args.use_modelscope)
    
    print("=" * 60)
    print("Download process completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()

