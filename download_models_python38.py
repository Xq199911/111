#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
针对Python 3.8的模型下载脚本
支持HuggingFace Token认证
"""
import os
import sys

# 模型配置
MODELS = {
    "Qwen2.5-3B-Instruct": {
        "huggingface": "Qwen/Qwen2.5-3B-Instruct",
        "modelscope": "qwen/Qwen2.5-3B-Instruct",
        "output_dir": "./models/Qwen2.5-3B-Instruct",
        "requires_auth": False
    },
    "Llama3-8B-Instruct": {
        "huggingface": "meta-llama/Meta-Llama-3-8B-Instruct",
        "modelscope": "LLM-Research/Meta-Llama-3-8B-Instruct",
        "output_dir": "./models/Llama3-8B-Instruct",
        "requires_auth": False  # ModelScope不需要认证
    },
    "Gemma2-9B-Instruct": {
        "huggingface": "google/gemma-2-9b-it",
        "modelscope": "LLM-Research/gemma-2-9b-it",
        "output_dir": "./models/Gemma2-9B-Instruct",
        "requires_auth": False
    }
}

def download_with_modelscope(model_name, model_id, output_dir):
    """使用ModelScope下载模型（推荐方式）"""
    try:
        import sys
        # 检查Python版本（ModelScope需要Python 3.9+）
        if sys.version_info < (3, 9):
            return False
        
        from modelscope import snapshot_download
        print(f"📥 Downloading {model_name} from ModelScope...")
        print(f"   Model ID: {model_id}")
        snapshot_download(
            model_id,
            cache_dir=None,
            local_dir=output_dir
        )
        print(f"✅ {model_name} downloaded successfully from ModelScope")
        return True
    except ImportError:
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
            print("⚠️  git-lfs not found. Please install git-lfs first.")
            return False
        
        # 构建ModelScope git URL
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

def download_model(model_name, model_id, output_dir, requires_auth=False, modelscope_id=None):
    """下载模型"""
    print("=" * 70)
    print(f"Downloading Model: {model_name}")
    print("=" * 70)
    print(f"HuggingFace ID: {model_id}")
    if modelscope_id:
        print(f"ModelScope ID: {modelscope_id}")
    print(f"Output: {output_dir}")
    if requires_auth:
        print("⚠️  This model requires HuggingFace authentication (if using HuggingFace)")
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
    
    # 获取token
    token = None
    if requires_auth:
        # 从环境变量获取
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        
        if not token:
            print("=" * 70)
            print("⚠️  HuggingFace Token Required!")
            print("=" * 70)
            print("This model requires HuggingFace authentication.")
            print()
            print("Steps to get token:")
            print("1. Visit: https://huggingface.co/settings/tokens")
            print("2. Create a new token with 'read' permission")
            print("3. Accept the model's license agreement:")
            print(f"   https://huggingface.co/{model_id}")
            print()
            print("Then set the token:")
            print("  export HF_TOKEN=your_token_here")
            print()
            response = input("Do you have a token? Enter it now (or press Enter to skip): ")
            if response.strip():
                token = response.strip()
            else:
                print("❌ Token required. Skipping download.")
                return False
    
    # 尝试多种下载方式
    success = False
    
    # 策略1: 优先使用ModelScope（推荐，国内访问快，无需认证）
    if modelscope_id:
        import sys
        if sys.version_info >= (3, 9):
            print("🔄 Strategy 1: Trying ModelScope (recommended, no authentication needed)...")
            success = download_with_modelscope(model_name, modelscope_id, output_dir)
            print()
    
    # 策略2: 如果ModelScope失败，尝试git clone（对于Llama3等模型）
    if not success and modelscope_id:
        print("🔄 Strategy 2: Trying ModelScope git clone...")
        success = download_with_git_clone(model_name, modelscope_id, output_dir)
        print()
    
    # 策略3: 如果都失败，尝试HuggingFace镜像
    if not success:
        try:
            # 设置HuggingFace镜像（可选）
            use_mirror = os.environ.get("USE_HF_MIRROR", "true").lower() == "true"
            
            if use_mirror:
                os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
                print("🔄 Strategy 3: Trying HuggingFace mirror (hf-mirror.com)...")
            else:
                os.environ.pop("HF_ENDPOINT", None)
                print("🔄 Strategy 3: Trying HuggingFace directly...")
            
            # 设置token
            if token:
                os.environ["HF_TOKEN"] = token
                print("🔑 Using HuggingFace token for authentication")
            
            print(f"Downloading {model_name}...")
            print("This may take a while (several GB to download)...")
            print()
            
            from huggingface_hub import snapshot_download
            
            # 新版本huggingface_hub不再支持local_dir_use_symlinks参数
            download_kwargs = {
                "repo_id": model_id,
                "local_dir": output_dir,
            }
            
            if use_mirror:
                download_kwargs["endpoint"] = "https://hf-mirror.com"
            
            if token:
                download_kwargs["token"] = token
            
            snapshot_download(**download_kwargs)
            
            print()
            print("=" * 70)
            print(f"✅ {model_name} downloaded successfully!")
            print(f"   Saved to: {output_dir}")
            print("=" * 70)
            return True
        except ImportError:
            print("❌ huggingface_hub not installed")
            print("Installing...")
            os.system("pip install huggingface_hub -i https://pypi.tuna.tsinghua.edu.cn/simple")
            print("Please run the script again")
            return False
        except Exception as e:
            error_str = str(e)
            print()
            print("=" * 70)
            print(f"❌ Error: {error_str}")
            print("=" * 70)
            
            if "403" in error_str or "gated" in error_str.lower() or "authentication" in error_str.lower():
                print()
                print("💡 This model requires HuggingFace authentication:")
                print("1. Get token from: https://huggingface.co/settings/tokens")
                print(f"2. Accept license: https://huggingface.co/{model_id}")
                print("3. Set token: export HF_TOKEN=your_token_here")
                print("4. Run this script again")
            elif "cannot find" in error_str.lower() or "connection" in error_str.lower():
                print()
                print("💡 Network or mirror issue. Try:")
                print("1. Use ModelScope instead (recommended)")
                print("2. Check internet connection")
                print("3. Try again later")
            
            return False
    
    if not success:
        print("=" * 70)
        print(f"❌ {model_name} download failed with all methods!")
        print("=" * 70)
        print("\n💡 Troubleshooting:")
        print("1. For Llama3/Gemma2: Use ModelScope (recommended)")
        print("2. Install ModelScope: pip install modelscope")
        print("3. Install git-lfs for git clone method")
        return False
    
    return True

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download models for Python 3.8",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download Qwen (no auth needed)
  python download_models_python38.py --model Qwen2.5-3B-Instruct
  
  # Download Llama3 (requires token)
  export HF_TOKEN=your_token_here
  python download_models_python38.py --model Llama3-8B-Instruct
  
  # Download Gemma2 without mirror
  USE_HF_MIRROR=false python download_models_python38.py --model Gemma2-9B-Instruct
  
  # Download all (will prompt for token if needed)
  python download_models_python38.py --model all
        """
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODELS.keys()) + ["all"],
        default="all",
        help="Model to download (default: all)"
    )
    
    args = parser.parse_args()
    
    # 检查依赖
    try:
        import huggingface_hub
    except ImportError:
        print("Installing huggingface_hub...")
        os.system("pip install huggingface_hub -i https://pypi.tuna.tsinghua.edu.cn/simple")
        import huggingface_hub
    
    print()
    
    # 下载模型
    if args.model == "all":
        print("Downloading all models...")
        print()
        success_count = 0
        for model_name, model_info in MODELS.items():
            if download_model(
                model_name,
                model_info["huggingface"],
                model_info["output_dir"],
                model_info["requires_auth"],
                model_info.get("modelscope")
            ):
                success_count += 1
            print()
        
        print("=" * 70)
        print(f"Download Summary: {success_count}/{len(MODELS)} models downloaded")
        print("=" * 70)
    else:
        model_info = MODELS[args.model]
        download_model(
            args.model,
            model_info["huggingface"],
            model_info["output_dir"],
            model_info["requires_auth"],
            model_info.get("modelscope")
        )

if __name__ == "__main__":
    main()

