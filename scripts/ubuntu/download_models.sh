#!/bin/bash
# 模型下载脚本 - Ubuntu版本
# 使用国内镜像源

set -e

echo "========================================="
echo "Model Download Script (Ubuntu)"
echo "========================================="
echo ""

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found!"
    exit 1
fi

# 检查Python版本
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python version: $PYTHON_VERSION"

# 安装huggingface_hub（必须）
echo "Checking huggingface_hub installation..."
if ! python3 -c "import huggingface_hub" 2>/dev/null; then
    echo "Installing huggingface_hub..."
    if command -v pip3 &> /dev/null; then
        pip3 install huggingface_hub -i https://pypi.tuna.tsinghua.edu.cn/simple
    elif command -v pip &> /dev/null; then
        pip install huggingface_hub -i https://pypi.tuna.tsinghua.edu.cn/simple
    else
        python3 -m pip install huggingface_hub -i https://pypi.tuna.tsinghua.edu.cn/simple
    fi
fi

# 安装ModelScope（可选，需要Python 3.9+）
echo "Checking ModelScope installation..."
if python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)" 2>/dev/null; then
    if ! python3 -c "import modelscope" 2>/dev/null; then
        echo "Installing ModelScope (Python 3.9+ required)..."
        if command -v pip3 &> /dev/null; then
            pip3 install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple || echo "⚠️  ModelScope installation failed, will use HuggingFace mirror"
        elif command -v pip &> /dev/null; then
            pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple || echo "⚠️  ModelScope installation failed, will use HuggingFace mirror"
        else
            python3 -m pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple || echo "⚠️  ModelScope installation failed, will use HuggingFace mirror"
        fi
    fi
else
    echo "⚠️  ModelScope requires Python 3.9+, will use HuggingFace mirror instead"
fi

# 模型列表
MODELS=(
    "Qwen2.5-3B-Instruct"
    "Llama3-8B-Instruct"
    "Gemma2-9B-Instruct"
)

echo ""
echo "Available models:"
for i in "${!MODELS[@]}"; do
    echo "  $((i+1)). ${MODELS[$i]}"
done
echo "  4. All models"
echo ""

read -p "Select model to download (1-4): " choice

case $choice in
    1)
        MODEL="Qwen2.5-3B-Instruct"
        ;;
    2)
        MODEL="Llama3-8B-Instruct"
        ;;
    3)
        MODEL="Gemma2-9B-Instruct"
        ;;
    4)
        MODEL="all"
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

# 运行下载脚本
echo ""
echo "Starting download..."
# 根据Python版本选择下载脚本
PYTHON_MAJOR=$(python3 -c "import sys; print(sys.version_info.major)")
PYTHON_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")

if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]; then
    # Python 3.8使用专用脚本
    if [ -f "download_models_python38.py" ]; then
        python3 download_models_python38.py --model "$MODEL"
    else
        python3 download_models_china.py --model "$MODEL"
    fi
else
    # Python 3.9+使用标准脚本
    python3 download_models_china.py --model "$MODEL" --use-modelscope
fi

echo ""
echo "========================================="
echo "Download completed!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Verify models: python3 check_model_integrity.py ./models/<model_name>"
echo "2. Run experiments: bash scripts/ubuntu/run_a_level_experiments.sh"
echo ""

