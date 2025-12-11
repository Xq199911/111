#!/bin/bash
# 环境配置脚本
# 用于Ubuntu系统，安装所有必需的依赖

set -e  # 遇到错误立即退出

echo "========================================="
echo "StreamingLLM Environment Setup"
echo "========================================="
echo ""

# 检查Python版本
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# 检查pip
if ! command -v pip &> /dev/null; then
    echo "pip not found. Installing pip..."
    python3 -m ensurepip --upgrade
fi

# 升级pip
echo "Upgrading pip..."
pip install --upgrade pip

# 检查CUDA
echo ""
echo "Checking CUDA..."
if command -v nvcc &> /dev/null; then
    nvcc_version=$(nvcc --version | grep "release" | awk '{print $5}' | cut -c2-)
    echo "CUDA version: $nvcc_version"
else
    echo "Warning: CUDA not found. PyTorch will use CPU."
fi

# 检查GPU
echo ""
echo "Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "Warning: nvidia-smi not found. GPU may not be available."
fi

# 安装依赖
echo ""
echo "========================================="
echo "Installing Python dependencies..."
echo "========================================="
echo ""

# 安装PyTorch (根据CUDA版本)
if command -v nvcc &> /dev/null; then
    echo "Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "Installing PyTorch (CPU only)..."
    pip install torch torchvision torchaudio
fi

# 安装其他依赖
echo ""
echo "Installing other dependencies..."
pip install -r requirements.txt

# 验证安装
echo ""
echo "========================================="
echo "Verifying installation..."
echo "========================================="
echo ""

python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    python3 -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
    python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
fi

python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python3 -c "import accelerate; print(f'Accelerate: {accelerate.__version__}')"
python3 -c "import datasets; print(f'Datasets: {datasets.__version__}')"
python3 -c "import pandas; print(f'Pandas: {pandas.__version__}')"
python3 -c "import matplotlib; print(f'Matplotlib: {matplotlib.__version__}')"
python3 -c "import sacrebleu; print(f'SacreBLEU: {sacrebleu.__version__}')"

echo ""
echo "========================================="
echo "Environment setup completed!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Verify model path: ./models/Qwen2.5-3B-Instruct"
echo "2. Run experiments: ./run_a_level_experiments.sh"
echo ""

