# Windows系统运行指南

## 📋 完整实验流程（Windows系统）

### Step 0: 环境准备

```powershell
# 1. 检查Python版本（需要3.8+）
python --version

# 2. 检查GPU（如果使用）
nvidia-smi

# 3. 检查环境依赖
python check_environment.py

# 4. 安装Python依赖
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Step 1: 下载模型

```powershell
# 方式1: 使用下载脚本（推荐，自动使用ModelScope）
.\scripts\windows\download_models.ps1

# 方式2: 直接使用Python脚本（Python 3.9+，优先使用ModelScope）
python download_models_china.py --model all --use-modelscope

# 方式3: Python 3.8（自动尝试ModelScope）
python download_models_python38.py --model all

# 单独下载某个模型
python download_models_china.py --model Llama3-8B-Instruct --use-modelscope
python download_models_china.py --model Gemma2-9B-Instruct --use-modelscope
```

**注意**: 新版本脚本优先使用ModelScope（国内访问快，无需认证）

**验证模型**:
```powershell
python check_model_integrity.py .\models\Qwen2.5-3B-Instruct
```

### Step 2: 测试Baseline

```powershell
# 基础功能测试
python test_baselines.py

# 小样本真实模型测试（H2O）
python StreamingLLM_GPE/evaluate/multi_model_eval.py `
    --LLM_backbone Qwen `
    --LLM_path .\models\Qwen2.5-3B-Instruct `
    --use_h2o `
    --h2o_budget 2048 `
    --output_dir .\output_logs\test_h2o `
    --max_samples 2 `
    --quantization 4bit

# 小样本真实模型测试（StreamingLLM）
python StreamingLLM_GPE/evaluate/multi_model_eval.py `
    --LLM_backbone Qwen `
    --LLM_path .\models\Qwen2.5-3B-Instruct `
    --use_streamingllm `
    --streamingllm_window 512 `
    --output_dir .\output_logs\test_streamingllm `
    --max_samples 2 `
    --quantization 4bit
```

### Step 3: 运行完整实验

```powershell
# 运行A级论文完整实验
.\scripts\windows\run_a_level_experiments.ps1
```

### Step 4: 分析结果

```powershell
# 分析实验结果
python analyze_experiment_results.py `
    --output_dir .\output_logs\a_level_paper\long_seq_10000 `
    --detailed `
    --save_csv .\output_logs\summary.csv

# 生成可视化
python visualize_results.py `
    --results_dir .\output_logs\a_level_paper `
    --output_dir .\output_logs\figures
```

## 📝 脚本说明

- `download_models.ps1` - 模型下载脚本
- `run_a_level_experiments.ps1` - A级论文完整实验脚本

## ⚠️ 注意事项

1. 使用PowerShell执行脚本（不是CMD）
2. 如果遇到执行策略错误，运行：
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```
3. 确保有足够的磁盘空间（至少50GB）
4. 路径使用反斜杠`\`（Windows格式）

