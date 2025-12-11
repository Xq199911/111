# A级论文完整实验指南（通用版）

> **注意**: 本指南为通用版本。请根据您的操作系统查看对应的详细指南：
> - **Ubuntu系统**: 查看 `scripts/ubuntu/README.md`
> - **Windows系统**: 查看 `scripts/windows/README.md`

## 🎯 项目目标

**研究问题**: Head-Aware Dynamic KV Budgeting for Efficient Long-Sequence Inference

**目标**: 发表A级会议/期刊论文（ACL, EMNLP, NeurIPS, ICML等）

**核心方法**: 
- Head-Aware Cache: 根据attention head的功能特性动态分配KV cache预算
- Group-Aware Eviction: 基于head group的协同eviction策略

**Baseline对比**:
- H2O (Heavy-Hitter Oracle)
- StreamingLLM (Fixed Window + Attention Sinks)

---

## 📁 项目结构

```
StreamingLLM/
├── StreamingLLM_GPE/              # 核心代码
│   ├── baselines/                 # Baseline实现
│   │   ├── h2o_cache.py          # H2O baseline
│   │   └── streamingllm_cache.py # StreamingLLM baseline
│   ├── models/                    # 模型实现
│   │   ├── Qwen2_5/              # Qwen模型
│   │   ├── Llama3/               # Llama模型
│   │   └── Gemma2/               # Gemma模型
│   ├── evaluate/                 # 评估脚本
│   │   └── multi_model_eval.py   # 主评估脚本
│   ├── utils/                     # 工具函数
│   └── configs/                   # 配置文件
├── models/                        # 模型文件（需下载）
│   └── Qwen2.5-3B-Instruct/      # Qwen模型
├── data_raw/                      # 原始数据
├── output_logs/                   # 实验结果输出
├── run_a_level_experiments.sh     # A级论文实验脚本 ⭐
├── run_multi_model_experiments.sh # 多模型实验脚本
├── download_models_china.py       # 模型下载脚本
├── download_models_python38.py    # Python 3.8下载脚本
├── check_environment.py           # 环境检查
├── check_model_integrity.py       # 模型检查
├── analyze_experiment_results.py  # 结果分析
├── visualize_results.py          # 可视化
├── test_baselines.py              # Baseline测试
└── EXPERIMENT_GUIDE.md            # 本文件 ⭐
```

---

## 📋 完整实验流程（按顺序执行）

### Step 0: 环境准备

**目标**: 确保环境配置正确

**执行命令**:
```bash
# 1. 检查Python版本（需要3.8+）
python --version

# 2. 检查GPU（如果使用）
nvidia-smi

# 3. 检查环境依赖
python check_environment.py

# 4. 安装Python依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**验证**: 所有检查通过，无错误信息

**预期时间**: 5-10分钟

---

### Step 1: 下载模型

**目标**: 下载实验所需的大语言模型

**执行命令**:
```bash
# 方法1: 直接使用Python脚本（推荐，避免pip命令问题）
# Python 3.8
python3 download_models_python38.py --model Qwen2.5-3B-Instruct

# Python 3.9+
python3 download_models_china.py --model Qwen2.5-3B-Instruct --use-modelscope

# 方法2: 使用bash脚本（如果pip命令可用）
bash setup_models_china.sh
```

**如果遇到pip命令找不到的错误**:
```bash
# 使用python -m pip安装依赖
python3 -m pip install huggingface_hub -i https://pypi.tuna.tsinghua.edu.cn/simple

# 然后直接使用Python脚本下载
python3 download_models_python38.py --model Qwen2.5-3B-Instruct
```

**验证模型**:
```bash
# 检查模型完整性
python check_model_integrity.py ./models/Qwen2.5-3B-Instruct
```

**预期输出**: `Model integrity check passed`

**预期时间**: 2-4小时（取决于网络）

**注意事项**:
- 至少需要50GB磁盘空间
- Qwen2.5-3B-Instruct约6GB（必须）
- Llama3-8B-Instruct约16GB（可选，用于多模型验证）
- Gemma2-9B-Instruct约18GB（可选，用于多模型验证）

---

### Step 2: 测试Baseline实现

**目标**: 验证H2O和StreamingLLM baseline是否正确实现

**执行命令**:
```bash
# 1. 基础功能测试
python test_baselines.py

# 2. 小样本真实模型测试（H2O）
python StreamingLLM_GPE/evaluate/multi_model_eval.py \
    --LLM_backbone Qwen \
    --LLM_path ./models/Qwen2.5-3B-Instruct \
    --use_h2o \
    --h2o_budget 2048 \
    --output_dir ./output_logs/test_h2o \
    --max_samples 2 \
    --quantization 4bit

# 3. 小样本真实模型测试（StreamingLLM）
python StreamingLLM_GPE/evaluate/multi_model_eval.py \
    --LLM_backbone Qwen \
    --LLM_path ./models/Qwen2.5-3B-Instruct \
    --use_streamingllm \
    --streamingllm_window 512 \
    --output_dir ./output_logs/test_streamingllm \
    --max_samples 2 \
    --quantization 4bit
```

**验证输出**:
```bash
# 检查结果文件是否存在
ls -la ./output_logs/test_h2o/results.json
ls -la ./output_logs/test_streamingllm/results.json

# 查看结果
cat ./output_logs/test_h2o/results.json | grep bleu
cat ./output_logs/test_streamingllm/results.json | grep bleu
```

**预期输出**: 
- `test_baselines.py` 所有测试通过
- 生成results.json文件
- BLEU分数正常（不会为0或异常低）

**预期时间**: 10-20分钟

---

### Step 3: 运行A级论文完整实验 ⭐⭐⭐⭐⭐

**目标**: 运行所有必需的实验（长序列对比、消融实验、预算分析）

**执行命令**:
```bash
bash run_a_level_experiments.sh
```

**实验包含的4个阶段**:

#### Phase 1: 长序列内存效率对比

**测试内容**:
- 序列长度: 2000, 5000, 10000, 20000 tokens
- 对比方法: Baseline (GPE), H2O, StreamingLLM, Head-Aware, Full
- 样本数: 100 samples/方法

**输出目录**: `./output_logs/a_level_paper/long_seq_{长度}/{方法名}/`

**预期时间**: 每个序列长度约2-4小时

#### Phase 2: 预算影响分析

**测试内容**:
- 预算: 2048, 4096, 8192 tokens/layer
- 方法: Full (Head-Aware + Group-Aware)
- 样本数: 100 samples

**输出目录**: `./output_logs/a_level_paper/budget_{预算}/`

**预期时间**: 约1-2小时

#### Phase 3: 消融实验

**测试内容**:
- 序列长度: 5000 tokens
- 对比配置:
  1. Baseline (GPE only)
  2. Head-Aware only
  3. Group-Aware only
  4. Full (Head-Aware + Group-Aware)
- 样本数: 100 samples/配置

**输出目录**: `./output_logs/a_level_paper/ablation/{配置名}/`

**预期时间**: 约1-2小时

#### Phase 4: 结果分析和可视化

**自动执行**:
- 分析长序列实验结果
- 分析消融实验结果
- 生成可视化图表

**输出文件**:
- `./output_logs/a_level_paper/long_seq_10000_summary.csv`
- `./output_logs/a_level_paper/ablation_summary.csv`
- `./output_logs/a_level_paper/figures/`

**预期时间**: 10-30分钟

**总预期时间**: 4-8小时（取决于硬件）

**验证实验运行**:
```bash
# 检查输出目录
ls -la ./output_logs/a_level_paper/long_seq_10000/

# 应该看到:
# baseline/
# h2o/
# streamingllm/
# head_aware/
# full/
```

---

### Step 4: 多模型验证（可选但推荐）

**目标**: 证明方法不依赖特定模型架构

**执行命令**:
```bash
bash run_multi_model_experiments.sh
```

**验证的模型**:
- Qwen2.5-3B-Instruct
- Llama3-8B-Instruct（如果已下载）
- Gemma2-9B-Instruct（如果已下载）

**输出目录**: `./output_logs/multi_model/{模型名}/`

**预期时间**: 每个模型约1-2天

**注意**: 如果只有Qwen模型，可以跳过此步骤，先用Qwen完成所有实验

---

### Step 5: 结果分析和论文准备

**目标**: 分析实验结果，准备论文数据

**执行命令**:
```bash
# 1. 分析长序列实验结果
python analyze_experiment_results.py \
    --output_dir ./output_logs/a_level_paper/long_seq_10000 \
    --detailed \
    --save_csv ./output_logs/long_seq_summary.csv \
    --save_json ./output_logs/long_seq_summary.json \
    --save_latex ./output_logs/long_seq_table.tex

# 2. 分析消融实验结果
python analyze_experiment_results.py \
    --output_dir ./output_logs/a_level_paper/ablation \
    --detailed \
    --save_csv ./output_logs/ablation_summary.csv \
    --save_json ./output_logs/ablation_summary.json \
    --save_latex ./output_logs/ablation_table.tex

# 3. 生成可视化图表
python visualize_results.py \
    --results_dir ./output_logs/a_level_paper \
    --output_dir ./output_logs/figures \
    --include_budget
```

**输出文件**:
- CSV格式: 便于Excel分析
- JSON格式: 便于程序处理
- LaTeX格式: 直接用于论文表格
- 图表: PNG/PDF格式，用于论文插图

**预期时间**: 1-2小时

---

## 🔍 验证清单

### 环境准备
- [ ] Python 3.8+ 已安装
- [ ] CUDA环境配置正确（如果使用GPU）
- [ ] 依赖包已安装
- [ ] 环境检查通过

### 模型下载
- [ ] Qwen2.5-3B-Instruct 已下载并验证
- [ ] 模型完整性检查通过

### Baseline测试
- [ ] `test_baselines.py` 所有测试通过
- [ ] H2O baseline测试成功
- [ ] StreamingLLM baseline测试成功

### A级论文实验
- [ ] Phase 1: 长序列内存效率对比完成
- [ ] Phase 2: 预算影响分析完成
- [ ] Phase 3: 消融实验完成
- [ ] Phase 4: 结果分析和可视化完成

### 结果验证
- [ ] 所有方法的实验结果文件存在
- [ ] 内存使用数据合理
- [ ] BLEU分数正常
- [ ] 可视化图表生成成功

---

## 🚀 快速开始（最小流程）

如果时间有限，可以只运行核心实验：

```bash
# 1. 环境准备
python check_environment.py
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 2. 下载模型（至少Qwen）
bash setup_models_china.sh
# 或
python download_models_python38.py --model Qwen2.5-3B-Instruct

# 3. 验证模型
python check_model_integrity.py ./models/Qwen2.5-3B-Instruct

# 4. 测试baseline
python test_baselines.py

# 5. 运行完整实验
bash run_a_level_experiments.sh

# 6. 分析结果
python analyze_experiment_results.py \
    --output_dir ./output_logs/a_level_paper/long_seq_10000 \
    --detailed \
    --save_csv ./output_logs/summary.csv
```

---

## 📊 实验配置说明

### 评估脚本参数

**主要参数**:
- `--LLM_backbone`: 模型架构 (Qwen/Llama/Gemma)
- `--LLM_path`: 模型路径
- `--use_h2o`: 使用H2O baseline
- `--use_streamingllm`: 使用StreamingLLM baseline
- `--use_head_aware`: 使用Head-Aware方法
- `--use_group_aware`: 使用Group-Aware方法
- `--total_budget`: KV cache预算（tokens/layer）
- `--max_samples`: 最大样本数
- `--quantization`: 量化策略 (4bit/8bit/none)

**示例命令**:
```bash
# H2O baseline
python StreamingLLM_GPE/evaluate/multi_model_eval.py \
    --LLM_backbone Qwen \
    --LLM_path ./models/Qwen2.5-3B-Instruct \
    --use_h2o \
    --h2o_budget 2048 \
    --output_dir ./output_logs/h2o \
    --max_samples 100 \
    --quantization 4bit

# StreamingLLM baseline
python StreamingLLM_GPE/evaluate/multi_model_eval.py \
    --LLM_backbone Qwen \
    --LLM_path ./models/Qwen2.5-3B-Instruct \
    --use_streamingllm \
    --streamingllm_window 512 \
    --output_dir ./output_logs/streamingllm \
    --max_samples 100 \
    --quantization 4bit

# Head-Aware方法
python StreamingLLM_GPE/evaluate/multi_model_eval.py \
    --LLM_backbone Qwen \
    --LLM_path ./models/Qwen2.5-3B-Instruct \
    --use_head_aware \
    --total_budget 2048 \
    --output_dir ./output_logs/head_aware \
    --max_samples 100 \
    --quantization 4bit

# Full方法（Head-Aware + Group-Aware）
python StreamingLLM_GPE/evaluate/multi_model_eval.py \
    --LLM_backbone Qwen \
    --LLM_path ./models/Qwen2.5-3B-Instruct \
    --use_head_aware \
    --use_group_aware \
    --total_budget 2048 \
    --output_dir ./output_logs/full \
    --max_samples 100 \
    --quantization 4bit
```

---

## ⚠️ 常见问题

### 问题1: 模型下载失败

**解决方案**:
- 检查网络连接
- 使用ModelScope镜像（如果Python 3.9+）
- 使用HuggingFace Token（对于Llama3）
- 参考 `download_models_python38.py` 的说明

### 问题2: 显存不足

**解决方案**:
- 使用4bit量化: `--quantization 4bit`
- 减少样本数: `--max_samples 50`
- 减少预算: `--total_budget 1024`

### 问题3: Baseline测试失败

**解决方案**:
- 检查baseline文件是否存在: `ls -la StreamingLLM_GPE/baselines/`
- 检查导入: `python -c "from StreamingLLM_GPE.baselines import H2OCache, StreamingLLMCache"`
- 查看错误日志

### 问题4: 实验结果异常

**解决方案**:
- 检查模型是否正确加载
- 检查数据文件是否存在
- 查看日志文件: `./output_logs/{方法名}/multi_model_eval.log`

---

## 📝 总结

**核心实验流程**:
1. 环境准备 → 2. 下载模型 → 3. 测试Baseline → 4. 运行完整实验 → 5. 分析结果

**关键文件**:
- `run_a_level_experiments.sh` - 主实验脚本
- `StreamingLLM_GPE/evaluate/multi_model_eval.py` - 评估脚本
- `test_baselines.py` - Baseline测试

**预期时间**:
- 环境准备: 10分钟
- 模型下载: 2-4小时
- Baseline测试: 20分钟
- 完整实验: 4-8小时
- 结果分析: 1-2小时

**总计**: 约1-2天（取决于硬件和网络）

