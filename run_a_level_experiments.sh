#!/bin/bash
# A级论文完整实验脚本
# 针对Ubuntu系统和24GB显存优化

set -e  # 遇到错误立即退出

# 配置
MODEL_NAME="Qwen"
MODEL_PATH="./models/Qwen2.5-3B-Instruct"
BASE_OUTPUT_DIR="./output_logs/a_level_paper"
PARAMS="./StreamingLLM_GPE/configs/params_qwen_inference.json"
WAIT_K=5
TOTAL_BUDGET=512   # 更小预算以触发压缩
MAX_MEMORY_GB=20.0  # 24GB显存，留4GB缓冲

# 实验配置
MAX_SAMPLES=50   # 控制运行时间
MIN_SOURCE_LENGTH=20

# 长序列测试配置
LONG_SEQUENCE_LENGTHS=(2000 5000 10000 20000)
BUDGETS=(512 1024 2048)

echo "========================================="
echo "A-Level Paper Experiment Suite"
echo "========================================="
echo "GPU: RTX 4090 (24GB)"
echo "Model: $MODEL_NAME"
echo "Max Samples: $MAX_SAMPLES"
echo "Output: $BASE_OUTPUT_DIR"
echo ""

# 创建输出目录
mkdir -p $BASE_OUTPUT_DIR

# 检查模型是否存在
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at: $MODEL_PATH"
    exit 1
fi

# ============================================
# Phase 1: 长序列内存效率对比实验
# ============================================
echo ""
echo "========================================="
echo "Phase 1: Long Sequence Memory Efficiency"
echo "========================================="
echo ""

for seq_len in "${LONG_SEQUENCE_LENGTHS[@]}"; do
    echo "----------------------------------------"
    echo "Testing sequence length: $seq_len tokens"
    echo "----------------------------------------"
    
    # Calculate min_source_length more reasonably
    min_length=$((seq_len / 100))
    if [ $min_length -lt 10 ]; then
        min_length=10
    fi
    if [ $min_length -gt 50 ]; then
        min_length=50
    fi
    
    # 1. Baseline (GPE)
    echo "[1/5] Running Baseline (GPE)..."
    python StreamingLLM_GPE/evaluate/multi_model_eval.py \
        --LLM_backbone $MODEL_NAME \
        --LLM_path $MODEL_PATH \
        --inference_mode streaming \
        --wait_k $WAIT_K \
        --output_dir "$BASE_OUTPUT_DIR/long_seq_${seq_len}/baseline" \
        --params $PARAMS \
        --min_source_length $min_length \
        --max_samples $MAX_SAMPLES \
        --max_new_tokens 256 || echo "Warning: Baseline failed for length $seq_len"
    
    # 2. H2O Baseline (if implemented)
    if [ -f "StreamingLLM_GPE/baselines/h2o_cache.py" ]; then
        echo "[2/5] Running H2O Baseline..."
        python StreamingLLM_GPE/evaluate/multi_model_eval.py \
            --LLM_backbone $MODEL_NAME \
            --LLM_path $MODEL_PATH \
            --inference_mode streaming \
            --wait_k $WAIT_K \
            --use_h2o \
            --total_budget $TOTAL_BUDGET \
            --max_memory_gb $MAX_MEMORY_GB \
            --output_dir "$BASE_OUTPUT_DIR/long_seq_${seq_len}/h2o" \
            --params $PARAMS \
            --min_source_length $min_length \
            --max_samples $MAX_SAMPLES \
            --max_new_tokens 256 || echo "Warning: H2O failed for length $seq_len"
    else
        echo "[2/5] Skipping H2O (not implemented yet)"
    fi
    
    # 3. StreamingLLM Baseline (if implemented)
    if [ -f "StreamingLLM_GPE/baselines/streamingllm_cache.py" ]; then
        echo "[3/5] Running StreamingLLM Baseline..."
        python StreamingLLM_GPE/evaluate/multi_model_eval.py \
            --LLM_backbone $MODEL_NAME \
            --LLM_path $MODEL_PATH \
            --inference_mode streaming \
            --wait_k $WAIT_K \
            --use_streamingllm \
            --total_budget $TOTAL_BUDGET \
            --max_memory_gb $MAX_MEMORY_GB \
            --output_dir "$BASE_OUTPUT_DIR/long_seq_${seq_len}/streamingllm" \
            --params $PARAMS \
            --min_source_length $min_length \
            --max_samples $MAX_SAMPLES \
            --max_new_tokens 256 || echo "Warning: StreamingLLM failed for length $seq_len"
    else
        echo "[3/5] Skipping StreamingLLM (not implemented yet)"
    fi
    
    # 4. Head-Aware
    echo "[4/5] Running Head-Aware..."
    python StreamingLLM_GPE/evaluate/multi_model_eval.py \
        --LLM_backbone $MODEL_NAME \
        --LLM_path $MODEL_PATH \
        --inference_mode streaming \
        --wait_k $WAIT_K \
        --use_head_aware \
        --total_budget $TOTAL_BUDGET \
        --max_memory_gb $MAX_MEMORY_GB \
        --output_dir "$BASE_OUTPUT_DIR/long_seq_${seq_len}/head_aware" \
        --params $PARAMS \
        --min_source_length $min_length \
        --max_samples $MAX_SAMPLES \
        --max_new_tokens 256 || echo "Warning: Head-Aware failed for length $seq_len"
    
    # 5. Full (Head-Aware + Group-Aware)
    echo "[5/5] Running Full (Head-Aware + Group-Aware)..."
    python StreamingLLM_GPE/evaluate/multi_model_eval.py \
        --LLM_backbone $MODEL_NAME \
        --LLM_path $MODEL_PATH \
        --inference_mode streaming \
        --wait_k $WAIT_K \
        --use_head_aware \
        --use_group_aware \
        --total_budget $TOTAL_BUDGET \
        --max_memory_gb $MAX_MEMORY_GB \
        --output_dir "$BASE_OUTPUT_DIR/long_seq_${seq_len}/full" \
        --params $PARAMS \
        --min_source_length $min_length \
        --max_samples $MAX_SAMPLES \
        --max_new_tokens 256 || echo "Warning: Full failed for length $seq_len"
    
    echo ""
done

# ============================================
# Phase 2: 不同预算的影响
# ============================================
echo ""
echo "========================================="
echo "Phase 2: Budget Impact Analysis"
echo "========================================="
echo ""

for budget in "${BUDGETS[@]}"; do
    echo "----------------------------------------"
    echo "Testing budget: $budget tokens/layer"
    echo "----------------------------------------"
    
    python StreamingLLM_GPE/evaluate/multi_model_eval.py \
        --LLM_backbone $MODEL_NAME \
        --LLM_path $MODEL_PATH \
        --inference_mode streaming \
        --wait_k $WAIT_K \
        --use_head_aware \
        --use_group_aware \
        --total_budget $budget \
        --max_memory_gb $MAX_MEMORY_GB \
        --output_dir "$BASE_OUTPUT_DIR/budget_${budget}" \
        --params $PARAMS \
        --min_source_length $MIN_SOURCE_LENGTH \
        --max_samples $MAX_SAMPLES \
        --max_new_tokens 256 || echo "Warning: Budget $budget failed"
    
    echo ""
done

# ============================================
# Phase 3: 消融实验
# ============================================
echo ""
echo "========================================="
echo "Phase 3: Ablation Study"
echo "========================================="
echo ""

# 使用中等长度序列进行消融实验
ABLATION_SEQ_LEN=5000

echo "Running ablation experiments with sequence length: $ABLATION_SEQ_LEN"
echo ""

    # Calculate min_source_length
    min_length=$((ABLATION_SEQ_LEN / 100))
    if [ $min_length -lt 10 ]; then
        min_length=10
    fi
    if [ $min_length -gt 50 ]; then
        min_length=50
    fi

    # 1. Baseline (GPE only)
    echo "1. Baseline (GPE only)..."
    python StreamingLLM_GPE/evaluate/multi_model_eval.py \
        --LLM_backbone $MODEL_NAME \
        --LLM_path $MODEL_PATH \
        --inference_mode streaming \
        --wait_k $WAIT_K \
        --output_dir "$BASE_OUTPUT_DIR/ablation/baseline" \
        --params $PARAMS \
        --min_source_length $min_length \
        --max_samples $MAX_SAMPLES \
        --max_new_tokens 256 || echo "Warning: Baseline ablation failed"

    # 2. Head-Aware only
    echo "2. Head-Aware only..."
    python StreamingLLM_GPE/evaluate/multi_model_eval.py \
        --LLM_backbone $MODEL_NAME \
        --LLM_path $MODEL_PATH \
        --inference_mode streaming \
        --wait_k $WAIT_K \
        --use_head_aware \
        --total_budget $TOTAL_BUDGET \
        --max_memory_gb $MAX_MEMORY_GB \
        --output_dir "$BASE_OUTPUT_DIR/ablation/head_aware" \
        --params $PARAMS \
        --min_source_length $min_length \
        --max_samples $MAX_SAMPLES \
        --max_new_tokens 256 || echo "Warning: Head-Aware ablation failed"

    # 3. Group-Aware only
    echo "3. Group-Aware only..."
    python StreamingLLM_GPE/evaluate/multi_model_eval.py \
        --LLM_backbone $MODEL_NAME \
        --LLM_path $MODEL_PATH \
        --inference_mode streaming \
        --wait_k $WAIT_K \
        --use_group_aware \
        --total_budget $TOTAL_BUDGET \
        --max_memory_gb $MAX_MEMORY_GB \
        --output_dir "$BASE_OUTPUT_DIR/ablation/group_aware" \
        --params $PARAMS \
        --min_source_length $min_length \
        --max_samples $MAX_SAMPLES \
        --max_new_tokens 256 || echo "Warning: Group-Aware ablation failed"

    # 4. Full
    echo "4. Full (Head-Aware + Group-Aware)..."
    python StreamingLLM_GPE/evaluate/multi_model_eval.py \
        --LLM_backbone $MODEL_NAME \
        --LLM_path $MODEL_PATH \
        --inference_mode streaming \
        --wait_k $WAIT_K \
        --use_head_aware \
        --use_group_aware \
        --total_budget $TOTAL_BUDGET \
        --max_memory_gb $MAX_MEMORY_GB \
        --output_dir "$BASE_OUTPUT_DIR/ablation/full" \
        --params $PARAMS \
        --min_source_length $min_length \
        --max_samples $MAX_SAMPLES \
        --max_new_tokens 256 || echo "Warning: Full ablation failed"

# ============================================
# Phase 4: 结果分析和汇总
# ============================================
echo ""
echo "========================================="
echo "Phase 4: Results Analysis"
echo "========================================="
echo ""

# 分析长序列实验结果
echo "Analyzing long sequence results..."
python analyze_experiment_results.py \
    --output_dir "$BASE_OUTPUT_DIR/long_seq_10000" \
    --detailed \
    --save_csv "$BASE_OUTPUT_DIR/long_seq_10000_summary.csv" \
    --save_json "$BASE_OUTPUT_DIR/long_seq_10000_summary.json" \
    --save_latex "$BASE_OUTPUT_DIR/long_seq_10000_table.tex" || echo "Warning: Analysis failed"

# 分析消融实验结果
echo "Analyzing ablation results..."
python analyze_experiment_results.py \
    --output_dir "$BASE_OUTPUT_DIR/ablation" \
    --detailed \
    --save_csv "$BASE_OUTPUT_DIR/ablation_summary.csv" \
    --save_json "$BASE_OUTPUT_DIR/ablation_summary.json" \
    --save_latex "$BASE_OUTPUT_DIR/ablation_table.tex" || echo "Warning: Ablation analysis failed"

# 生成可视化
echo "Generating visualizations..."
python visualize_results.py \
    --results_dir "$BASE_OUTPUT_DIR/long_seq_10000" \
    --output_dir "$BASE_OUTPUT_DIR/figures" \
    --include_budget || echo "Warning: Visualization failed"

echo ""
echo "========================================="
echo "All A-Level Experiments Completed!"
echo "========================================="
echo ""
echo "Results saved to: $BASE_OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "1. Review results in: $BASE_OUTPUT_DIR"
echo "2. Implement H2O and StreamingLLM baselines"
echo "3. Run multi-model experiments (Llama, Gemma)"
echo "4. Run multi-task experiments (SQuAD, WikiText)"
echo ""

