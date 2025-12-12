"""
多模型评估脚本
支持: Qwen, Llama, Gemma, Phi3
用于A级会议/期刊的多模型验证
"""
import os
import sys

os.environ["USE_TF"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# ============================================

# 禁用HuggingFace数据集缓存，确保使用最新的数据文件
os.environ["HF_DATASETS_DISABLE_CACHE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["NO_TF"] = "1"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from transformers import AutoTokenizer, AutoConfig
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator
import peft
from argparse import ArgumentParser
import logging
from transformers import BitsAndBytesConfig
import json
import time

from StreamingLLM_GPE.dataloader_hf import StreamingDataCollator
from StreamingLLM_GPE.models.Qwen2_5.qwen_streaming import Qwen2ForCausalLM_stream
from StreamingLLM_GPE.models.Llama3.llama_streaming import LlamaForCausalLM_stream
from StreamingLLM_GPE.models.Gemma2.gemma2_streaming import Gemma2ForCausalLM_stream
from StreamingLLM_GPE.models.Qwen2_5.head_aware_cache import HeadAwareDynamicCache as QwenHeadAwareCache
from StreamingLLM_GPE.models.Llama3.head_aware_cache import HeadAwareDynamicCache as LlamaHeadAwareCache
from StreamingLLM_GPE.models.Gemma2.head_aware_cache import HeadAwareDynamicCache as GemmaHeadAwareCache
from StreamingLLM_GPE.utils.head_analyzer import HeadAnalyzer
from StreamingLLM_GPE.utils.group_tracker import GroupTracker
from StreamingLLM_GPE.utils.budget_monitor import BudgetMonitor
import importlib.util

_utils_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'utils.py')
_spec = importlib.util.spec_from_file_location("StreamingLLM_GPE.utils_module", _utils_file_path)
utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(utils)
from StreamingLLM_GPE.evaluate.lagging import calculate_al_and_laal
import sacrebleu

# 模型类映射
MODEL_CLASSES = {
    'Qwen': Qwen2ForCausalLM_stream,
    'Llama': LlamaForCausalLM_stream,
    'Gemma': Gemma2ForCausalLM_stream,
}


class MemoryMonitor:
    """监控GPU内存使用"""

    def __init__(self, device=0):
        self.device = device
        self.peak_memory = 0
        self.memory_history = []

    def record(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            # 使用 max_memory_allocated 获取峰值
            current_peak = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)  # GB
            current_alloc = torch.cuda.memory_allocated(self.device) / (1024 ** 3)

            self.memory_history.append(current_alloc)
            self.peak_memory = max(self.peak_memory, current_peak)

            return current_alloc
        return 0

    def get_stats(self):
        return {
            'peak_memory_gb': self.peak_memory,
            'avg_memory_gb': sum(self.memory_history) / len(self.memory_history) if self.memory_history else 0,
            'final_memory_gb': self.memory_history[-1] if self.memory_history else 0
        }


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--pe_cache_length", type=float, default=0)
    parser.add_argument("--inference_mode", type=str, default="streaming",
                        choices=["batch", "streaming"])
    parser.add_argument("--LLM_backbone", type=str, default="Qwen",
                        choices=list(MODEL_CLASSES.keys()), help="Model architecture")
    parser.add_argument("--LLM_path", type=str, required=True, help="Path to the LLM model.")
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--wait_k", type=int, default=5)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default='./output_logs')
    parser.add_argument("--split_mode", type=str, default="word",
                        choices=["word", "token"])
    parser.add_argument("--params", type=str, default="./StreamingLLM_GPE/configs/params_qwen_inference.json")

    # Head-Aware specific arguments
    parser.add_argument("--use_head_aware", action="store_true", help="Use Head-Aware Cache")
    parser.add_argument("--use_group_aware", action="store_true", help="Use Group-Aware Eviction")
    parser.add_argument("--total_budget", type=int, default=2048, help="KV cache budget per layer")
    parser.add_argument("--max_memory_gb", type=float, default=4.0, help="Max KV cache memory (GB)")
    parser.add_argument("--analyze_heads", action="store_true", help="Analyze head functionality")

    # Baseline methods
    parser.add_argument("--use_h2o", action="store_true", help="Use H2O baseline")
    parser.add_argument("--use_streamingllm", action="store_true", help="Use StreamingLLM baseline")
    parser.add_argument("--h2o_budget", type=int, default=2048, help="H2O budget per layer")
    parser.add_argument("--streamingllm_window", type=int, default=512, help="StreamingLLM window size")

    # Multi-model evaluation arguments
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process")

    # 强制测试长序列 (默认3000)
    parser.add_argument("--min_source_length", type=int, default=3000, help="Minimum source length in words")

    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum number of new tokens to generate")

    # Quantization options
    parser.add_argument("--quantization", type=str, default="4bit",
                        choices=["4bit", "8bit", "none"],
                        help="Quantization strategy: 4bit (default), 8bit (better performance), none (best performance)")

    return parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_model_class(model_name):
    """根据模型名称返回模型类"""
    if model_name not in MODEL_CLASSES:
        raise ValueError(f"Unsupported model: {model_name}. Supported models: {list(MODEL_CLASSES.keys())}")
    return MODEL_CLASSES[model_name]


def initialize_head_aware_components(model, config, args):
    """初始化Head-Aware相关组件"""
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

    head_analyzer = HeadAnalyzer(num_layers, num_heads, device=device)
    group_tracker = GroupTracker(sink_groups=2) if args.use_group_aware else None
    budget_monitor = BudgetMonitor(max_memory_gb=args.max_memory_gb) if args.use_head_aware else None

    return head_analyzer, group_tracker, budget_monitor


def create_cache(head_analyzer, group_tracker, args, model_name='Qwen'):
    """创建KV cache"""
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

    # ================= [关键修复 1] 增加 Sink Tokens =================
    # 将 sink_tokens 从 4 增加到 128
    # 这是为了防止在处理长文本时，H2O 为了节省空间把开头的 System Prompt（翻译指令）删掉
    # 删掉指令后，模型就会忘记自己是在做翻译，变成做阅读理解

    if args.use_h2o:
        try:
            from StreamingLLM_GPE.baselines.h2o_cache import H2OCache
            return H2OCache(
                budget_per_layer=args.h2o_budget,
                sink_tokens=128,  # [FIXED] 4 -> 128
                device=device
            )
        except ImportError:
            raise ImportError("H2O baseline not implemented. See BASELINE_IMPLEMENTATION_GUIDE.md")

    if args.use_streamingllm:
        try:
            from StreamingLLM_GPE.baselines.streamingllm_cache import StreamingLLMCache
            return StreamingLLMCache(
                window_size=args.streamingllm_window,
                sink_tokens=128,
                device=device
            )
        except ImportError:
            raise ImportError("StreamingLLM baseline not implemented. See BASELINE_IMPLEMENTATION_GUIDE.md")
    # ================================================================

    if args.use_head_aware:
        # 根据模型类型选择对应的HeadAwareCache
        if model_name == 'Qwen':
            cache = QwenHeadAwareCache(
                head_analyzer=head_analyzer,
                group_tracker=group_tracker,
                total_budget=args.total_budget,
                sink_tokens=128,
                adaptive=True,
                device=device
            )
        elif model_name == 'Llama':
            cache = LlamaHeadAwareCache(
                head_analyzer=head_analyzer,
                group_tracker=group_tracker,
                total_budget=args.total_budget,
                sink_tokens=128,
                adaptive=True,
                device=device
            )
        elif model_name == 'Gemma':
            cache = GemmaHeadAwareCache(
                head_analyzer=head_analyzer,
                group_tracker=group_tracker,
                total_budget=args.total_budget,
                sink_tokens=128,
                adaptive=True,
                device=device
            )
        else:
            # 默认使用Qwen的实现
            cache = QwenHeadAwareCache(
                head_analyzer=head_analyzer,
                group_tracker=group_tracker,
                total_budget=args.total_budget,
                sink_tokens=128,
                adaptive=True,
                device=device
            )
        return cache
    else:
        # 根据模型类型选择DynamicCache
        if model_name == 'Qwen':
            from StreamingLLM_GPE.models.Qwen2_5.qwen_streaming import DynamicCache
            return DynamicCache()
        elif model_name == 'Llama':
            from StreamingLLM_GPE.models.Llama3.llama_streaming import DynamicCache
            return DynamicCache()
        elif model_name == 'Gemma':
            from StreamingLLM_GPE.models.Gemma2.gemma2_streaming import DynamicCache
            return DynamicCache()
        else:
            from StreamingLLM_GPE.models.Qwen2_5.qwen_streaming import DynamicCache
            return DynamicCache()


def main():
    args = get_args()
    params = utils.Params(args.params)
    setup_seed(0)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 设置日志
    log_file = f"{args.output_dir}/multi_model_eval.log"
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # 记录配置
    # 确定显示的预算值
    if args.use_h2o:
        budget_display = f"{args.h2o_budget} tokens/layer (H2O)"
    elif args.use_streamingllm:
        budget_display = f"{args.streamingllm_window} tokens (StreamingLLM window)"
    elif args.use_head_aware:
        budget_display = f"{args.total_budget} tokens/layer (Head-Aware)"
    else:
        budget_display = f"{args.total_budget} tokens/layer (default)"

    config_str = f"""
    Multi-Model Evaluation Configuration:
    - Model Architecture: {args.LLM_backbone}
    - Model Path: {args.LLM_path}
    - Inference Mode: {args.inference_mode}
    - Head-Aware: {args.use_head_aware}
    - Group-Aware: {args.use_group_aware}
    - H2O: {args.use_h2o}
    - StreamingLLM: {args.use_streamingllm}
    - Cache Budget: {budget_display}
    - Max Memory: {args.max_memory_gb} GB
    - Wait-k: {args.wait_k}
    - Min Source Length: {args.min_source_length}
    - Max Samples: {args.max_samples}
    - Max New Tokens: {args.max_new_tokens}
    """
    print(config_str)
    logging.info(config_str)

    # 获取模型类
    ModelClass = get_model_class(args.LLM_backbone)

    # 加载模型
    config = AutoConfig.from_pretrained(args.LLM_path)
    config._attn_implementation = "eager"
    tokenizer = AutoTokenizer.from_pretrained(args.LLM_path, padding_side='right', config=config)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 量化配置（根据参数选择）
    quantization_config = None
    torch_dtype = torch.bfloat16

    if args.quantization == "4bit":
        # 4-bit量化：最小显存占用
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    elif args.quantization == "8bit":
        # 8-bit量化：平衡性能和显存
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False
        )
    elif args.quantization == "none":
        # 无量化：最佳性能，需要更多显存
        quantization_config = None
        torch_dtype = torch.bfloat16

    # 加载模型
    model_kwargs = {
        "ignore_mismatched_sizes": True,
        "config": config,
        "device_map": "auto"
    }

    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    else:
        model_kwargs["torch_dtype"] = torch_dtype

    model = ModelClass.from_pretrained(
        args.LLM_path,
        **model_kwargs
    )

    if args.lora_path is not None:
        model = peft.PeftModel.from_pretrained(model, args.lora_path)

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.eos_token_id

    # 初始化Head-Aware组件
    head_analyzer, group_tracker, budget_monitor = initialize_head_aware_components(
        model, config, args
    )

    # 如果启用Head-Aware，预分析heads（可选）
    if args.use_head_aware and args.analyze_heads:
        print("Analyzing head functionality...")
        logging.info("Head analysis will be performed during inference")

    # 数据加载
    data_collator = StreamingDataCollator(
        file_path=params.file_path,
        tokenizer=tokenizer,
        Instruct=params.Instruct,
        user_Instruct=params.user_Instruct,
        assistant_Instruct=params.assistant_Instruct,
        end_Instruct=params.end_Instruct,
        source_key=params.source_key,
        target_key=params.target_key,
        inference_mode=args.inference_mode,
        split_mode=args.split_mode,
        if_add_space=params.if_add_space,
        pe_cache_length=args.pe_cache_length,
        wait_k=args.wait_k,
    )

    data_collator_dataset = data_collator.dataset_loader()

    # 记录初始数据集大小
    initial_size = len(data_collator_dataset)
    logging.info(f"Loaded dataset with {initial_size} samples from {params.file_path}")

    if initial_size == 0:
        logging.error(f"No samples found in data file: {params.file_path}")
        raise ValueError(
            f"No samples available in data file: {params.file_path}. Please check the file path and format.")

    # 过滤短序列（如果指定了最小长度）
    if args.min_source_length > 0:
        def filter_long_sequences(example):
            source_txt = example.get("source_txt", "")
            if not source_txt:
                return False
            source_words = source_txt.split()
            return len(source_words) >= args.min_source_length

        before_filter_size = len(data_collator_dataset)
        data_collator_dataset = data_collator_dataset.filter(filter_long_sequences)
        after_filter_size = len(data_collator_dataset)
        logging.info(
            f"Filtered dataset: {before_filter_size} -> {after_filter_size} samples (keeping sequences with >= {args.min_source_length} source words)")

        if after_filter_size == 0:
            # 提供更详细的诊断信息
            logging.warning(f"All samples were filtered out! Checking sample lengths...")
            # 检查前几个样本的长度
            sample_lengths = []
            for i, example in enumerate(data_collator.dataset_loader()):
                if i >= 5:  # 只检查前5个样本
                    break
                source_txt = example.get("source_txt", "")
                if source_txt:
                    word_count = len(source_txt.split())
                    sample_lengths.append(word_count)

            if sample_lengths:
                max_length = max(sample_lengths)
                avg_length = sum(sample_lengths) / len(sample_lengths)
                logging.error(f"Sample word counts (first {len(sample_lengths)}): {sample_lengths}")
                logging.error(f"Max length: {max_length}, Average length: {avg_length:.1f}")
                logging.error(f"Required minimum: {args.min_source_length}")
                logging.error(
                    f"Try reducing --min_source_length (e.g., --min_source_length {max(100, int(avg_length))})")
            else:
                logging.error("Could not read sample lengths. Check data file format.")

            raise ValueError(
                f"No samples available after filtering with min_source_length={args.min_source_length}. "
                f"Original dataset size: {before_filter_size}. "
                f"Try reducing --min_source_length or check your data file."
            )

    # 限制样本数量（用于测试）
    original_size = len(data_collator_dataset)
    if args.max_samples is not None and args.max_samples > 0:
        data_collator_dataset = data_collator_dataset.select(range(min(args.max_samples, len(data_collator_dataset))))
        logging.info(f"Limited dataset to {len(data_collator_dataset)} samples (from {original_size})")

    if len(data_collator_dataset) == 0:
        logging.error("No samples in dataset after filtering!")
        raise ValueError("No samples available for evaluation. Check data file and filtering criteria.")

    dataloader = DataLoader(
        data_collator_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=data_collator.collate_fn_inference
    )

    # 设置环境变量避免tensorflow和tokenizers冲突
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 禁用tensorflow警告

    # 当使用量化模型时，模型已经通过device_map="auto"分配到设备
    # 需要禁用accelerate的设备管理以避免冲突
    use_accelerate = False
    # accelerator = None

    if quantization_config is not None:
        # 量化模型已经通过device_map="auto"分配到设备，不使用accelerate
        stream_model = model
        # 获取模型所在的设备
        try:
            device = next(model.parameters()).device
        except:
            # 如果无法获取，使用默认设备
            device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu")
    else:
        # 非量化模型使用accelerate
        accelerator = Accelerator(mixed_precision="bf16")
        stream_model, dataloader = accelerator.prepare(model, dataloader)
        device = accelerator.device
        use_accelerate = True

    # 评估指标
    target_txt_lt = []
    output_text_lt = []
    AL = []
    LAAL = []

    # 内存监控
    memory_monitor = MemoryMonitor(device=args.device)

    # 统计信息
    stats = {
        'total_tokens': 0,
        'max_length': 0,
        'cache_memory_gb': [],
        'inference_times': []
    }

    stream_model.eval()

    # 循环评估
    for step, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {args.LLM_backbone}")):
        # 创建cache（在generate之前设置）
        if args.use_head_aware:
            cache = create_cache(head_analyzer, group_tracker, args, args.LLM_backbone)
            # 设置source和target cache
            stream_model.source_key_values = cache
            stream_model.target_key_values = create_cache(head_analyzer, group_tracker, args, args.LLM_backbone)
            stream_model.past_key_values = create_cache(head_analyzer, group_tracker, args, args.LLM_backbone)
        else:
            # 根据模型类型创建DynamicCache
            stream_model.source_key_values = create_cache(head_analyzer, group_tracker, args, args.LLM_backbone)
            stream_model.target_key_values = create_cache(head_analyzer, group_tracker, args, args.LLM_backbone)
            stream_model.past_key_values = create_cache(head_analyzer, group_tracker, args, args.LLM_backbone)

        # 获取原始文本
        source_txt = batch.get("source_txt", None)
        target_txt = batch.get("target_txt", None)

        # ================= [关键修复: 手动构建 Prompt 以保持流式对齐] =================
        if "Instruct" in args.LLM_path or "Chat" in args.LLM_path:
            # 1. 获取原始的 input_ids (对应 source_seg_len)
            original_input_ids = batch.get("source_tokens").to(device)

            # 2. 定义 System Prompt 和 强制翻译指令
            system_content = params.Instruct if params.Instruct else "You are a helpful assistant."
            # 在 User 输入后加强指令，防止模型遗忘
            user_suffix_text = "\n\nImportant: Please translate the above text into French immediately."

            # 3. 手动构建前缀和后缀 (适配 Qwen/ChatML 格式)
            # 注意: 如果换用 Llama-3，需要改为 <|begin_of_text|>...<|start_header_id|>...
            prefix_str = f"<|im_start|>system\n{system_content}<|im_end|>\n<|im_start|>user\n"
            suffix_str = f"{user_suffix_text}<|im_end|>\n<|im_start|>assistant\n"

            # 4. Tokenize 前缀和后缀
            prefix_tokens = tokenizer(prefix_str, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
            suffix_tokens = tokenizer(suffix_str, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

            # 5. 拼接: [Prefix] + [Original Source] + [Suffix]
            # 这样中间的 [Original Source] 依然保持原样，不会破坏 source_seg_len 的对应关系
            new_input_ids = torch.cat([prefix_tokens, original_input_ids, suffix_tokens], dim=1)

            # 6. 更新 batch
            batch["source_tokens"] = new_input_ids
            batch["attention_mask"] = torch.ones_like(new_input_ids)

            # 7. [最关键的一步] 更新 source_seg_len
            # 将前缀长度加到第一个词上，后缀长度加到最后一个词上
            if "_lengths" in batch:
                # 获取源文本分段长度列表
                seg_lens = batch["_lengths"][0]["source_seg_len"]

                # 修改第一个词的长度：模型读第一个词时，会连同 System Prompt 一起读入
                seg_lens[0] += prefix_tokens.shape[1]

                # 修改最后一个词的长度：模型读完最后词时，会连同 Assistant Prompt 一起读入
                seg_lens[-1] += suffix_tokens.shape[1]

                # 显式写回
                batch["_lengths"][0]["source_seg_len"] = seg_lens

            # 调试打印 (仅第一次)
            if step == 0:
                print(f"\n[FIX DEBUG] Chat Template Applied Correctly:")
                print(f"  Prefix Length: {prefix_tokens.shape[1]}")
                print(f"  Original Length: {original_input_ids.shape[1]}")
                print(f"  Suffix Length: {suffix_tokens.shape[1]}")
                print(f"  New Total Length: {new_input_ids.shape[1]}")
                print(f"  Updated source_seg_len[0]: {batch['_lengths'][0]['source_seg_len'][0]}")

            # 更新 input_ids 变量供后续使用
            input_ids = new_input_ids
            attention_mask = batch["attention_mask"]
        else:
            # 非Chat模型，保持原样
            input_ids = batch.get("source_tokens", None)
            attention_mask = batch.get("attention_mask", None)
        # ==========================================================================

        _lengths = batch.get("_lengths", None)
        inference_mode = batch.get("inference_mode", "streaming")
        split_mode = batch.get("split_mode", None)
        _lengths_index = batch.get("_lengths_index", None)
        wait_k = batch.get("wait_k", None)
        assistant_token = batch.get("assistant_token", None)

        # 记录内存
        memory_monitor.record()

        # 添加调试信息
        if step == 0 or step < 3:  # 只打印前几个样本的详细信息
            source_text_full = source_txt[0] if source_txt and len(source_txt) > 0 else 'N/A'
            source_words = source_text_full.split() if source_text_full != 'N/A' else []
            source_word_count = len(source_words)

            logging.info(f"[DEBUG] Sample {step}:")
            logging.info(
                f"  Source text (first 200 chars): {source_text_full[:200] if source_text_full != 'N/A' else 'N/A'}...")
            logging.info(f"  Source length: {source_word_count} words")

            logging.info(f"  Input IDs shape: {input_ids.shape}")
            logging.info(f"  Max new tokens: {args.max_new_tokens}")

            # 打印到控制台
            print(f"\n[DEBUG] Sample {step}:")
            print(f"  Source length: {source_word_count} words")
            if source_word_count > 1000:
                print(f"  ⚠️ 警告: 源文本异常长（{source_word_count} words）")

        # 推理
        start_time = time.time()

        if inference_mode == "streaming":
            output_sequences, wait_lagging = stream_model.generate(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                max_new_tokens=args.max_new_tokens,
                generate_mode=inference_mode,
                split_mode=split_mode,
                pe_cache_length=args.pe_cache_length,
                tokenizer=tokenizer,
                end_Instruct=params.end_Instruct,
                _lengths=_lengths,
                _lengths_index=_lengths_index,
                wait_k=wait_k,
                source_words=source_txt,
                assistant_token=assistant_token.to(device),
            )
        elif inference_mode == "batch":
            output_sequences, wait_lagging = stream_model.generate(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                max_new_tokens=args.max_new_tokens,
            )

        inference_time = time.time() - start_time
        stats['inference_times'].append(inference_time)

        # Batch 模式下，generate 返回 [Input + Output]
        # 我们必须切掉 Input 部分，否则计算 BLEU 时会因为包含英文原文而导致分数极低（接近0）

        if inference_mode == "batch":
            input_token_len = input_ids.shape[1]
            generated_tokens = output_sequences[0][input_token_len:]
        else:
            # Streaming 模式通常只返回生成的 token
            generated_tokens = output_sequences[0]

        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        # ==============================================================

        target_txt_lt.extend(target_txt)
        output_text_lt.extend([output_text])

        # 添加调试信息（文件 + 控制台）
        if step == 0 or step < 3:  # 只打印前几个样本的详细信息
            logging.info(f"[DEBUG] Sample {step} output:")
            logging.info(f"  Output length: {len(generated_tokens)} tokens")
            logging.info(f"  Output text (first 500 chars): {output_text[:500]}...")
            logging.info(f"  Target text (first 500 chars): {target_txt[0][:500] if target_txt else ''}...")

            print(f"\n[DEBUG] Sample {step} output:")
            print(f"  Output length: {len(generated_tokens)} tokens")
            print(f"  Output text (first 300 chars): {output_text[:300]}...")
            if target_txt:
                print(f"  Target text (first 300 chars): {target_txt[0][:300]}")

        # 记录统计信息
        seq_len = len(generated_tokens)
        stats['total_tokens'] += seq_len
        stats['max_length'] = max(stats['max_length'], seq_len)

        # 记录cache内存
        if args.use_head_aware:
            # 检查是否是HeadAwareCache（可能是Qwen、Llama或Gemma的实现）
            if hasattr(stream_model.source_key_values, 'get_memory_usage'):
                cache_memory = stream_model.source_key_values.get_memory_usage()
                stats['cache_memory_gb'].append(cache_memory)

        # 检查预算（如果启用）
        if args.use_head_aware and budget_monitor is not None:
            # 检查是否是HeadAwareCache（可能是Qwen、Llama或Gemma的实现）
            if hasattr(stream_model.source_key_values, 'get_memory_usage'):
                budget_monitor.check_and_evict(
                    stream_model.source_key_values,
                    group_tracker
                )

        if inference_mode == "streaming":
            # 延迟计算（使用与head_aware_eval.py相同的逻辑）
            source_txt_words = source_txt[0].split() if source_txt and len(source_txt) > 0 else []
            output_text_words = output_text.split()

            source_length = len(source_txt_words)
            target_length = len(output_text_words)

            if wait_lagging is not None and len(wait_lagging) > 0:
                wait_lagging_valid = wait_lagging
                target_length_for_calc = len(wait_lagging)

                if wait_lagging_valid and len(wait_lagging_valid) > 0:
                    max_delay = max(wait_lagging_valid) if wait_lagging_valid else 0
                    min_delay = min(wait_lagging_valid) if wait_lagging_valid else 0

                    if min_delay < 0 or max_delay > source_length * 2:
                        wait_lagging_valid = [max(0, min(int(d), source_length)) for d in wait_lagging_valid]

                    wait_lagging_valid = [int(max(0, d)) for d in wait_lagging_valid]

                    try:
                        al, laal = calculate_al_and_laal(
                            source_length,
                            target_length_for_calc,
                            wait_lagging_valid
                        )
                        al = max(0.0, al)
                        laal = max(0.0, laal)
                        AL.append(al)
                        LAAL.append(laal)
                    except Exception as e:
                        logging.error(f"Failed to calculate AL/LAAL: {e}")
                        AL.append(0)
                        LAAL.append(0)
                else:
                    AL.append(0)
                    LAAL.append(0)
            else:
                AL.append(0)
                LAAL.append(0)

        # 打印进度（不使用accelerate.is_main_process，因为量化模型可能不使用accelerate）
        if step % 10 == 0 or step == len(dataloader) - 1:
            print(f"\n[{args.LLM_backbone}] Step {step}:")
            print(f"  Output Length: {seq_len} tokens")
            if args.use_head_aware:
                print(f"  Cache Memory: {stats['cache_memory_gb'][-1]:.2f}GB" if stats[
                    'cache_memory_gb'] else "  Cache Memory: 0.00GB")

    # 计算最终指标
    if len(output_text_lt) == 0:
        logging.warning("No output texts generated!")
        bleu_score = 0.0
    else:
        if isinstance(target_txt_lt[0], list):
            target_txt_lt = [item for sublist in target_txt_lt for item in sublist]

        min_len = min(len(output_text_lt), len(target_txt_lt))
        if min_len == 0:
            logging.warning("Empty target or output lists!")
            bleu_score = 0.0
        else:
            output_text_lt = output_text_lt[:min_len]
            target_txt_lt = target_txt_lt[:min_len]

            # 记录前几个样本用于调试
            logging.info(f"\n=== BLEU Calculation Debug ===")
            logging.info(f"Number of samples: {len(output_text_lt)}")
            for i in range(min(3, len(output_text_lt))):
                logging.info(f"\nSample {i}:")
                logging.info(f"  Target: {target_txt_lt[i][:200] if len(target_txt_lt[i]) > 200 else target_txt_lt[i]}")
                logging.info(
                    f"  Output: {output_text_lt[i][:200] if len(output_text_lt[i]) > 200 else output_text_lt[i]}")

            try:
                bleu = sacrebleu.corpus_bleu(output_text_lt, [target_txt_lt])
                bleu_score = bleu.score
                logging.info(f"BLEU score: {bleu_score:.2f}")
            except Exception as e:
                logging.error(f"Failed to calculate BLEU: {e}")
                logging.error(f"Output texts count: {len(output_text_lt)}")
                logging.error(f"Target texts count: {len(target_txt_lt)}")
                bleu_score = 0.0

    # 内存统计
    memory_stats = memory_monitor.get_stats()
    logging.info(f"Peak GPU Memory: {memory_stats['peak_memory_gb']:.2f}GB")
    logging.info(f"Average GPU Memory: {memory_stats['avg_memory_gb']:.2f}GB")

    if args.use_head_aware:
        if stats['cache_memory_gb']:
            avg_cache = sum(stats['cache_memory_gb']) / len(stats['cache_memory_gb'])
            peak_cache = max(stats['cache_memory_gb'])
            logging.info(f"Average Cache Memory: {avg_cache:.4f}GB")
            logging.info(f"Peak Cache Memory: {peak_cache:.4f}GB")

    if inference_mode == "streaming":
        avg_LAAL = sum(LAAL) / len(LAAL) if LAAL else 0
        avg_AL = sum(AL) / len(AL) if AL else 0
        logging.info(f"Average AL: {avg_AL:.2f}")
        logging.info(f"Average LAAL: {avg_LAAL:.2f}")

    # 保存结果
    results = {
        'model_architecture': args.LLM_backbone,
        'model_path': args.LLM_path,
        'bleu_score': bleu_score,
        'memory_stats': memory_stats,
        'cache_stats': {
            'avg_cache_memory_gb': sum(stats['cache_memory_gb']) / len(stats['cache_memory_gb']) if stats[
                'cache_memory_gb'] else 0,
            'peak_cache_memory_gb': max(stats['cache_memory_gb']) if stats['cache_memory_gb'] else 0,
        },
        'length_stats': {
            'total_tokens': stats['total_tokens'],
            'max_length': stats['max_length'],
            'avg_length': stats['total_tokens'] / len(output_text_lt) if output_text_lt else 0,
        },
        'latency_stats': {
            'avg_inference_time': sum(stats['inference_times']) / len(stats['inference_times']) if stats[
                'inference_times'] else 0,
        },
        'streaming_stats': {
            'avg_AL': avg_AL if inference_mode == "streaming" else 0,
            'avg_LAAL': avg_LAAL if inference_mode == "streaming" else 0,
        } if inference_mode == "streaming" else {}
    }

    results_file = f"{args.output_dir}/results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[{args.LLM_backbone}] Results saved to {results_file}")
    print(f"[{args.LLM_backbone}] BLEU Score: {bleu_score:.2f}")
    print(f"[{args.LLM_backbone}] Peak GPU Memory: {memory_stats['peak_memory_gb']:.2f}GB")


if __name__ == "__main__":
    main()