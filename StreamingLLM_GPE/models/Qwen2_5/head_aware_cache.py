"""
Head-Aware Dynamic KV Cache

基于head功能特性的动态KV cache压缩和驱逐
结合Group-aware策略，实现细粒度的内存管理
"""
import torch
from typing import List, Tuple, Optional, Dict
import sys
import os

# Add parent directory to path before importing
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_current_dir, '../../..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Import utils modules first
from StreamingLLM_GPE.utils.head_analyzer import HeadAnalyzer
from StreamingLLM_GPE.utils.group_tracker import GroupTracker

# 优先使用 transformers 的标准 DynamicCache
try:
    from transformers.cache_utils import DynamicCache
except ImportError:
    # 兼容旧版本 transformers
    from transformers.cache_utils import Cache as DynamicCache

class HeadAwareDynamicCache(DynamicCache):
    """
    Head-Aware Dynamic KV Cache

    核心特性：
    1. 基于head功能特性分配预算
    2. 支持Group-level驱逐
    3. 自适应预算调整
    """

    def __init__(
        self,
        head_analyzer: HeadAnalyzer,
        group_tracker: Optional[GroupTracker] = None,
        total_budget: int = 2048,
        sink_tokens: int = 4,
        adaptive: bool = True,
        device: str = 'cuda'
    ):
        """
        Args:
            head_analyzer: Head分析器
            group_tracker: Group跟踪器（可选）
            total_budget: 总预算（tokens per layer）
            sink_tokens: Attention sink tokens数量
            adaptive: 是否使用自适应分配
            device: 设备
        """
        super().__init__()
        self.head_analyzer = head_analyzer
        self.group_tracker = group_tracker
        self.total_budget = total_budget
        self.sink_tokens = sink_tokens
        self.adaptive = adaptive
        self.device = device

        # 初始化 _seen_tokens (如果父类没有初始化)
        if not hasattr(self, "_seen_tokens"):
            self._seen_tokens = 0

        # 存储每个head的独立cache（可选，用于细粒度控制）
        self.head_caches: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        更新KV cache，根据head特性进行压缩

        Args:
            key_states: [batch_size, num_heads, seq_len, head_dim]
            value_states: [batch_size, num_heads, seq_len, head_dim]
            layer_idx: 层索引
            cache_kwargs: 缓存参数 (可能包含 attention_scores)

        Returns:
            (compressed_key_states, compressed_value_states)
        """
        bsz, num_heads, seq_len, head_dim = key_states.shape

        # 维护 _seen_tokens (对于 Position Embedding 至关重要)
        if layer_idx == 0:
            self._seen_tokens += seq_len

        # 如果有现有cache，先合并
        if len(self.key_cache) > layer_idx:
            # 合并新tokens到现有cache
            existing_keys = self.key_cache[layer_idx]
            existing_values = self.value_cache[layer_idx]

            # 拼接
            combined_keys = torch.cat([existing_keys, key_states], dim=2)
            combined_values = torch.cat([existing_values, value_states], dim=2)
        else:
            combined_keys = key_states
            combined_values = value_states

        # 检查是否需要压缩
        current_length = combined_keys.shape[2]

        if current_length <= self.total_budget:
            # 不需要压缩，直接更新
            if len(self.key_cache) <= layer_idx:
                self.key_cache.append(combined_keys)
                self.value_cache.append(combined_values)
            else:
                self.key_cache[layer_idx] = combined_keys
                self.value_cache[layer_idx] = combined_values

            return combined_keys, combined_values

        # 需要压缩：根据head特性进行压缩
        # 尝试从 cache_kwargs 获取 attention_scores
        attention_scores = None
        if cache_kwargs is not None:
            attention_scores = cache_kwargs.get("attention_scores", None)

        compressed_keys, compressed_values = self._compress_by_head(
            combined_keys, combined_values, layer_idx, attention_scores
        )

        # 更新cache
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(compressed_keys)
            self.value_cache.append(compressed_values)
        else:
            self.key_cache[layer_idx] = compressed_keys
            self.value_cache[layer_idx] = compressed_values

        return compressed_keys, compressed_values

    def _compress_by_head(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        attention_scores: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        根据head特性进行压缩
        """
        bsz, num_heads, seq_len, head_dim = key_states.shape

        # 获取每个head的预算
        head_budgets = self.head_analyzer.get_all_head_budgets(
            layer_idx, self.total_budget, self.adaptive
        )

        compressed_keys_list = []
        compressed_values_list = []

        for head_idx in range(num_heads):
            head_key = key_states[:, head_idx, :, :]  # [bsz, seq_len, head_dim]
            head_value = value_states[:, head_idx, :, :]

            # 提取该 head 的 attention scores (如果存在)
            # attention_scores shape usually: [bsz, num_heads, q_len, kv_len]
            # 我们需要聚合 query 维度的分数来得到 importance
            head_attn_score = None
            if attention_scores is not None:
                # [bsz, q_len, kv_len] -> sum over q_len -> [bsz, kv_len]
                # 注意：这里假设 attention_scores 对应当前的 tokens
                # 如果 attention_scores 维度匹配，则提取
                if attention_scores.dim() == 4 and attention_scores.shape[1] == num_heads:
                     # 简单的 sum 聚合，作为重要性近似
                     head_attn_score = attention_scores[:, head_idx, :, :].sum(dim=1) # [bsz, seq_len]

            budget = head_budgets.get(head_idx, self.total_budget // num_heads)
            head_type = self.head_analyzer.head_profiles.get(layer_idx, {}).get(head_idx, 'induction')

            # 根据head类型选择压缩策略
            if head_type == 'retrieval':
                # Retrieval heads: 保留重要性高的tokens
                comp_key, comp_value = self._compress_by_importance(
                    head_key, head_value, budget, head_attn_score
                )
            elif head_type == 'local':
                # Local heads: 只保留最近的tokens
                comp_key, comp_value = self._compress_recent(
                    head_key, head_value, budget
                )
            else:  # induction
                # Induction heads: 保留有模式的关键tokens
                comp_key, comp_value = self._compress_pattern(
                    head_key, head_value, budget
                )

            compressed_keys_list.append(comp_key)
            compressed_values_list.append(comp_value)

        # 重新组合
        compressed_keys = torch.stack(compressed_keys_list, dim=1)  # [bsz, num_heads, budget, head_dim]
        compressed_values = torch.stack(compressed_values_list, dim=1)

        return compressed_keys, compressed_values

    def _compress_by_importance(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        budget: int,
        importance_scores: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        基于重要性压缩（用于retrieval heads）
        """
        bsz, seq_len, head_dim = key.shape

        if seq_len <= budget:
            return key, value

        # 1. 保留sink tokens
        sink_size = min(self.sink_tokens, budget // 2)
        sink_keys = key[:, :sink_size, :]
        sink_values = value[:, :sink_size, :]

        # 2. 准备剩余部分的 keys 和 values
        remaining_keys = key[:, sink_size:, :]
        remaining_values = value[:, sink_size:, :]

        # 3. 计算重要性分数
        if importance_scores is not None:
            # 如果有传入 attention scores，使用它 (切片掉 sink 部分)
            # importance_scores: [bsz, seq_len]
            scores = importance_scores[:, sink_size:]
        else:
            # Fallback: 使用 key 的 L2 norm 作为重要性近似 (Magnitude-based)
            scores = torch.norm(remaining_keys, dim=-1)  # [bsz, seq_len - sink_size]

        # 选择top-k重要的tokens
        remaining_budget = budget - sink_size
        # 确保不超过实际长度
        k = min(remaining_budget, scores.size(1))

        _, top_indices = torch.topk(scores, k, dim=1)

        # 为了保持时序性，通常建议对索引排序 (可选，取决于是否需要保留相对位置)
        top_indices, _ = torch.sort(top_indices, dim=1)

        # 收集重要的tokens
        selected_keys = torch.gather(
            remaining_keys, 1,
            top_indices.unsqueeze(-1).expand(-1, -1, head_dim)
        )
        selected_values = torch.gather(
            remaining_values, 1,
            top_indices.unsqueeze(-1).expand(-1, -1, head_dim)
        )

        # 合并sink和重要的tokens
        compressed_key = torch.cat([sink_keys, selected_keys], dim=1)
        compressed_value = torch.cat([sink_values, selected_values], dim=1)

        return compressed_key, compressed_value

    def _compress_recent(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        budget: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        保留最近的tokens（用于local heads）
        """
        bsz, seq_len, head_dim = key.shape

        if seq_len <= budget:
            return key, value

        # 只保留最近的budget个tokens
        compressed_key = key[:, -budget:, :]
        compressed_value = value[:, -budget:, :]

        return compressed_key, compressed_value

    def _compress_pattern(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        budget: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        保留有模式的关键tokens（用于induction heads）
        """
        bsz, seq_len, head_dim = key.shape

        if seq_len <= budget:
            return key, value

        # 1. 保留sink tokens
        sink_size = min(self.sink_tokens, budget // 3)
        sink_keys = key[:, :sink_size, :]
        sink_values = value[:, :sink_size, :]

        # 2. 均匀采样剩余tokens
        remaining_keys = key[:, sink_size:, :]
        remaining_values = value[:, sink_size:, :]
        remaining_len = remaining_keys.shape[1]

        remaining_budget = budget - sink_size

        # 均匀采样
        step = remaining_len / remaining_budget
        indices = torch.arange(remaining_budget, device=key.device) * step
        indices = indices.long()
        # 确保索引不越界
        indices = torch.clamp(indices, max=remaining_len - 1)

        selected_keys = remaining_keys[:, indices, :]
        selected_values = remaining_values[:, indices, :]

        # 合并
        compressed_key = torch.cat([sink_keys, selected_keys], dim=1)
        compressed_value = torch.cat([sink_values, selected_values], dim=1)

        return compressed_key, compressed_value

    def evict_by_groups(
        self,
        layer_idx: int,
        evict_start: int,
        evict_end: int
    ):
        """
        根据Group边界驱逐tokens
        """
        if len(self.key_cache) <= layer_idx:
            return

        key_cache = self.key_cache[layer_idx]  # [bsz, num_heads, seq_len, head_dim]
        value_cache = self.value_cache[layer_idx]

        # 移除指定范围的tokens
        before_keys = key_cache[:, :, :evict_start, :]
        after_keys = key_cache[:, :, evict_end:, :]
        before_values = value_cache[:, :, :evict_start, :]
        after_values = value_cache[:, :, evict_end:, :]

        # 重新拼接
        self.key_cache[layer_idx] = torch.cat([before_keys, after_keys], dim=2)
        self.value_cache[layer_idx] = torch.cat([before_values, after_values], dim=2)

        # 更新seen_tokens (反映物理缓存的减少，或者你可以选择不减少seen_tokens以保持position ID增长)
        # 注意：通常 seen_tokens 应该单调递增以生成正确的 PosID。
        # 这里如果 evict_by_groups 是物理移除，不需要修改 seen_tokens，
        # 除非你的模型依赖 seen_tokens 来计算 Attention Mask 的大小。
        # 暂时保持原逻辑：
        # evicted_count = evict_end - evict_
        # start
        # self._seen_tokens -= evicted_count

    def get_memory_usage(self) -> float:
        """
        获取当前KV cache的内存占用（GB）
        """
        total_elements = 0

        for key_cache, value_cache in zip(self.key_cache, self.value_cache):
            if key_cache is not None:
                total_elements += key_cache.numel()
            if value_cache is not None:
                total_elements += value_cache.numel()

        # 假设float16 (2 bytes per element)
        memory_bytes = total_elements * 2
        memory_gb = memory_bytes / (1024 ** 3)

        return memory_gb

    def adjust_budget(self, new_budget: int):
        """动态调整预算"""
        self.total_budget = new_budget

    def pop(self):
        """
        移除最后一个token的cache（用于回退）
        确保 pop 操作也能正确执行
        """
        if len(self.key_cache) > 0:
            # 简单实现：截断最后一个 token
            for i in range(len(self.key_cache)):
                self.key_cache[i] = self.key_cache[i][:, :, :-1, :]
                self.value_cache[i] = self.value_cache[i][:, :, :-1, :]