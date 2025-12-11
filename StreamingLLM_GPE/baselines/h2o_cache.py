# """
# H2O (Heavy-Hitter Oracle) Cache Implementation - FIXED
# Based on: "H2O: Heavy-Hitter Oracle for Efficient Generative Inference" (NeurIPS 2023)
#
# Core idea: Unified compression of all heads' KV cache based on importance scores
# """
# import torch
# from typing import Optional, Tuple, List
# from transformers.cache_utils import Cache
#
#
# class H2OCache(Cache):
#     """
#     H2O Cache: 基于重要性分数的统一压缩
#     (已修复：缓存重置 Bug 和 RoPE 位置编码逻辑长度问题)
#     """
#
#     def __init__(
#         self,
#         budget_per_layer: int = 2048,
#         sink_tokens: int = 4,
#         device: str = "cuda"
#     ):
#         """
#         Args:
#             budget_per_layer: 每层的KV cache预算（tokens）
#             sink_tokens: 保留的sink tokens数量
#             device: 设备
#         """
#         super().__init__()
#         self.budget_per_layer = budget_per_layer
#         self.sink_tokens = sink_tokens
#         self.device = device
#
#         # KV cache存储
#         self.key_cache: List[torch.Tensor] = []
#         self.value_cache: List[torch.Tensor] = []
#
#         # [Fix 2] 维护真实的逻辑长度，用于 RoPE 位置计算
#         self.seen_tokens = 0
#
#     def update(
#         self,
#         key_states: torch.Tensor,
#         value_states: torch.Tensor,
#         layer_idx: int,
#         cache_kwargs: Optional[dict] = None
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         更新KV cache，使用H2O压缩策略
#         """
#         # [Fix 1] 删除了此处 "if layer_idx == 0: self.key_cache = []" 的致命错误代码
#         # 缓存列表的初始化由下方的 append 逻辑自动处理
#
#         # [Fix 2] 更新 seen_tokens (仅在第0层更新一次，避免重复计数)
#         if layer_idx == 0:
#             self.seen_tokens += key_states.shape[-2]
#
#         # 将新tokens添加到cache
#         if len(self.key_cache) <= layer_idx:
#             self.key_cache.append(key_states)
#             self.value_cache.append(value_states)
#         else:
#             # 拼接新tokens
#             self.key_cache[layer_idx] = torch.cat([
#                 self.key_cache[layer_idx], key_states
#             ], dim=2)
#             self.value_cache[layer_idx] = torch.cat([
#                 self.value_cache[layer_idx], value_states
#             ], dim=2)
#
#         current_seq_len = self.key_cache[layer_idx].size(2)
#
#         # 如果序列长度超过预算，进行压缩
#         if current_seq_len > self.budget_per_layer:
#             # 尝试从cache_kwargs获取attention_scores
#             attention_scores = None
#             if cache_kwargs is not None:
#                 attention_scores = cache_kwargs.get("attention_scores", None)
#
#             # 压缩cache
#             compressed_key, compressed_value = self._compress(
#                 self.key_cache[layer_idx],
#                 self.value_cache[layer_idx],
#                 attention_scores,
#                 layer_idx
#             )
#
#             self.key_cache[layer_idx] = compressed_key
#             self.value_cache[layer_idx] = compressed_value
#
#         return self.key_cache[layer_idx], self.value_cache[layer_idx]
#
#     def _sliding_window_fallback(
#             self,
#             key: torch.Tensor,
#             value: torch.Tensor,
#             remaining_budget: int
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         回退到 Sliding Window 策略（保留 Sink + 最近的 Window）
#         """
#         sink_key = key[:, :, :self.sink_tokens]
#         sink_value = value[:, :, :self.sink_tokens]
#
#         # 取最近的 remaining_budget 个 tokens
#         window_key = key[:, :, -remaining_budget:]
#         window_value = value[:, :, -remaining_budget:]
#
#         # 拼接
#         compressed_key = torch.cat([sink_key, window_key], dim=2)
#         compressed_value = torch.cat([sink_value, window_value], dim=2)
#
#         return compressed_key, compressed_value
#
#     def _compress(
#             self,
#             key: torch.Tensor,
#             value: torch.Tensor,
#             attention_scores: Optional[torch.Tensor],
#             layer_idx: int
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         压缩KV cache
#         """
#         batch_size, num_heads, seq_len, head_dim = key.shape
#
#         # 计算除了sink之外还能保留多少个token
#         remaining_budget = self.budget_per_layer - self.sink_tokens
#
#         # 如果 attention_scores 为空，强制回退到 StreamingLLM (Sliding Window)
#         if attention_scores is None:
#             # 仅在第一次发生时打印警告
#             if not hasattr(self, "_has_warned_no_attn"):
#                 # print(f"[H2O Warning] Layer {layer_idx}: No attention scores found! Falling back to Sliding Window.")
#                 self._has_warned_no_attn = True
#             return self._sliding_window_fallback(key, value, remaining_budget)
#
#         # 处理不同形状的 attention_scores
#         # H2O 需要累积注意力分数
#         if attention_scores.dim() == 4:
#             if attention_scores.size(2) == 1:
#                 # 生成阶段: [batch, num_heads, 1, seq_len] -> [batch, num_heads, seq_len]
#                 importance = attention_scores.squeeze(2)
#             else:
#                 # 预填充阶段: [batch, num_heads, seq_len, seq_len] -> [batch, num_heads, seq_len]
#                 importance = attention_scores.sum(dim=-2)
#         else:
#             return self._sliding_window_fallback(key, value, remaining_budget)
#
#         # 对所有heads求平均
#         importance = importance.mean(dim=1)  # [batch, seq_len]
#
#         if importance.size(1) != seq_len:
#             return self._sliding_window_fallback(key, value, remaining_budget)
#
#         # 排除sink tokens
#         importance_no_sink = importance[:, self.sink_tokens:]
#
#         # 确保不会越界
#         current_candidates = importance_no_sink.size(1)
#         actual_k = min(remaining_budget, current_candidates)
#
#         # 选择top-k最重要的tokens
#         _, top_indices = torch.topk(importance_no_sink, actual_k, dim=1)
#
#         # 调整索引
#         top_indices = top_indices + self.sink_tokens
#
#         # 排序索引，保持时序
#         top_indices, _ = top_indices.sort(dim=1)
#
#         # 收集选中的tokens
#         gather_index = top_indices.unsqueeze(1).unsqueeze(-1).expand(-1, num_heads, -1, head_dim).long()
#
#         selected_key = torch.gather(key, dim=2, index=gather_index)
#         selected_value = torch.gather(value, dim=2, index=gather_index)
#
#         # 拼接sink + 选中的tokens
#         compressed_key = torch.cat([sink_key, selected_key], dim=2)
#         compressed_value = torch.cat([sink_value, selected_value], dim=2)
#
#         return compressed_key, compressed_value
#
#     def get_seq_length(self, layer_idx: Optional[int] = None) -> int:
#         """
#         [Fix 2] 获取序列长度
#         注意：必须返回逻辑长度（seen_tokens），而不是物理缓存长度，
#         否则 RoPE 位置编码会出错，导致长序列生成崩溃。
#         """
#         return self.seen_tokens
#
#     def get_max_length(self) -> Optional[int]:
#         """获取最大长度"""
#         return self.budget_per_layer
#
#     def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = None) -> int:
#         """获取可用长度"""
#         return self.seen_tokens # 配合 get_seq_length 返回逻辑长度

"""
H2O (Heavy-Hitter Oracle) Cache Implementation - FIXED v4 (Robust)
Based on: "H2O: Heavy-Hitter Oracle for Efficient Generative Inference" (NeurIPS 2023)
"""
import torch
from typing import Optional, Tuple, List
from transformers.cache_utils import Cache

class H2OCache(Cache):
    """
    H2O Cache: 基于重要性分数的统一压缩
    (修复：RoPE逻辑长度、维度匹配、Sliding Window重复拼接Bug)
    """

    def __init__(
        self,
        budget_per_layer: int = 2048,
        sink_tokens: int = 4,
        device: str = "cuda"
    ):
        super().__init__()
        self.budget_per_layer = budget_per_layer
        self.sink_tokens = sink_tokens
        self.device = device

        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self._seen_tokens = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # [逻辑长度] 仅在第0层更新
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # [缓存拼接]
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat([
                self.key_cache[layer_idx], key_states
            ], dim=2)
            self.value_cache[layer_idx] = torch.cat([
                self.value_cache[layer_idx], value_states
            ], dim=2)

        current_seq_len = self.key_cache[layer_idx].size(2)

        # [压缩触发条件] 严格检查是否超过预算
        if current_seq_len > self.budget_per_layer:
            attention_scores = None
            if cache_kwargs is not None:
                attention_scores = cache_kwargs.get("attention_scores", None)

            compressed_key, compressed_value = self._compress(
                self.key_cache[layer_idx],
                self.value_cache[layer_idx],
                attention_scores,
                layer_idx
            )

            self.key_cache[layer_idx] = compressed_key
            self.value_cache[layer_idx] = compressed_value

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def _sliding_window_fallback(self, key, value, remaining_budget):
        seq_len = key.size(2)

        # [Bug修复] 如果当前长度还未超过总预算（含Sink），不需要压缩！
        # 虽然外部 update 有检查，但这里作为 fallback 更安全
        if seq_len <= self.budget_per_layer:
            return key, value

        sink_key = key[:, :, :self.sink_tokens]
        sink_value = value[:, :, :self.sink_tokens]

        # 取最近的 remaining_budget 个 tokens
        # 注意：这里前提是 seq_len > budget，所以不会与 sink 重叠
        window_key = key[:, :, -remaining_budget:]
        window_value = value[:, :, -remaining_budget:]

        return torch.cat([sink_key, window_key], dim=2), torch.cat([sink_value, window_value], dim=2)

    def _compress(
            self,
            key: torch.Tensor,
            value: torch.Tensor,
            attention_scores: Optional[torch.Tensor],
            layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size, num_heads, seq_len, head_dim = key.shape
        remaining_budget = self.budget_per_layer - self.sink_tokens

        # [Safety Check] 如果长度未超标，直接返回
        if seq_len <= self.budget_per_layer:
            return key, value

        # Fallback to Sliding Window if no attention scores
        if attention_scores is None:
            if not hasattr(self, "_has_warned_no_attn"):
                # print(f"[H2O Warning] Layer {layer_idx}: No attention scores found! Falling back.")
                self._has_warned_no_attn = True
            return self._sliding_window_fallback(key, value, remaining_budget)

        # Handle Attention Scores
        if attention_scores.dim() == 4:
            if attention_scores.size(2) == 1:
                importance = attention_scores.squeeze(2)
            else:
                importance = attention_scores.sum(dim=-2)
        else:
            return self._sliding_window_fallback(key, value, remaining_budget)

        importance = importance.mean(dim=1)  # [batch, seq_len]

        if importance.size(1) != seq_len:
            return self._sliding_window_fallback(key, value, remaining_budget)

        # Select Top-K
        importance_no_sink = importance[:, self.sink_tokens:]
        current_candidates = importance_no_sink.size(1)
        actual_k = min(remaining_budget, current_candidates)

        _, top_indices = torch.topk(importance_no_sink, actual_k, dim=1)
        top_indices = top_indices + self.sink_tokens
        top_indices, _ = top_indices.sort(dim=1)

        try:
            top_indices = top_indices.long().to(key.device)
            gather_index = top_indices.unsqueeze(1).unsqueeze(-1).expand(
                batch_size, num_heads, actual_k, head_dim
            )

            selected_key = torch.gather(key, dim=2, index=gather_index)
            selected_value = torch.gather(value, dim=2, index=gather_index)

            sink_key = key[:, :, :self.sink_tokens]
            sink_value = value[:, :, :self.sink_tokens]

            compressed_key = torch.cat([sink_key, selected_key], dim=2)
            compressed_value = torch.cat([sink_value, selected_value], dim=2)

            return compressed_key, compressed_value

        except RuntimeError as e:
            print(f"[H2O Error] Layer {layer_idx} failed: {e}")
            return self._sliding_window_fallback(key, value, remaining_budget)

    def get_seq_length(self, layer_idx: Optional[int] = None) -> int:
        return self._seen_tokens

    def get_max_length(self) -> Optional[int]:
        return self.budget_per_layer

    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = None) -> int:
        return self._seen_tokens