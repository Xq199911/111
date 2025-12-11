"""
H2O (Heavy-Hitter Oracle) Cache Implementation
Based on: "H2O: Heavy-Hitter Oracle for Efficient Generative Inference" (NeurIPS 2023)

Core idea: Unified compression of all heads' KV cache based on importance scores
"""
import torch
from typing import Optional, Tuple, List
from transformers.cache_utils import Cache


class H2OCache(Cache):
    """
    H2O Cache: 基于重要性分数的统一压缩
    """
    
    def __init__(
        self,
        budget_per_layer: int = 2048,
        sink_tokens: int = 4,
        device: str = "cuda"
    ):
        """
        Args:
            budget_per_layer: 每层的KV cache预算（tokens）
            sink_tokens: 保留的sink tokens数量
            device: 设备
        """
        super().__init__()
        self.budget_per_layer = budget_per_layer
        self.sink_tokens = sink_tokens
        self.device = device
        
        # KV cache存储
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        更新KV cache，使用H2O压缩策略
        
        Args:
            key_states: [batch, num_heads, seq_len, head_dim] 或 [batch, num_key_value_heads, seq_len, head_dim]
            value_states: [batch, num_heads, seq_len, head_dim] 或 [batch, num_key_value_heads, seq_len, head_dim]
            layer_idx: 层索引
            cache_kwargs: 其他参数（可能包含attention_scores）
        """
        # 初始化cache
        if layer_idx == 0:
            self.key_cache = []
            self.value_cache = []
        
        # 将新tokens添加到cache
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # 拼接新tokens
            self.key_cache[layer_idx] = torch.cat([
                self.key_cache[layer_idx], key_states
            ], dim=2)
            self.value_cache[layer_idx] = torch.cat([
                self.value_cache[layer_idx], value_states
            ], dim=2)
        
        current_seq_len = self.key_cache[layer_idx].size(2)
        
        # 如果序列长度超过预算，进行压缩
        if current_seq_len > self.budget_per_layer:
            # 尝试从cache_kwargs获取attention_scores
            attention_scores = None
            if cache_kwargs is not None:
                attention_scores = cache_kwargs.get("attention_scores", None)
            
            # 压缩cache
            compressed_key, compressed_value = self._compress(
                self.key_cache[layer_idx],
                self.value_cache[layer_idx],
                attention_scores,
                layer_idx
            )
            
            self.key_cache[layer_idx] = compressed_key
            self.value_cache[layer_idx] = compressed_value
        
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def _compress(
            self,
            key: torch.Tensor,
            value: torch.Tensor,
            attention_scores: Optional[torch.Tensor],
            layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        压缩KV cache

        Args:
            key: [batch, num_heads, seq_len, head_dim]
            value: [batch, num_heads, seq_len, head_dim]
            attention_scores: [batch, num_heads, seq_len, seq_len] 或 None
        """
        batch_size, num_heads, seq_len, head_dim = key.shape

        # 计算除了sink之外还能保留多少个token
        remaining_budget = self.budget_per_layer - self.sink_tokens

        # 1. 关键修正：如果 attention_scores 为空，强制回退到 StreamingLLM (Sliding Window)
        # 这对于 Baseline 来说是必须的保底逻辑，否则会 crash 或产生不可预知的行为
        if attention_scores is None:
            # 仅在第一次发生时打印警告，避免日志刷屏
            if not hasattr(self, "_has_warned_no_attn"):
                print(
                    f"[H2O Warning] Layer {layer_idx}: No attention scores found! Falling back to Sliding Window (StreamingLLM).")
                self._has_warned_no_attn = True

            # 策略：保留 Sink + 最近的 Window
            sink_key = key[:, :, :self.sink_tokens]
            sink_value = value[:, :, :self.sink_tokens]

            # 取最近的 remaining_budget 个 tokens
            window_key = key[:, :, -remaining_budget:]
            window_value = value[:, :, -remaining_budget:]

            # 拼接
            compressed_key = torch.cat([sink_key, window_key], dim=2)
            compressed_value = torch.cat([sink_value, window_value], dim=2)

            return compressed_key, compressed_value

        # 2. 如果有 attention_scores，执行正常的 H2O 逻辑
        # 保留sink tokens
        sink_key = key[:, :, :self.sink_tokens]
        sink_value = value[:, :, :self.sink_tokens]

        # 使用attention scores作为重要性
        # importance shape: [batch, num_heads, seq_len, seq_len] (通常)
        # 注意：这里假设 attention_scores 包含了历史所有 token 的注意力
        # 如果是 generation 阶段，attention_scores 可能只包含当前 token 对过去的注意力
        # H2O 实际上需要累积注意力分数 (Accumulated Attention Scores)

        # 计算每个token的重要性：对query维度求和
        importance = attention_scores.sum(dim=-2)  # [batch, num_heads, seq_len] (兼容性更好的写法 dim=-2)

        # 对所有heads求平均 (H2O 论文中通常是对每个 Head 独立做，但这里统一处理也可以)
        importance = importance.mean(dim=1)  # [batch, seq_len]

        # 排除sink tokens (因为sink会被强制保留，不参与竞争)
        importance_no_sink = importance[:, self.sink_tokens:]

        # 确保不会越界
        current_candidates = importance_no_sink.size(1)
        actual_k = min(remaining_budget, current_candidates)

        # 选择top-k最重要的tokens
        _, top_indices = torch.topk(importance_no_sink, actual_k, dim=1)

        # 调整索引（加上sink tokens的偏移，恢复到原始 key 的索引）
        top_indices = top_indices + self.sink_tokens  # [batch, k]

        # 排序索引，保持时序 (可选，但在 KV Cache 中通常保持时序更好)
        top_indices, _ = top_indices.sort(dim=1)

        # 收集选中的tokens
        # 扩展索引维度以匹配 gather 的要求: [batch, num_heads, k, head_dim]
        gather_index = top_indices.unsqueeze(1).unsqueeze(-1).expand(-1, num_heads, -1, head_dim)

        selected_key = torch.gather(key, dim=2, index=gather_index)
        selected_value = torch.gather(value, dim=2, index=gather_index)

        # 拼接sink + 选中的tokens
        compressed_key = torch.cat([sink_key, selected_key], dim=2)
        compressed_value = torch.cat([sink_value, selected_value], dim=2)

        return compressed_key, compressed_value
    
    def get_seq_length(self, layer_idx: Optional[int] = None) -> int:
        """获取序列长度"""
        if len(self.key_cache) == 0:
            return 0
        if layer_idx is None:
            layer_idx = 0
        return self.key_cache[layer_idx].size(2)
    
    def get_max_length(self) -> Optional[int]:
        """获取最大长度"""
        return self.budget_per_layer
    
    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = None) -> int:
        """获取可用长度"""
        return min(new_seq_length, self.budget_per_layer)

