"""
StreamingLLM Cache Implementation
Based on: "StreamingLLM: Efficient Inference with Attention Sinks" (ICLR 2024)

Core idea: Fixed window size + Attention Sinks
"""
import torch
from typing import Optional, Tuple, List
from transformers.cache_utils import Cache


class StreamingLLMCache(Cache):
    """
    StreamingLLM Cache: 固定窗口 + Attention Sinks
    """
    
    def __init__(
        self,
        window_size: int = 512,
        sink_tokens: int = 4,
        device: str = "cuda"
    ):
        """
        Args:
            window_size: 滑动窗口大小
            sink_tokens: Attention sink tokens数量
            device: 设备
        """
        super().__init__()
        self.window_size = window_size
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
        更新KV cache，使用StreamingLLM策略
        
        Args:
            key_states: [batch, num_heads, seq_len, head_dim] 或 [batch, num_key_value_heads, seq_len, head_dim]
            value_states: [batch, num_heads, seq_len, head_dim] 或 [batch, num_key_value_heads, seq_len, head_dim]
            layer_idx: 层索引
            cache_kwargs: 其他参数
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
        
        # 如果序列长度超过窗口，进行压缩
        max_len = self.sink_tokens + self.window_size
        if current_seq_len > max_len:
            # 保留sink tokens + 最近窗口
            sink_key = self.key_cache[layer_idx][:, :, :self.sink_tokens]
            sink_value = self.value_cache[layer_idx][:, :, :self.sink_tokens]
            
            window_key = self.key_cache[layer_idx][:, :, -self.window_size:]
            window_value = self.value_cache[layer_idx][:, :, -self.window_size:]
            
            # 拼接
            self.key_cache[layer_idx] = torch.cat([sink_key, window_key], dim=2)
            self.value_cache[layer_idx] = torch.cat([sink_value, window_value], dim=2)
        
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
    
    def get_seq_length(self, layer_idx: Optional[int] = None) -> int:
        """获取序列长度"""
        if len(self.key_cache) == 0:
            return 0
        if layer_idx is None:
            layer_idx = 0
        return self.key_cache[layer_idx].size(2)
    
    def get_max_length(self) -> Optional[int]:
        """获取最大长度"""
        return self.sink_tokens + self.window_size
    
    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = None) -> int:
        """获取可用长度"""
        return min(new_seq_length, self.sink_tokens + self.window_size)

