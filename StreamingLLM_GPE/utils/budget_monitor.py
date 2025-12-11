"""
Budget-Constrained Inference: 预算监控模块

监控显存使用，动态触发KV cache驱逐
"""
import torch
from typing import Optional, TYPE_CHECKING
import sys
import os

if TYPE_CHECKING:
    from StreamingLLM_GPE.models.Qwen2_5.head_aware_cache import HeadAwareDynamicCache
    from StreamingLLM_GPE.utils.group_tracker import GroupTracker


class BudgetMonitor:
    """
    监控显存使用，触发KV cache驱逐
    
    在资源受限场景下（如11GB显存），确保KV cache不超过预算
    """
    
    def __init__(
        self,
        max_memory_gb: float = 4.0,
        check_interval: int = 100,  # 每N个tokens检查一次
        safety_margin: float = 0.1  # 10%安全边际
    ):
        """
        Args:
            max_memory_gb: 最大KV cache内存占用（GB）
            check_interval: 检查间隔（tokens）
            safety_margin: 安全边际（避免OOM）
        """
        self.max_memory_gb = max_memory_gb
        self.check_interval = check_interval
        self.safety_margin = safety_margin
        self.token_count = 0
        
        # 记录内存使用历史
        self.memory_history = []
        
    def check_and_evict(
        self,
        cache: "HeadAwareDynamicCache",
        group_tracker: Optional["GroupTracker"] = None,
        force_check: bool = False
    ) -> bool:
        """
        检查显存，如果超标则触发驱逐
        
        Args:
            cache: HeadAwareDynamicCache实例
            group_tracker: GroupTracker实例（可选，用于Group-level驱逐）
            force_check: 是否强制检查（忽略interval）
        
        Returns:
            是否触发了驱逐
        """
        self.token_count += 1
        
        # 检查是否需要检查
        if not force_check and self.token_count % self.check_interval != 0:
            return False
        
        # 获取当前内存使用
        current_memory = cache.get_memory_usage()
        self.memory_history.append(current_memory)
        
        # 计算目标内存（考虑安全边际）
        target_memory = self.max_memory_gb * (1 - self.safety_margin)
        
        if current_memory <= target_memory:
            return False
        
        # 内存超标，需要驱逐
        print(f"[BudgetMonitor] Memory overflow: {current_memory:.2f}GB > {target_memory:.2f}GB")
        
        # 计算需要释放的内存
        excess_memory = current_memory - target_memory
        excess_ratio = excess_memory / current_memory
        
        # 估算需要减少的tokens数量
        current_tokens = cache.get_seq_length(0) if hasattr(cache, 'get_seq_length') else 0
        if current_tokens == 0:
            # 估算：假设每个token占用约2KB（float16, 32 layers, 32 heads）
            tokens_per_gb = 500000  # 粗略估算
            excess_tokens = int(excess_memory * tokens_per_gb)
        else:
            excess_tokens = int(current_tokens * excess_ratio)
        
        # 执行驱逐
        if group_tracker is not None:
            # Group-level驱逐
            self._evict_by_groups(cache, group_tracker, excess_tokens)
        else:
            # Token-level驱逐（调整预算）
            new_budget = max(512, current_tokens - excess_tokens)  # 至少保留512 tokens
            cache.adjust_budget(new_budget)
            print(f"[BudgetMonitor] Adjusted budget to {new_budget} tokens")
        
        return True
    
    def _evict_by_groups(
        self,
        cache: "HeadAwareDynamicCache",
        group_tracker: "GroupTracker",
        excess_tokens: int
    ):
        """
        基于Group进行驱逐
        
        Args:
            cache: HeadAwareDynamicCache实例
            group_tracker: GroupTracker实例
            excess_tokens: 需要减少的tokens数量
        """
        # 计算需要驱逐的groups数量
        avg_group_size = group_tracker.get_total_tokens() / max(group_tracker.get_group_count(), 1)
        groups_to_evict = max(1, int(excess_tokens / avg_group_size))
        
        # 获取需要驱逐的group IDs
        evict_group_ids = group_tracker.get_groups_to_evict(
            max_groups=group_tracker.get_group_count() - groups_to_evict
        )
        
        if not evict_group_ids:
            return
        
        # 执行驱逐
        evict_start, evict_end = group_tracker.evict_groups(evict_group_ids)
        
        if evict_start is not None and evict_end is not None:
            # 对所有层执行驱逐
            for layer_idx in range(len(cache.key_cache)):
                cache.evict_by_groups(layer_idx, evict_start, evict_end)
            
            print(f"[BudgetMonitor] Evicted {len(evict_group_ids)} groups "
                  f"({evict_end - evict_start} tokens)")
    
    def get_memory_stats(self) -> dict:
        """获取内存统计信息"""
        if not self.memory_history:
            return {}
        
        return {
            'current_memory_gb': self.memory_history[-1],
            'max_memory_gb': max(self.memory_history),
            'avg_memory_gb': sum(self.memory_history) / len(self.memory_history),
            'check_count': len(self.memory_history)
        }
    
    def reset(self):
        """重置监控器"""
        self.token_count = 0
        self.memory_history = []

