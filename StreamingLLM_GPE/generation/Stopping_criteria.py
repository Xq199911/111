# Modified 2025 by Junlong Tong (Shanghai Jiao Tong University & Eastern Institute of Technology).
# Copy and modified from 'Simul-LLM' repository.


import torch
from transformers.generation.stopping_criteria import StoppingCriteria


class StopTokenCriteria(StoppingCriteria):
    def __init__(
            self,
            tokenizer,
            max_new_tokens: int,
            end_Instruct = '<|end|>'
        ):
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.end_Instruct = end_Instruct


    def __call__(self, target_ids: torch.LongTensor, scores: torch.FloatTensor, token_count, **kwargs) -> bool:
        token_pred = self.tokenizer.decode(target_ids[0][-1:])
        token_preds = self.tokenizer.decode(target_ids[0])
        is_done = False
        remove_last_token = False
        terminating_tok = [".", ",", ":", ";", "?", "!"]
        
        # 检查是否达到max_new_tokens（这是真正的停止条件）
        if token_count >= self.max_new_tokens:
            is_done = True
            return torch.tensor(is_done), torch.tensor(remove_last_token)
        
        # 检查是否遇到end_Instruct（这也是停止条件）
        if self.end_Instruct in token_pred:
            is_done = True
            return torch.tensor(is_done), torch.tensor(remove_last_token)
        
        # 检查是否遇到空格或终止符（这表示一个word生成完成，但不是整个生成完成）
        # 在streaming模式下，遇到空格只是表示当前word完成，应该继续生成下一个word
        if ' ' in token_preds[1:] or token_pred in terminating_tok:
            # 遇到空格或终止符，表示当前word生成完成
            # 但这不应该停止整个生成，只是表示需要读取下一个源端word
            is_done = False  # 不停止，继续生成
            if ' ' in token_preds[1:] and target_ids[0].shape[0] >= 2:
                remove_last_token = True  # 移除最后一个token（空格）
        
        return torch.tensor(is_done), torch.tensor(remove_last_token)