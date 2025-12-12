第一步 验证H2O Baseline
python StreamingLLM_GPE/evaluate/multi_model_eval.py  --LLM_backbone Qwen  --LLM_path .\models\Qwen2.5-3B-Instruct  --use_h2o  --h2o_budget 512  --max_new_tokens 2048  --output_dir .  --min_source_length 1000  --max_samples 1  --quantization none  --inference_mode batch
结果：
(py) PS C:\Users\xiong\Desktop\StreamingLLM> python StreamingLLM_GPE/evaluate/multi_model_eval.py  --LLM_backbone Qwen  --LLM_path .\models\Qwen2.5-3B-Instruct  --use_h2o  --h2o_budget 512  --max_new_tokens 2048  --output_dir .  --min_source_length 1000  --max_samples 1  --quantization none  --inference_mode batch

    Multi-Model Evaluation Configuration:
    - Model Architecture: Qwen
    - Model Path: .\models\Qwen2.5-3B-Instruct
    - Inference Mode: batch
    - Head-Aware: False
    - Group-Aware: False
    - H2O: True
    - StreamingLLM: False
    - Cache Budget: 512 tokens/layer (H2O)
    - Max Memory: 4.0 GB
    - Wait-k: 5
    - Min Source Length: 1000
    - Max Samples: 1
    - Max New Tokens: 2048

Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:03<00:00,  1.64s/it]
Filter: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 16/16 [00:00<00:00, 333.31 examples/s]
Evaluating Qwen:   0%|                                                                                                                      | 0/1 [00:00<?, ?it/s]
[DEBUG] Sample 0:
  Source length: 1523 words
  ⚠️ 警告: 源文本异常长（1523 words）

[DEBUG] Sample 0 output:
  Output length: 2048 tokens
  Output text (first 300 chars): Le sportif Jhonathan Florez a sauté d'un hélicoptère au-dessus de Bogota, la capitale du Colombie, le jeudi dernier. Vêtu d'une ai
le, il a volé près du célèbre sanctuaire de Monserrate à une vitesse de 160 km/h. Ce sanctuaire se trouve à une altitude de plus de 3000 mètres et de nombreux spectateurs...
  Target text (first 300 chars): Le sportif Jhonathan Florez a sauté jeudi d'un hélicoptère au-dessus de Bogota, la capitale colombienne. Equipé d'un wingsuit (une
 combinaison munie d'ailes), il est passé à 160 km/h au-dessus du célèbre sanctuaire Monserrate, situé à plus de 3 000 mètres d'altitude, où de nombreux badauds s'étaient

[Qwen] Step 0:
  Output Length: 2048 tokens
Evaluating Qwen: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [02:39<00:00, 159.00s/it] 

[Qwen] Results saved to ./results.json
[Qwen] BLEU Score: 17.89
[Qwen] Peak GPU Memory: 5.85GB
(py) PS C:\Users\xiong\Desktop\StreamingLLM> 
results.json
{
  "model_architecture": "Qwen",
  "model_path": ".\\models\\Qwen2.5-3B-Instruct",
  "bleu_score": 17.888648591173734,
  "memory_stats": {
    "peak_memory_gb": 5.854067802429199,
    "avg_memory_gb": 5.854040145874023,
    "final_memory_gb": 5.854040145874023
  },
  "cache_stats": {
    "avg_cache_memory_gb": 0,
    "peak_cache_memory_gb": 0
  },
  "length_stats": {
    "total_tokens": 2048,
    "max_length": 2048,
    "avg_length": 2048.0
  },
  "latency_stats": {
    "avg_inference_time": 158.7904942035675
  },
  "streaming_stats": {}
}

第二步 验证Baseline (StreamingLLM)
python StreamingLLM_GPE/evaluate/multi_model_eval.py `
    --LLM_backbone Qwen `
    --LLM_path .\models\Qwen2.5-3B-Instruct `
    --use_streamingllm `
    --streamingllm_window 512 `
    --max_new_tokens 2048 `
    --output_dir .\output_logs\streamingllm_test `
    --min_source_length 1000 `
    --max_samples 1 `
    --quantization none `
    --inference_mode batch
结果：
(py) PS C:\Users\xiong\Desktop\StreamingLLM> python StreamingLLM_GPE/evaluate/multi_model_eval.py `
>>     --LLM_backbone Qwen `
>>     --LLM_path .\models\Qwen2.5-3B-Instruct `
>>     --use_streamingllm `
>>     --streamingllm_window 512 `
>>     --max_new_tokens 2048 `
>>     --output_dir .\output_logs\streamingllm_test `
>>     --min_source_length 1000 `
>>     --max_samples 1 `
>>     --quantization none `
>>     --inference_mode batch

    Multi-Model Evaluation Configuration:
    - Model Architecture: Qwen
    - Model Path: .\models\Qwen2.5-3B-Instruct
    - Inference Mode: batch
    - Head-Aware: False
    - Group-Aware: False
    - H2O: False
    - StreamingLLM: True
    - Cache Budget: 512 tokens (StreamingLLM window)
    - Max Memory: 4.0 GB
    - Wait-k: 5
    - Min Source Length: 1000
    - Max Samples: 1
    - Max New Tokens: 2048

Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:03<00:00,  1.66s/it]
Filter: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 16/16 [00:00<00:00, 404.94 examples/s]
Evaluating Qwen:   0%|                                                                                                                      | 0/1 [00:00<?, ?it/s]
[DEBUG] Sample 0:
  Source length: 1523 words
  ⚠️ 警告: 源文本异常长（1523 words）

[DEBUG] Sample 0 output:
  Output length: 2048 tokens
  Output text (first 300 chars): Le sportif Jhonathan Florez a sauté d'un hélicoptère au-dessus de Bogota, la capitale du Colombie, le jeudi dernier. Vêtu d'une ai
le, il a volé près du célèbre sanctuaire de Monserrate à une vitesse de 160 km/h. Ce sanctuaire se trouve à une altitude de plus de 3000 mètres et de nombreux spectateurs...
  Target text (first 300 chars): Le sportif Jhonathan Florez a sauté jeudi d'un hélicoptère au-dessus de Bogota, la capitale colombienne. Equipé d'un wingsuit (une
 combinaison munie d'ailes), il est passé à 160 km/h au-dessus du célèbre sanctuaire Monserrate, situé à plus de 3 000 mètres d'altitude, où de nombreux badauds s'étaient

[Qwen] Step 0:
  Output Length: 2048 tokens
Evaluating Qwen: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [02:40<00:00, 160.16s/it] 

[Qwen] Results saved to .\output_logs\streamingllm_test/results.json
[Qwen] BLEU Score: 17.89
[Qwen] Peak GPU Memory: 5.85GB
(py) PS C:\Users\xiong\Desktop\StreamingLLM> 

results.json
{
  "model_architecture": "Qwen",
  "model_path": ".\\models\\Qwen2.5-3B-Instruct",
  "bleu_score": 17.888648591173734,
  "memory_stats": {
    "peak_memory_gb": 5.854067802429199,
    "avg_memory_gb": 5.854040145874023,
    "final_memory_gb": 5.854040145874023
  },
  "cache_stats": {
    "avg_cache_memory_gb": 0,
    "peak_cache_memory_gb": 0
  },
  "length_stats": {
    "total_tokens": 2048,
    "max_length": 2048,
    "avg_length": 2048.0
  },
  "latency_stats": {
    "avg_inference_time": 159.95353293418884
  },
  "streaming_stats": {}
}

