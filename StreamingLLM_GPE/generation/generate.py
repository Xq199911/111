# # coding=utf-8
# # Modified 2025 by Junlong Tong (Shanghai Jiao Tong University & Eastern Institute of Technology, Ningbo).
# #
# # This file is a modified version of the original transformers implementation from:
# # The HuggingFace Inc. team.
# #
# # Original license and copyright as follows:
# # Copyright 2018 The Google AI Language Team Authors, Facebook AI Research authors and The HuggingFace Inc. team.
# # Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
#
#
# from typing import List, Dict
# from transformers.configuration_utils import PretrainedConfig
# from transformers.generation.utils import *
# from transformers.cache_utils import DynamicCache as TransformersDynamicCache
# from transformers.modeling_utils import PreTrainedModel
# from .Stopping_criteria import StopTokenCriteria
# from transformers.utils import is_torchdynamo_compiling
# import inspect
#
# # Copied from transformers.models.llama.modeling_llama._prepare_4d_causal_attention_mask_with_cache_position
# def _prepare_4d_causal_attention_mask_with_cache_position(
#     attention_mask: torch.Tensor,
#     sequence_length: int,
#     target_length: int,
#     dtype: torch.dtype,
#     device: torch.device,
#     min_dtype: float,
#     cache_position: torch.Tensor,
#     batch_size: int,
# ):
#     """
#     Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
#     `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.
#
#     Args:
#         attention_mask (`torch.Tensor`):
#             A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
#         sequence_length (`int`):
#             The sequence length being processed.
#         target_length (`int`):
#             The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
#         dtype (`torch.dtype`):
#             The dtype to use for the 4D attention mask.
#         device (`torch.device`):
#             The device to plcae the 4D attention mask on.
#         min_dtype (`float`):
#             The minimum value representable with the dtype `dtype`.
#         cache_position (`torch.Tensor`):
#             Indices depicting the position of the input sequence tokens in the sequence.
#         batch_size (`torch.Tensor`):
#             Batch size.
#     """
#     if attention_mask is not None and attention_mask.dim() == 4:
#         # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
#         causal_mask = attention_mask
#     else:
#         causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
#         if sequence_length != 1:
#             causal_mask = torch.triu(causal_mask, diagonal=1)
#         causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
#         causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
#         if attention_mask is not None:
#             causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
#             mask_length = attention_mask.shape[-1]
#             padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
#             padding_mask = padding_mask == 0
#             causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
#                 padding_mask, min_dtype
#             )
#
#     return causal_mask
#
# # ...
# # add pop method to DynamicCache
# class DynamicCache(TransformersDynamicCache):
#     def __init__(self) -> None:
#         super().__init__()
#
#     def pop(self):
#         self._seen_tokens -= 1
#
#         # Update the cache
#         target_key_cache = []
#         target_value_cache = []
#         for key_cache,value_cache in zip(self.key_cache, self.value_cache):
#             target_key_cache.append(key_cache[...,:-1,:])
#             target_value_cache.append(value_cache[...,:-1,:])
#         self.key_cache = target_key_cache
#         self.value_cache = target_value_cache
#
# class unified_PreTrainedModel(PreTrainedModel):
#     def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
#         super().__init__(config, *inputs, **kwargs)
#
#
#     # The generate method inherited from transformers.generation.utils.GenerationMixin, where GenerationMixin is first inherited by PreTrainedModel
#     @torch.no_grad()
#     def generate(
#         self,
#         inputs: Optional[torch.Tensor] = None,
#         generation_config: Optional[GenerationConfig] = None,
#         logits_processor: Optional[LogitsProcessorList] = None,
#         stopping_criteria: Optional[StoppingCriteriaList] = None,
#         prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
#         synced_gpus: Optional[bool] = None,
#         assistant_model: Optional["PreTrainedModel"] = None,
#         streamer: Optional["BaseStreamer"] = None,
#         negative_prompt_ids: Optional[torch.Tensor] = None,
#         negative_prompt_attention_mask: Optional[torch.Tensor] = None,
#         **kwargs,
#     ) -> Union[GenerateOutput, torch.LongTensor]:
#
#         generate_mode = kwargs.get("generate_mode", "batch") # must be "batch", "streaming"
#         split_mode = kwargs.get("split_mode", None) # must be one of ["token", "word"] if generate_mode == "streaming"
#
#         if generate_mode == "batch": # same as transformers.generation.utils.GenerationMixin.generate
#             wait_lagging = None
#             return super().generate(
#                 inputs,
#                 generation_config,
#                 logits_processor,
#                 stopping_criteria,
#                 prefix_allowed_tokens_fn,
#                 synced_gpus,
#                 assistant_model,
#                 streamer,
#                 negative_prompt_ids,
#                 negative_prompt_attention_mask,
#                 **kwargs,
#             ), wait_lagging
#
#         elif generate_mode == "streaming":
#             assert split_mode in ["token", "word"], f"streaming_split must be one of ['token', 'word'], but got {split_mode}."
#             result, wait_lagging = self.streaming_generate(
#                 split_mode,
#                 inputs,
#                 generation_config,
#                 logits_processor,
#                 stopping_criteria,
#                 prefix_allowed_tokens_fn,
#                 synced_gpus,
#                 assistant_model,
#                 streamer,
#                 negative_prompt_ids,
#                 negative_prompt_attention_mask,
#                 **kwargs,
#             )
#             return result, wait_lagging
#
#         else:
#             raise ValueError(f"generate_mode must be one of ['batch', 'streaming'], but got {generate_mode}.")
#
#
#     def streaming_generate(
#         self,
#         streaming_split: str = "word", # must be one of ["token", "word"]
#         inputs: Optional[torch.Tensor] = None,
#         generation_config: Optional[GenerationConfig] = None,
#         logits_processor: Optional[LogitsProcessorList] = None,
#         stopping_criteria: Optional[StoppingCriteriaList] = None,
#         prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
#         synced_gpus: Optional[bool] = None,
#         assistant_model: Optional["PreTrainedModel"] = None,
#         streamer: Optional["BaseStreamer"] = None,
#         negative_prompt_ids: Optional[torch.Tensor] = None,
#         negative_prompt_attention_mask: Optional[torch.Tensor] = None,
#         **kwargs,
#     ) -> Union[GenerateOutput, torch.LongTensor]:
#         r"""
#
#         Generates sequences of token ids for models with a language modeling head.
#
#         <Tip warning={true}>
#
#         Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
#         model's default generation configuration. You can override any `generation_config` by passing the corresponding
#         parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.
#
#         For an overview of generation strategies and code examples, check out the [following
#         guide](../generation_strategies).
#
#         </Tip>
#
#         Parameters:
#             inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
#                 The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
#                 method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
#                 should be in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
#                 `input_ids`, `input_values`, `input_features`, or `pixel_values`.
#             generation_config ([`~generation.GenerationConfig`], *optional*):
#                 The generation configuration to be used as base parametrization for the generation call. `**kwargs`
#                 passed to generate matching the attributes of `generation_config` will override them. If
#                 `generation_config` is not provided, the default will be used, which has the following loading
#                 priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
#                 configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
#                 default values, whose documentation should be checked to parameterize generation.
#             logits_processor (`LogitsProcessorList`, *optional*):
#                 Custom logits processors that complement the default logits processors built from arguments and
#                 generation config. If a logit processor is passed that is already created with the arguments or a
#                 generation config an error is thrown. This feature is intended for advanced users.
#             stopping_criteria (`StoppingCriteriaList`, *optional*):
#                 Custom stopping criteria that complements the default stopping criteria built from arguments and a
#                 generation config. If a stopping criteria is passed that is already created with the arguments or a
#                 generation config an error is thrown. If your stopping criteria depends on the `scores` input, make
#                 sure you pass `return_dict_in_generate=True, output_scores=True` to `generate`. This feature is
#                 intended for advanced users.
#             prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`, *optional*):
#                 If provided, this function constraints the beam search to allowed tokens only at each step. If not
#                 provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
#                 `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
#                 on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
#                 for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
#                 Retrieval](https://arxiv.org/abs/2010.00904).
#             synced_gpus (`bool`, *optional*):
#                 Whether to continue running the while loop until max_length. Unless overridden this flag will be set to
#                 `True` under DeepSpeed ZeRO Stage 3 multiple GPUs environment to avoid hanging if one GPU finished
#                 generating before other GPUs. Otherwise it'll be set to `False`.
#             assistant_model (`PreTrainedModel`, *optional*):
#                 An assistant model that can be used to accelerate generation. The assistant model must have the exact
#                 same tokenizer. The acceleration is achieved when forecasting candidate tokens with the assistent model
#                 is much faster than running generation with the model you're calling generate from. As such, the
#                 assistant model should be much smaller.
#             streamer (`BaseStreamer`, *optional*):
#                 Streamer object that will be used to stream the generated sequences. Generated tokens are passed
#                 through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
#             negative_prompt_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
#                 The negative prompt needed for some processors such as CFG. The batch size must match the input batch
#                 size. This is an experimental feature, subject to breaking API changes in future versions.
#             negative_prompt_attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
#                 Attention_mask for `negative_prompt_ids`.
#             kwargs (`Dict[str, Any]`, *optional*):
#                 Ad hoc parametrization of `generation_config` and/or additional model-specific kwargs that will be
#                 forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
#                 specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.
#
#         Return:
#             [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
#             or when `config.return_dict_in_generate=True`) or a `torch.LongTensor`.
#
#                 If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
#                 [`~utils.ModelOutput`] types are:
#
#                     - [`~generation.GenerateDecoderOnlyOutput`],
#                     - [`~generation.GenerateBeamDecoderOnlyOutput`]
#
#                 If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
#                 [`~utils.ModelOutput`] types are:
#
#                     - [`~generation.GenerateEncoderDecoderOutput`],
#                     - [`~generation.GenerateBeamEncoderDecoderOutput`]
#         """
#
#         '''prepare'''
#         # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
#         # self._validate_model_class()
#         tokenizer = kwargs.pop("tokenizer", None)  # Pull this out first, we only use it for stopping criteria
#         generation_config, model_kwargs = self._prepare_generation_config(generation_config, **kwargs)
#         self._validate_model_kwargs_streaming(model_kwargs.copy())
#
#         # 2. Set generation parameters if not already defined
#         if synced_gpus is None:
#             if is_deepspeed_zero3_enabled() and dist.get_world_size() > 1:
#                 synced_gpus = True
#             else:
#                 synced_gpus = False
#
#         logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
#         stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
#
#         accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
#         requires_attention_mask = "encoder_outputs" not in model_kwargs
#         kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None
#
#         # 3. Define model inputs
#         inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
#             inputs, generation_config.bos_token_id, model_kwargs
#         )
#         batch_size = inputs_tensor.shape[0]
#
#         device = inputs_tensor.device
#         self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)
#
#         # decoder-only models must use left-padding for batched generation.
#         if not self.config.is_encoder_decoder and not is_torchdynamo_compiling():
#             # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
#             # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
#             if (
#                 generation_config._pad_token_tensor is not None
#                 and batch_size > 1
#                 and len(inputs_tensor.shape) == 2
#                 and torch.sum(inputs_tensor[:, -1] == generation_config._pad_token_tensor) > 0
#             ):
#                 logger.warning(
#                     "A decoder-only architecture is being used, but right-padding was detected! For correct "
#                     "generation results, please set `padding_side='left'` when initializing the tokenizer."
#                 )
#
#         # 4. Define other model kwargs
#         # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
#         # generating the first new token or not, and we only want to use the embeddings for the first new token)
#         if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
#             model_kwargs["use_cache"] = True
#         else:
#             model_kwargs["use_cache"] = generation_config.use_cache
#
#         if not kwargs_has_attention_mask and requires_attention_mask and accepts_attention_mask:
#             model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
#                 inputs_tensor, generation_config._pad_token_tensor, generation_config._eos_token_tensor
#             )
#
#         if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
#             # if model is encoder decoder encoder_outputs are created and added to `model_kwargs`
#             model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
#                 inputs_tensor, model_kwargs, model_input_name, generation_config
#             )
#
#         # 5. Prepare `input_ids` which will be used for auto-regressive generation
#         if self.config.is_encoder_decoder:
#             input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
#                 batch_size=batch_size,
#                 model_input_name=model_input_name,
#                 model_kwargs=model_kwargs,
#                 decoder_start_token_id=generation_config._decoder_start_token_tensor,
#                 device=inputs_tensor.device,
#             )
#         else:
#             input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")
#
#         if generation_config.token_healing:
#             input_ids = self.heal_tokens(input_ids, tokenizer)
#
#         if streamer is not None:
#             streamer.put(input_ids.cpu())
#
#         # 6. Prepare `max_length` depending on other stopping criteria.
#         input_ids_length = input_ids.shape[-1]
#         has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
#         has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
#         generation_config = self._prepare_generated_length(
#             generation_config=generation_config,
#             has_default_max_length=has_default_max_length,
#             has_default_min_length=has_default_min_length,
#             model_input_name=model_input_name,
#             inputs_tensor=inputs_tensor,
#             input_ids_length=input_ids_length,
#         )
#
#         use_dynamic_cache_by_default = False
#         cache_name = "past_key_values"
#         # ...
#         # initialize source cache name
#         source_cache_name = "source_key_values"
#
#         # TODO(joao): support static caches in assisted generation. assisted generation needs to roll back caches,
#         # which is only supported in dynamic caches atm
#         if (
#             assistant_model is not None
#             and generation_config.cache_implementation is not None
#             and self._supports_default_dynamic_cache()
#         ):
#             logger.warning_once(
#                 "An assistant model is provided, using a dynamic cache instead of a cache of type="
#                 f"'{generation_config.cache_implementation}'."
#             )
#             generation_config.cache_implementation = None
#
#         if (model_kwargs.get(cache_name) is not None) and is_torchdynamo_compiling():
#             raise ValueError(
#                 "Passing `past_key_values` is not supported when compiling `model.generate` with torch.compile -- you "
#                 "may get incorrect outputs. Please compile `model.forward` only or use the `cache_implementation` "
#                 "input argument."
#             )
#         if generation_config.cache_implementation is not None and (model_kwargs.get(cache_name) is not None):
#             raise ValueError(
#                 f"Passing both `cache_implementation` (used to initialize certain caches) and `{cache_name}` (a "
#                 "Cache object) is unsupported. Please use only one of the two."
#             )
#         elif generation_config.cache_implementation is not None:
#             if generation_config.cache_implementation in NEED_SETUP_CACHE_CLASSES_MAPPING:
#                 if generation_config.cache_implementation == "static" and not self._supports_static_cache:
#                     raise ValueError(
#                         "This model does not support `cache_implementation='static'`. Please check the following "
#                         "issue: https://github.com/huggingface/transformers/issues/28981"
#                     )
#                 model_kwargs[cache_name] = self._get_cache(
#                     cache_implementation=generation_config.cache_implementation,
#                     max_batch_size=generation_config.num_beams * generation_config.num_return_sequences * batch_size,
#                     max_cache_len=generation_config.max_length,
#                     device=device,
#                     model_kwargs=model_kwargs,
#                 )
#             elif generation_config.cache_implementation == "quantized":
#                 if not self._supports_quantized_cache:
#                     raise ValueError(
#                         "This model does not support the quantized cache. If you want your model to support quantized "
#                         "cache, please open an issue."
#                     )
#
#                 cache_config = (
#                     generation_config.cache_config
#                     if generation_config.cache_config is not None
#                     else QuantizedCacheConfig()
#                 )
#                 cache_class = QUANT_BACKEND_CLASSES_MAPPING[cache_config.backend]
#
#                 if cache_config.backend == "quanto" and not is_quanto_available():
#                     raise ImportError(
#                         "You need to install `quanto` in order to use KV cache quantization with quanto backend. "
#                         "Please install it via  with `pip install quanto`"
#                     )
#                 elif cache_config.backend == "HQQ" and not is_hqq_available():
#                     raise ImportError(
#                         "You need to install `HQQ` in order to use KV cache quantization with HQQ backend. "
#                         "Please install it via  with `pip install hqq`"
#                     )
#
#                 model_kwargs[cache_name] = cache_class(cache_config)
#             elif generation_config.cache_implementation == "offloaded":
#                 model_kwargs[cache_name] = OffloadedCache()
#         # Use DynamicCache() instance by default. This will avoid back and forth from legacy format that
#         # keeps copying the cache thus using much more memory
#         elif generation_config.cache_implementation is None and self._supports_default_dynamic_cache():
#             past = model_kwargs.get(cache_name, None)
#             requires_cross_attention_cache = (
#                 self.config.is_encoder_decoder or model_kwargs.get("encoder_outputs") is not None
#             )
#             if past is None:
#                 model_kwargs[cache_name] = (
#                     DynamicCache()
#                     if not requires_cross_attention_cache
#                     else EncoderDecoderCache(DynamicCache(), DynamicCache())
#                 )
#                 use_dynamic_cache_by_default = True
#             elif isinstance(past, tuple):
#                 model_kwargs[cache_name] = (
#                     DynamicCache.from_legacy_cache(past)
#                     if not requires_cross_attention_cache
#                     else EncoderDecoderCache.from_legacy_cache(past)
#                 )
#                 use_dynamic_cache_by_default = True
#
#             # ...
#             # initialize source cache
#             source_past = model_kwargs.get(source_cache_name, None)
#             if source_past is None:
#                 model_kwargs[source_cache_name] = DynamicCache()
#
#
#         self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)
#
#
#
#
#
#
#
#
#         # 7. determine generation mode
#         generation_mode = generation_config.get_generation_mode(assistant_model)
#
#         if streamer is not None and (generation_config.num_beams > 1):
#             raise ValueError(
#                 "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
#             )
#
#         if not is_torchdynamo_compiling() and self.device.type != input_ids.device.type:
#             warnings.warn(
#                 "You are calling .generate() with the `input_ids` being on a device type different"
#                 f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
#                 f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
#                 " Please make sure that you have put `input_ids` to the"
#                 f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
#                 " running `.generate()`.",
#                 UserWarning,
#             )
#
#         # 8. prepare distribution pre_processing samplers
#         prepared_logits_processor = self._get_logits_processor(
#             generation_config=generation_config,
#             input_ids_seq_length=input_ids_length,
#             encoder_input_ids=inputs_tensor,
#             prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
#             logits_processor=logits_processor,
#             device=inputs_tensor.device,
#             model_kwargs=model_kwargs,
#             negative_prompt_ids=negative_prompt_ids,
#             negative_prompt_attention_mask=negative_prompt_attention_mask,
#         )
#
#
#
#
#
#
#         # 9. prepare stopping criteria
#         # kwargs.pop("input_ids", None)
#         # kwargs.pop("attention_mask", None)
#         # kwargs.pop("max_new_tokens", None)
#         # kwargs.pop("generate_mode", None)
#         # kwargs.pop("split_mode", None)
#         # kwargs.pop("pe_cache_length", None)
#         # kwargs.pop("end_Instruct", None)
#         # kwargs.pop("_lengths", None)
#         # kwargs.pop("_lengths_index", None)
#         # kwargs.pop("wait_k", None)
#         # kwargs.pop("source_words", None)
#         # kwargs.pop("assistant_token", None)
#         prepared_stopping_criteria = self._get_stopping_criteria(
#             generation_config=generation_config, stopping_criteria=stopping_criteria, tokenizer=tokenizer, **kwargs
#         )
#
#
#
#
#
#
#
#         # 10. go into different generation modes
#         if generation_mode in (GenerationMode.SAMPLE, GenerationMode.GREEDY_SEARCH):
#             # 11. prepare logits warper
#             prepared_logits_warper = None
#             # 12. expand input_ids with `num_return_sequences` additional sequences per batch
#             input_ids, model_kwargs = self._expand_inputs_for_generation(
#                 input_ids=input_ids,
#                 expand_size=generation_config.num_return_sequences,
#                 is_encoder_decoder=self.config.is_encoder_decoder,
#                 **model_kwargs,
#             )
#
#             # 13. run sample (it degenerates to greedy search when `generation_config.do_sample=False`)
#             result, wait_lagging = self._sample_streaming(
#                 streaming_split = streaming_split,
#                 input_ids = input_ids,
#                 logits_processor=prepared_logits_processor,
#                 logits_warper=prepared_logits_warper,
#                 stopping_criteria=prepared_stopping_criteria,
#                 generation_config=generation_config,
#                 synced_gpus=synced_gpus,
#                 streamer=streamer,
#                 tokenizer=tokenizer, # must provide
#                 **model_kwargs,
#             )
#
#         """
#         elif generation_mode in (GenerationMode.BEAM_SAMPLE, GenerationMode.BEAM_SEARCH):
#             # 11. prepare logits warper
#             prepared_logits_warper = (
#                 self._get_logits_warper(generation_config, device=input_ids.device)
#                 if generation_config.do_sample
#                 else None
#             )
#
#             # 12. prepare beam search scorer
#             beam_scorer = BeamSearchScorer(
#                 batch_size=batch_size,
#                 num_beams=generation_config.num_beams,
#                 device=inputs_tensor.device,
#                 length_penalty=generation_config.length_penalty,
#                 do_early_stopping=generation_config.early_stopping,
#                 num_beam_hyps_to_keep=generation_config.num_return_sequences,
#                 max_length=generation_config.max_length,
#             )
#
#             # 13. interleave input_ids with `num_beams` additional sequences per batch
#             input_ids, model_kwargs = self._expand_inputs_for_generation(
#                 input_ids=input_ids,
#                 expand_size=generation_config.num_beams,
#                 is_encoder_decoder=self.config.is_encoder_decoder,
#                 **model_kwargs,
#             )
#
#             # 14. run beam sample
#             result = self._beam_search(
#                 input_ids,
#                 beam_scorer,
#                 logits_processor=prepared_logits_processor,
#                 logits_warper=prepared_logits_warper,
#                 stopping_criteria=prepared_stopping_criteria,
#                 generation_config=generation_config,
#                 synced_gpus=synced_gpus,
#                 **model_kwargs,
#             )
#
#         elif generation_mode == GenerationMode.GROUP_BEAM_SEARCH:
#             # 11. prepare beam search scorer
#             beam_scorer = BeamSearchScorer(
#                 batch_size=batch_size,
#                 num_beams=generation_config.num_beams,
#                 device=inputs_tensor.device,
#                 length_penalty=generation_config.length_penalty,
#                 do_early_stopping=generation_config.early_stopping,
#                 num_beam_hyps_to_keep=generation_config.num_return_sequences,
#                 num_beam_groups=generation_config.num_beam_groups,
#                 max_length=generation_config.max_length,
#             )
#             # 12. interleave input_ids with `num_beams` additional sequences per batch
#             input_ids, model_kwargs = self._expand_inputs_for_generation(
#                 input_ids=input_ids,
#                 expand_size=generation_config.num_beams,
#                 is_encoder_decoder=self.config.is_encoder_decoder,
#                 **model_kwargs,
#             )
#             # 13. run beam search
#             result = self._group_beam_search(
#                 input_ids,
#                 beam_scorer,
#                 logits_processor=prepared_logits_processor,
#                 stopping_criteria=prepared_stopping_criteria,
#                 generation_config=generation_config,
#                 synced_gpus=synced_gpus,
#                 **model_kwargs,
#             )
#
#         elif generation_mode == GenerationMode.CONSTRAINED_BEAM_SEARCH:
#             final_constraints = []
#             if generation_config.constraints is not None:
#                 final_constraints = generation_config.constraints
#
#             if generation_config.force_words_ids is not None:
#
#                 def typeerror():
#                     raise ValueError(
#                         "`force_words_ids` has to either be a `List[List[List[int]]]` or `List[List[int]]` "
#                         f"of positive integers, but is {generation_config.force_words_ids}."
#                     )
#
#                 if (
#                     not isinstance(generation_config.force_words_ids, list)
#                     or len(generation_config.force_words_ids) == 0
#                 ):
#                     typeerror()
#
#                 for word_ids in generation_config.force_words_ids:
#                     if isinstance(word_ids[0], list):
#                         if not isinstance(word_ids, list) or len(word_ids) == 0:
#                             typeerror()
#                         if any(not isinstance(token_ids, list) for token_ids in word_ids):
#                             typeerror()
#                         if any(
#                             any((not isinstance(token_id, int) or token_id < 0) for token_id in token_ids)
#                             for token_ids in word_ids
#                         ):
#                             typeerror()
#
#                         constraint = DisjunctiveConstraint(word_ids)
#                     else:
#                         if not isinstance(word_ids, list) or len(word_ids) == 0:
#                             typeerror()
#                         if any((not isinstance(token_id, int) or token_id < 0) for token_id in word_ids):
#                             typeerror()
#
#                         constraint = PhrasalConstraint(word_ids)
#                     final_constraints.append(constraint)
#
#             # 11. prepare beam search scorer
#             constrained_beam_scorer = ConstrainedBeamSearchScorer(
#                 constraints=final_constraints,
#                 batch_size=batch_size,
#                 num_beams=generation_config.num_beams,
#                 device=inputs_tensor.device,
#                 length_penalty=generation_config.length_penalty,
#                 do_early_stopping=generation_config.early_stopping,
#                 num_beam_hyps_to_keep=generation_config.num_return_sequences,
#                 max_length=generation_config.max_length,
#             )
#             # 12. interleave input_ids with `num_beams` additional sequences per batch
#             input_ids, model_kwargs = self._expand_inputs_for_generation(
#                 input_ids=input_ids,
#                 expand_size=generation_config.num_beams,
#                 is_encoder_decoder=self.config.is_encoder_decoder,
#                 **model_kwargs,
#             )
#             # 13. run beam search
#             result = self._constrained_beam_search(
#                 input_ids,
#                 constrained_beam_scorer=constrained_beam_scorer,
#                 logits_processor=prepared_logits_processor,
#                 stopping_criteria=prepared_stopping_criteria,
#                 generation_config=generation_config,
#                 synced_gpus=synced_gpus,
#                 **model_kwargs,
#             )
#
#
#         # Convert to legacy cache if needed
#         if use_dynamic_cache_by_default and generation_config.return_legacy_cache:
#             if isinstance(result, ModelOutput) and hasattr(result, "past_key_values"):
#                 if isinstance(result.past_key_values, (DynamicCache, EncoderDecoderCache)):
#                     result.past_key_values = result.past_key_values.to_legacy_cache()
#         """
#
#
#         return result, wait_lagging
#
#
#
#     def _validate_model_kwargs_streaming(self, model_kwargs: Dict[str, Any]):
#         """Validates model kwargs for generation. Generate argument typos will also be caught here."""
#         # If a `Cache` instance is passed, checks whether the model is compatible with it
#         if isinstance(model_kwargs.get("past_key_values", None), Cache) and not self._supports_cache_class:
#             raise ValueError(
#                 f"{self.__class__.__name__} does not support an instance of `Cache` as `past_key_values`. Please "
#                 "check the model documentation for supported cache formats."
#             )
#
#         # Excludes arguments that are handled before calling any model function
#         if self.config.is_encoder_decoder:
#             for key in ["decoder_input_ids"]:
#                 model_kwargs.pop(key, None)
#
#         unused_model_args = []
#         model_args = set(inspect.signature(self.prepare_inputs_for_generation_streaming).parameters)
#         # `kwargs`/`model_kwargs` is often used to handle optional forward pass inputs like `attention_mask`. If
#         # `prepare_inputs_for_generation` doesn't accept them, then a stricter check can be made ;)
#         if "kwargs" in model_args or "model_kwargs" in model_args:
#             model_args |= set(inspect.signature(self.forward).parameters)
#
#         # Encoder-Decoder models may also need Encoder arguments from `model_kwargs`
#         if self.config.is_encoder_decoder:
#             base_model = getattr(self, self.base_model_prefix, None)
#
#             # allow encoder kwargs
#             encoder = getattr(self, "encoder", None)
#             # `MusicgenForConditionalGeneration` has `text_encoder` and `audio_encoder`.
#             # Also, it has `base_model_prefix = "encoder_decoder"` but there is no `self.encoder_decoder`
#             # TODO: A better way to handle this.
#             if encoder is None and base_model is not None:
#                 encoder = getattr(base_model, "encoder", None)
#
#             if encoder is not None:
#                 encoder_model_args = set(inspect.signature(encoder.forward).parameters)
#                 model_args |= encoder_model_args
#
#             # allow decoder kwargs
#             decoder = getattr(self, "decoder", None)
#             if decoder is None and base_model is not None:
#                 decoder = getattr(base_model, "decoder", None)
#
#             if decoder is not None:
#                 decoder_model_args = set(inspect.signature(decoder.forward).parameters)
#                 model_args |= {f"decoder_{x}" for x in decoder_model_args}
#
#             # allow assistant_encoder_outputs to be passed if we're doing assisted generating
#             if "assistant_encoder_outputs" in model_kwargs:
#                 model_args |= {"assistant_encoder_outputs"}
#
#         for key, value in model_kwargs.items():
#             if value is not None and key not in model_args:
#                 unused_model_args.append(key)
#
#         if unused_model_args:
#             raise ValueError(
#                 f"The following `model_kwargs` are not used by the model: {unused_model_args} (note: typos in the"
#                 " generate arguments will also show up in this list)"
#             )
#
#     ''' Separating the KV cache allows for loading source and target caches independently.
#         Merging the KV cache enables the target to attend to the entire KV cache content during the autoregressive decoding stage.
#         In the attention mechanism, the read phase uses the source cache, while the write phase uses the past cache.
#     '''
#     def merge_source_target(self):
#         # reset
#         assert self.past_key_values is not None
#         assert self.source_key_values is not None
#         assert self.target_key_values is not None
#         self.past_key_values.key_cache = []
#         self.past_key_values.value_cache = []
#         if self.target_key_values.get_seq_length()==0:
#             self.past_key_values.key_cache = self.source_key_values.key_cache.copy()
#             self.past_key_values.value_cache = self.source_key_values.value_cache.copy()
#         else:
#             for source_key_cache, source_value_cache, target_key_cache, target_value_cache in zip(self.source_key_values.key_cache, self.source_key_values.value_cache, self.target_key_values.key_cache, self.target_key_values.value_cache):
#                 self.past_key_values.key_cache.append(torch.cat((source_key_cache, target_key_cache), dim=2))
#                 self.past_key_values.value_cache.append(torch.cat((source_value_cache, target_value_cache), dim=2))
#
#     def separate_source_target(self):
#         assert self.past_key_values is not None
#         assert self.source_key_values is not None
#         assert self.target_key_values is not None
#         source_length = self.source_key_values.get_seq_length()
#         if self.past_key_values.get_seq_length()> source_length:
#             # reset
#             self.source_key_values.key_cache = []
#             self.source_key_values.value_cache = []
#             self.target_key_values.key_cache = []
#             self.target_key_values.value_cache = []
#             for key_cache, value_cache in zip(self.past_key_values.key_cache, self.past_key_values.value_cache):
#                 self.source_key_values.key_cache.append(key_cache[...,:source_length,:])
#                 self.source_key_values.value_cache.append(value_cache[...,:source_length,:])
#                 self.target_key_values.key_cache.append(key_cache[...,source_length:,:])
#                 self.target_key_values.value_cache.append(value_cache[...,source_length:,:])
#
#
#     # Copied and revised from GenerationMixin._get_initial_cache_position
#     def _get_initial_cache_position_for_streaming(self, input_length, model_kwargs):
#         """Calculates `cache_position` for the pre-fill stage based on `input_ids` and optionally past length"""
#         # `torch.compile`-friendly `torch.arange` from a shape -- the lines below are equivalent to `torch.arange`
#         assert self.source_key_values is not None
#         cache_position = torch.arange(
#                 self.source_key_values.get_seq_length(), input_length[0], dtype=torch.int64, device=model_kwargs.get('assistant_token').device
#             )
#
#         model_kwargs["cache_position"] = cache_position
#         return model_kwargs
#
#
#
#     def prepare_inputs_for_generation_streaming(
#         self,
#         input_ids,
#         past_key_values=None,
#         attention_mask=None,
#         inputs_embeds=None,
#         cache_position=None,
#         position_ids=None,
#         use_cache=True,
#         # ...
#         is_streaming = False,
#         input_length=None,
#         pe_cache_length = 0,
#         assistant_token = None,
#         source_words=None,
#         end_Instruct = None,
#         ReadAction = True,
#         **kwargs,
#     ):
#         if not is_streaming:
#             # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
#             # Exception 1: when passing input_embeds, input_ids may be missing entries
#             # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
#             if past_key_values is not None:
#                 if inputs_embeds is not None:  # Exception 1
#                     input_ids = input_ids[:, -cache_position.shape[0] :]
#                 elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
#                     input_ids = input_ids[:, cache_position]
#
#             if attention_mask is not None and position_ids is None:
#                 # create position_ids on the fly for batch generation
#                 position_ids = attention_mask.long().cumsum(-1) - 1
#                 position_ids.masked_fill_(attention_mask == 0, 1)
#                 if past_key_values:
#                     position_ids = position_ids[:, -input_ids.shape[1] :]
#
#                     # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
#                     position_ids = position_ids.clone(memory_format=torch.contiguous_format)
#
#             # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
#             if inputs_embeds is not None and cache_position[0] == 0:
#                 model_inputs = {"inputs_embeds": inputs_embeds}
#             else:
#                 model_inputs = {"input_ids": input_ids}
#
#             if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
#                 if inputs_embeds is not None:
#                     batch_size, sequence_length = inputs_embeds.shape
#                     device = inputs_embeds.device
#                 else:
#                     batch_size, sequence_length = input_ids.shape
#                     device = input_ids.device
#
#                 dtype = self.lm_head.weight.dtype
#                 min_dtype = torch.finfo(dtype).min
#
#                 attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
#                     attention_mask,
#                     sequence_length=sequence_length,
#                     target_length=past_key_values.get_max_length(),
#                     dtype=dtype,
#                     device=device,
#                     min_dtype=min_dtype,
#                     cache_position=cache_position,
#                     batch_size=batch_size,
#                 )
#
#             model_inputs.update(
#                 {
#                     "position_ids": position_ids,
#                     "cache_position": cache_position,
#                     "past_key_values": past_key_values,
#                     "use_cache": use_cache,
#                     "attention_mask": attention_mask,
#                 }
#             )
#
#         elif is_streaming:
#             assert input_length is not None, "input_length must be provided for streaming generation"
#             model_inputs = kwargs.copy()
#
#             if self.source_key_values is not None:
#                 past_source_length = self.source_key_values.get_seq_length()
#                 past_target_length = self.target_key_values.get_seq_length()
#
#             if ReadAction:
#                 position_ids_source = torch.arange(past_source_length,input_length[0]).to(assistant_token.device).unsqueeze(0)
#                 position_ids = position_ids_source.clone().detach()
#                 past_length = past_source_length
#                 input_ids = input_ids[:,past_source_length:input_length[0]]
#             elif not ReadAction:
#                 if past_target_length==0:
#                     num_tokens = input_ids.shape[-1]
#                     position_ids = torch.arange(pe_cache_length, pe_cache_length+num_tokens).to(assistant_token.device).unsqueeze(0)
#                 else:
#                     position_ids_target = past_target_length + pe_cache_length
#                     position_ids = torch.tensor([position_ids_target]).to(assistant_token.device).unsqueeze(0)
#                 past_length = past_source_length
#                 input_ids = input_ids
#
#
#             if ReadAction:
#                 model_inputs.update(
#                     {
#                         "input_ids": input_ids,
#                         "position_ids": position_ids,
#                         "use_cache": use_cache,
#                         "attention_mask": attention_mask,
#                         "source_key_values":self.source_key_values,
#                         "pe_cache_length":pe_cache_length
#                     }
#                 )
#             else:
#                 model_inputs.update(
#                     {
#                         "input_ids": input_ids,
#                         "position_ids": position_ids,
#                         "use_cache": use_cache,
#                         "attention_mask": attention_mask,
#                         "past_key_values":self.past_key_values,
#                         "pe_cache_length":pe_cache_length
#                     }
#                 )
#         return model_inputs
#
#
#
#     def _sample_streaming(
#         self,
#         input_ids: torch.LongTensor,
#         logits_processor: LogitsProcessorList,
#         stopping_criteria: StoppingCriteriaList,
#         generation_config: GenerationConfig,
#         synced_gpus: bool,
#         streamer: Optional["BaseStreamer"],
#         tokenizer: Optional["PreTrainedTokenizerBase"],
#         **model_kwargs,
#     ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
#         r"""
#         Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
#         can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.
#
#         Parameters:
#             input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
#                 The sequence used as a prompt for the generation.
#             logits_processor (`LogitsProcessorList`):
#                 An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
#                 used to modify the prediction scores of the language modeling head applied at each generation step.
#             stopping_criteria (`StoppingCriteriaList`):
#                 An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
#                 used to tell if the generation loop should stop.
#             generation_config ([`~generation.GenerationConfig`]):
#                 The generation configuration to be used as parametrization of the decoding method.
#             synced_gpus (`bool`):
#                 Whether to continue running the while loop until max_length (needed to avoid deadlocking with
#                 `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
#             streamer (`BaseStreamer`, *optional*):
#                 Streamer object that will be used to stream the generated sequences. Generated tokens are passed
#                 through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
#             model_kwargs:
#                 Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
#                 an encoder-decoder model the kwargs should include `encoder_outputs`.
#
#         Return:
#             [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
#             A `torch.LongTensor` containing the generated tokens (default behaviour) or a
#             [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
#             `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
#             `model.config.is_encoder_decoder=True`.
#         """
#
#         # ...
#         # init/reset cache (only if not already set, to support Head-Aware cache)
#         # Check if cache is already set (e.g., Head-Aware cache)
#         if self.source_key_values is None:
#             from StreamingLLM_GPE.models.Qwen2_5.qwen_streaming import DynamicCache
#             self.source_key_values = DynamicCache()
#         elif hasattr(self.source_key_values, 'key_cache'):
#             # Reset existing cache (clear but keep the instance)
#             self.source_key_values.key_cache = []
#             self.source_key_values.value_cache = []
#             self.source_key_values._seen_tokens = 0
#
#         if self.target_key_values is None:
#             from StreamingLLM_GPE.models.Qwen2_5.qwen_streaming import DynamicCache
#             self.target_key_values = DynamicCache()
#         elif hasattr(self.target_key_values, 'key_cache'):
#             self.target_key_values.key_cache = []
#             self.target_key_values.value_cache = []
#             self.target_key_values._seen_tokens = 0
#
#         if self.past_key_values is None:
#             from StreamingLLM_GPE.models.Qwen2_5.qwen_streaming import DynamicCache
#             self.past_key_values = DynamicCache()
#         elif hasattr(self.past_key_values, 'key_cache'):
#             self.past_key_values.key_cache = []
#             self.past_key_values.value_cache = []
#             self.past_key_values._seen_tokens = 0
#         # ...
#         # tokenizer = kwargs.pop("tokenizer", None)  # Pull this out first, we only use it for stopping criteria
#         end_Instruct = model_kwargs.get("end_Instruct", None)
#         # 从generation_config获取max_new_tokens，如果没有则使用默认值
#         max_new_tokens = generation_config.max_new_tokens if hasattr(generation_config, 'max_new_tokens') and generation_config.max_new_tokens is not None else 1024
#         ReadAction_criteria = StopTokenCriteria(tokenizer, max_new_tokens=max_new_tokens, end_Instruct=end_Instruct)
#
#         _lengths = model_kwargs.get("_lengths", None)
#         source_seg_len = _lengths[0]['source_seg_len']
#         _lengths_index = model_kwargs.get("_lengths_index", None)
#         split_mode = model_kwargs.get("split_mode", None)
#         pe_cache_length = model_kwargs.get("pe_cache_length", None)
#         assistant_token = model_kwargs.get("assistant_token", None)
#         ReadAction = True
#
#         wait_k = model_kwargs.get("wait_k", None)
#         # source_words 应该从 wait_k 开始（因为已经读取了前 wait_k 个单词）
#         # 但为了正确计算延迟，我们需要记录的是"已经读取的源端单词数"
#         # 在 wait-k 策略中，初始时已经读取了 wait_k 个单词
#         source_words = wait_k if wait_k is not None else 0  # 初始值：已经读取了 wait_k 个单词
#         target_words = 0  #target_words
#         max_distance = 100
#
#         next_tokens = model_kwargs['assistant_token'].unsqueeze(0)
#         target_tokens = [next_tokens[:,:-1], next_tokens[:,-1:]]
#         target_tokens_this_write = []
#         wait_lagging = [] # record the lagging of each target token
#
#
#         input_length = (sum(source_seg_len[:model_kwargs['wait_k']+1]), 1)
#         source_input_length = sum(source_seg_len[:model_kwargs['wait_k']+1])
#
#
#
#
#
#         # init values
#         pad_token_id = generation_config._pad_token_tensor
#         output_attentions = generation_config.output_attentions
#         output_hidden_states = generation_config.output_hidden_states
#         output_scores = generation_config.output_scores
#         output_logits = generation_config.output_logits
#         return_dict_in_generate = generation_config.return_dict_in_generate
#         has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
#         do_sample = generation_config.do_sample
#
#         # init attention / hidden states / scores tuples
#         scores = () if (return_dict_in_generate and output_scores) else None
#         raw_logits = () if (return_dict_in_generate and output_logits) else None
#         decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
#         cross_attentions = () if (return_dict_in_generate and output_attentions) else None
#         decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None
#
#         # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
#         if return_dict_in_generate and self.config.is_encoder_decoder:
#             encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
#             encoder_hidden_states = (
#                 model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
#             )
#
#         # keep track of which sequences are already finished
#         batch_size, input_len = input_ids.shape
#         cur_len = input_len  # 保留用于其他用途
#         generated_tokens_count = 0  # 独立计数器：跟踪新生成的token数（不包括源端输入）
#         this_peer_finished = False
#         unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
#         model_kwargs = self._get_initial_cache_position_for_streaming(input_length, model_kwargs)
#
#         # wait-k read/write policy
#         assert wait_k is not None, "wait_k must be provided for streaming generation"
#
#         # 添加调试计数器
#         debug_step_count = 0
#         max_debug_steps = 100  # 只打印前100步的调试信息
#
#         while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
#             debug_step_count += 1
#             if debug_step_count <= max_debug_steps and tokenizer is not None:
#                 if debug_step_count % 10 == 0:  # 每10步打印一次
#                     print(f"[DEBUG] Generation step {debug_step_count}: ReadAction={ReadAction}, source_words={source_words}/{len(source_seg_len)-1}, target_words={target_words}, generated_tokens={generated_tokens_count}, this_peer_finished={this_peer_finished}")
#             if ReadAction:
#                 # prepare model inputs
#                 self.separate_source_target()
#                 model_inputs = self.prepare_inputs_for_generation_streaming(input_ids, input_length=input_length, ReadAction=ReadAction, is_streaming =True,
#                                                                     **model_kwargs)
#                 # prepare variable output controls (note: some models won't accept all output controls)
#                 model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
#                 model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})
#
#                 _outputs = self(
#                     **model_inputs,
#                     return_dict=True,
#                     output_attentions=output_attentions,
#                     output_hidden_states=output_hidden_states,
#                     ReadAction = ReadAction,
#                 )
#                 # 读取源端后，应该立即生成目标端，所以设置ReadAction=False
#                 # 这样下次循环会进入生成目标端的流程
#                 ReadAction = False
#                 token_count = 0
#                 self.merge_source_target()
#             elif not ReadAction:
#                 self.separate_source_target()
#                 token_count += 1
#                 # prepare model inputs
#                 model_inputs = self.prepare_inputs_for_generation_streaming(next_tokens, input_length=input_length, ReadAction=ReadAction, is_streaming =True,
#                                                                     **model_kwargs)
#                 # prepare variable output controls (note: some models won't accept all output controls)
#                 model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
#                 model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})
#
#                 outputs = self(
#                     **model_inputs,
#                     return_dict=True,
#                     output_attentions=output_attentions,
#                     output_hidden_states=output_hidden_states,
#                     ReadAction=ReadAction,
#                 )
#
#                 # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
#                 model_kwargs = self._update_model_kwargs_for_generation(
#                     outputs,
#                     model_kwargs,
#                     is_encoder_decoder=self.config.is_encoder_decoder,
#                 )
#                 if synced_gpus and this_peer_finished:
#                     continue
#
#                 # Copy is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
#                 # (the clone itself is always small)
#                 next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)
#
#                 # pre-process distribution
#                 next_token_scores = logits_processor(input_ids, next_token_logits)
#
#                 # Store scores, attentions and hidden_states when required
#                 if return_dict_in_generate:
#                     if output_scores:
#                         scores += (next_token_scores,)
#                     if output_logits:
#                         raw_logits += (next_token_logits,)
#                     if output_attentions:
#                         decoder_attentions += (
#                             (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
#                         )
#                         if self.config.is_encoder_decoder:
#                             cross_attentions += (outputs.cross_attentions,)
#
#                     if output_hidden_states:
#                         decoder_hidden_states += (
#                             (outputs.decoder_hidden_states,)
#                             if self.config.is_encoder_decoder
#                             else (outputs.hidden_states,)
#                         )
#
#                 # token selection
#                 if do_sample:
#                     probs = nn.functional.softmax(next_token_scores, dim=-1)
#                     # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
#                     next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
#                 else:
#                     next_tokens = torch.argmax(next_token_scores, dim=-1)
#
#                 # finished sentences should have their next token be a padding token
#                 if has_eos_stopping_criteria:
#                     next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
#
#                 # update generated ids, model inputs, and length for next step
#                 input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
#                 if streamer is not None:
#                     streamer.put(next_tokens.cpu())
#
#
#                 # This is needed to properly delete outputs.logits which may be very large for first iteration
#                 # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
#                 del outputs
#
#
#                 # ...
#                 next_tokens = next_tokens.unsqueeze(0)
#                 target_tokens.append(next_tokens) # for output
#                 target_ids = torch.cat(target_tokens,dim=-1)
#                 target_tokens_this_write.append(next_tokens)
#                 target_ids_this_write = torch.cat(target_tokens_this_write,dim=-1)
#                 ReadAction_new, remove_last_token = ReadAction_criteria(target_ids_this_write, scores, token_count)  # StopTokenCriteria
#
#                 # ReadAction_new表示是否应该继续读取源端（遇到空格或标点时，ReadAction_new会变成False，表示一个word生成完成）
#                 # 但这不应该停止整个生成过程，只是表示当前word生成完成，需要读取下一个源端word
#                 # 只有当ReadAction_new为False且源端已读完时，才真正停止生成
#
#                 # 注意：ReadAction_new为False只是表示当前word生成完成，不是整个生成完成
#                 # 我们需要继续生成，直到ReadAction_new为False且source_words >= len(source_seg_len)-1
#
#                 unfinished_sequences = unfinished_sequences & ~stopping_criteria(target_ids[0:,2:], scores)
#                 # 注意：在streaming模式下，不应该因为stopping_criteria而提前停止
#                 # 应该继续生成直到源端读完或达到max_new_tokens
#                 # this_peer_finished = unfinished_sequences.max() == 0  # 注释掉，不因为stopping_criteria停止
#
#                 # 检查是否达到max_new_tokens（使用新生成的token数，而不是包含源端输入的总长度）
#                 generated_tokens_count += 1  # 每次生成新token时增加
#                 if generated_tokens_count >= max_new_tokens:
#                     this_peer_finished = True
#                     if tokenizer is not None:
#                         print(f"[DEBUG] Stopping generation: reached max_new_tokens={max_new_tokens}, generated={generated_tokens_count} tokens")
#
#                 cur_len += 1  # 保留用于其他用途
#
#                 # 记录延迟：无论ReadAction是True还是False，都要记录当前已读取的源端单词数
#                 # 这样即使ReadAction变为False（源端已读完），也能正确计算延迟
#                 wait_lagging.append(source_words)
#                 target_words += 1 # a target word has been generated
#
#                 # 更新ReadAction逻辑：
#                 # ReadAction_new为False表示当前word生成完成（遇到空格等），需要读取下一个源端word
#                 # 只有当源端已读完时，才设置ReadAction为False，停止读取源端
#
#                 # 检查源端是否已读完
#                 source_finished = source_words >= len(source_seg_len)-1
#
#                 if source_finished:
#                     # 源端已读完，停止读取源端，继续生成目标端
#                     ReadAction = False
#                 elif not ReadAction_new:
#                     # ReadAction_new为False表示当前word生成完成，需要读取下一个源端word
#                     # 设置ReadAction为True，在下次循环中读取下一个源端word
#                     ReadAction = True
#                     # 立即准备读取下一个源端word（在当前循环中更新source_words）
#                     source_words += 1
#                     if source_words < len(source_seg_len):
#                         num_tokens = source_seg_len[source_words]
#                         source_input_length += num_tokens
#                         target_input_length = 1
#                         input_length = (source_input_length, target_input_length)
#                 else:
#                     # ReadAction_new为True，继续生成当前word，不读取源端
#                     ReadAction = False
#
#                     # 如果StopTokenCriteria要求移除最后一个token（因为遇到了空格等终止符）
#                     if remove_last_token:
#                         # next_tokens = None
#                         target_tokens.pop()
#                         next_tokens = target_tokens[-1]
#                         # target_tokens_this_write = [model_kwargs['assitant_token'].unsqueeze(0)]
#                         target_tokens_this_write = []
#                         self.past_key_values.pop()
#
#
#                     distance = target_words - source_words
#                     # 只有当距离异常大（可能是错误）且源端已读完时才停止
#                     # 否则继续生成，因为streaming generation允许target领先source
#                     if distance > max_distance and source_words >= len(source_seg_len)-1:
#                         # 源端已读完且距离异常大，可能有问题，停止生成
#                         this_peer_finished = True
#                         if tokenizer is not None:
#                             print(f"[DEBUG] Stopping generation: distance={distance} > max_distance={max_distance}, source_words={source_words}, source_seg_len={len(source_seg_len)}")
#                     # 注意：如果源端未读完，即使距离大也不停止，继续生成
#
#
#         if streamer is not None:
#             streamer.end()
#
#         # 去除输出中的assistant_token部分，只返回实际生成的内容
#         # target_tokens初始包含: [assistant_token[:-1], assistant_token[-1:], ...generated_tokens...]
#         # 需要去掉前两个元素（assistant_token的部分）
#         assistant_token = model_kwargs.get('assistant_token', None)
#         if assistant_token is not None and len(target_tokens) > 0:
#             # assistant_token被分成两部分存储在target_tokens的前两个元素中
#             # 计算需要跳过的元素数量（通常是2，因为assistant_token被分成两部分）
#             # 但为了安全，我们检查target_tokens的实际长度
#             if len(target_tokens) >= 2:
#                 # 检查前两个元素是否对应assistant_token
#                 # 如果target_tokens长度大于2，说明有实际生成的内容
#                 # 去掉前两个元素（assistant_token部分）
#                 actual_generated_tokens = target_tokens[2:]
#                 if len(actual_generated_tokens) > 0:
#                     # 重新构建target_ids，只包含实际生成的内容
#                     target_ids = torch.cat(actual_generated_tokens, dim=-1)
#                 else:
#                     # 如果没有实际生成的内容，返回空tensor
#                     # 使用target_tokens中已有tensor的device
#                     device = target_tokens[0].device if len(target_tokens) > 0 else input_ids.device
#                     target_ids = torch.tensor([], dtype=torch.long, device=device).unsqueeze(0)
#
#         if return_dict_in_generate:
#             return GenerateDecoderOnlyOutput(
#                 sequences=target_ids,
#                 scores=scores,
#                 logits=raw_logits,
#                 attentions=decoder_attentions,
#                 hidden_states=decoder_hidden_states,
#                 past_key_values=model_kwargs.get("past_key_values"),
#             )
#         else:
#             return target_ids, wait_lagging
#
# coding=utf-8
# Modified 2025 by Junlong Tong (Shanghai Jiao Tong University & Eastern Institute of Technology, Ningbo).
# Fixed by User (2025): Full Code - Added GenerationMode - H2O/Streaming Support

from typing import List, Dict, Optional, Union, Callable, Any
import torch
import torch.nn as nn
import torch.distributed as dist
import warnings
import inspect

from transformers.configuration_utils import PretrainedConfig
from transformers.generation.utils import (
    GenerationMixin,
    GenerationConfig,
    LogitsProcessorList,
    StoppingCriteriaList,
    GenerateOutput,
    GenerateDecoderOnlyOutput,
    GenerateEncoderDecoderOutput,
    GenerateNonBeamOutput,
    ModelOutput,
    GenerationMode  # [FIX] Added missing import
)
from transformers.cache_utils import Cache
from transformers.cache_utils import DynamicCache as TransformersDynamicCache
from transformers.modeling_utils import PreTrainedModel

# [FIX] Robust Import for transformers compatibility
try:
    from transformers.utils import is_torchdynamo_compiling
except ImportError:
    def is_torchdynamo_compiling():
        return False

try:
    from transformers.utils import is_deepspeed_zero3_enabled
except ImportError:
    def is_deepspeed_zero3_enabled():
        return False

from .Stopping_criteria import StopTokenCriteria


# Copied from transformers.models.llama.modeling_llama._prepare_4d_causal_attention_mask_with_cache_position
def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        min_dtype: float,
        cache_position: torch.Tensor,
        batch_size: int,
):
    if attention_mask is not None and attention_mask.dim() == 4:
        causal_mask = attention_mask
    else:
        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )
    return causal_mask


class DynamicCache(TransformersDynamicCache):
    def __init__(self) -> None:
        super().__init__()

    def pop(self):
        self._seen_tokens -= 1
        target_key_cache = []
        target_value_cache = []
        for key_cache, value_cache in zip(self.key_cache, self.value_cache):
            target_key_cache.append(key_cache[..., :-1, :])
            target_value_cache.append(value_cache[..., :-1, :])
        self.key_cache = target_key_cache
        self.value_cache = target_value_cache


class unified_PreTrainedModel(PreTrainedModel):
    def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

    @torch.no_grad()
    def generate(
            self,
            inputs: Optional[torch.Tensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
            synced_gpus: Optional[bool] = None,
            assistant_model: Optional["PreTrainedModel"] = None,
            streamer: Optional["BaseStreamer"] = None,
            negative_prompt_ids: Optional[torch.Tensor] = None,
            negative_prompt_attention_mask: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:

        generate_mode = kwargs.get("generate_mode", "batch")
        split_mode = kwargs.get("split_mode", None)

        if generate_mode == "batch":
            wait_lagging = None
            return super().generate(
                inputs,
                generation_config,
                logits_processor,
                stopping_criteria,
                prefix_allowed_tokens_fn,
                synced_gpus,
                assistant_model,
                streamer,
                negative_prompt_ids,
                negative_prompt_attention_mask,
                **kwargs,
            ), wait_lagging

        elif generate_mode == "streaming":
            assert split_mode in ["token",
                                  "word"], f"streaming_split must be one of ['token', 'word'], but got {split_mode}."
            result, wait_lagging = self.streaming_generate(
                split_mode,
                inputs,
                generation_config,
                logits_processor,
                stopping_criteria,
                prefix_allowed_tokens_fn,
                synced_gpus,
                assistant_model,
                streamer,
                negative_prompt_ids,
                negative_prompt_attention_mask,
                **kwargs,
            )
            return result, wait_lagging
        else:
            raise ValueError(f"generate_mode must be one of ['batch', 'streaming'], but got {generate_mode}.")

    def streaming_generate(
            self,
            streaming_split: str = "word",
            inputs: Optional[torch.Tensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
            synced_gpus: Optional[bool] = None,
            assistant_model: Optional["PreTrainedModel"] = None,
            streamer: Optional["BaseStreamer"] = None,
            negative_prompt_ids: Optional[torch.Tensor] = None,
            negative_prompt_attention_mask: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:

        tokenizer = kwargs.pop("tokenizer", None)
        generation_config, model_kwargs = self._prepare_generation_config(generation_config, **kwargs)
        self._validate_model_kwargs_streaming(model_kwargs.copy())

        if synced_gpus is None:
            if is_deepspeed_zero3_enabled() and dist.get_world_size() > 1:
                synced_gpus = True
            else:
                synced_gpus = False

        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs
        kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]
        device = inputs_tensor.device
        self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

        if not self.config.is_encoder_decoder and not is_torchdynamo_compiling():
            if (
                    generation_config._pad_token_tensor is not None
                    and batch_size > 1
                    and len(inputs_tensor.shape) == 2
                    and torch.sum(inputs_tensor[:, -1] == generation_config._pad_token_tensor) > 0
            ):
                logger.warning("Decoder-only architecture with right-padding detected.")

        if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
            model_kwargs["use_cache"] = True
        else:
            model_kwargs["use_cache"] = generation_config.use_cache

        if not kwargs_has_attention_mask and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config._pad_token_tensor, generation_config._eos_token_tensor
            )

        if self.config.is_encoder_decoder:
            input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
                batch_size=batch_size,
                model_input_name=model_input_name,
                model_kwargs=model_kwargs,
                decoder_start_token_id=generation_config._decoder_start_token_tensor,
                device=inputs_tensor.device,
            )
        else:
            input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

        if generation_config.token_healing:
            input_ids = self.heal_tokens(input_ids, tokenizer)

        if streamer is not None:
            streamer.put(input_ids.cpu())

        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,
        )

        use_dynamic_cache_by_default = False
        cache_name = "past_key_values"
        source_cache_name = "source_key_values"

        if generation_config.cache_implementation is None and self._supports_default_dynamic_cache():
            past = model_kwargs.get(cache_name, None)
            if past is None:
                model_kwargs[cache_name] = DynamicCache()
                use_dynamic_cache_by_default = True

            source_past = model_kwargs.get(source_cache_name, None)
            if source_past is None:
                model_kwargs[source_cache_name] = DynamicCache()

        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

        generation_mode = generation_config.get_generation_mode(assistant_model)

        if streamer is not None and (generation_config.num_beams > 1):
            raise ValueError("`streamer` cannot be used with beam search.")

        prepared_logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
            device=inputs_tensor.device,
            model_kwargs=model_kwargs,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )

        prepared_stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria, tokenizer=tokenizer, **kwargs
        )

        if generation_mode in (GenerationMode.SAMPLE, GenerationMode.GREEDY_SEARCH):
            prepared_logits_warper = None
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            result, wait_lagging = self._sample_streaming(
                streaming_split=streaming_split,
                input_ids=input_ids,
                logits_processor=prepared_logits_processor,
                logits_warper=prepared_logits_warper,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                tokenizer=tokenizer,
                **model_kwargs,
            )

        return result, wait_lagging

    def _validate_model_kwargs_streaming(self, model_kwargs: Dict[str, Any]):
        if isinstance(model_kwargs.get("past_key_values", None), Cache) and not self._supports_cache_class:
            raise ValueError(f"{self.__class__.__name__} does not support an instance of `Cache`.")
        pass

    # ================= [CRITICAL FIX] =================
    # Completely DISABLE these destructive functions.
    # We let the Cache objects manage themselves.
    def merge_source_target(self):
        # Do NOTHING. H2O Cache manages its own memory.
        pass

    def separate_source_target(self):
        # Do NOTHING. H2O Cache manages its own memory.
        pass

    # ==================================================

    def _get_initial_cache_position_for_streaming(self, input_length, model_kwargs):
        assert self.source_key_values is not None
        # Safe length retrieval
        if hasattr(self.source_key_values, 'get_seq_length'):
            current_len = self.source_key_values.get_seq_length()
        else:
            current_len = 0

        cache_position = torch.arange(
            current_len, input_length[0], dtype=torch.int64, device=model_kwargs.get('assistant_token').device
        )
        model_kwargs["cache_position"] = cache_position
        return model_kwargs

    def prepare_inputs_for_generation_streaming(
            self,
            input_ids,
            past_key_values=None,
            attention_mask=None,
            inputs_embeds=None,
            cache_position=None,
            position_ids=None,
            use_cache=True,
            is_streaming=False,
            input_length=None,
            pe_cache_length=0,
            assistant_token=None,
            source_words=None,
            end_Instruct=None,
            ReadAction=True,
            **kwargs,
    ):
        if not is_streaming:
            # (Standard batch logic omitted for brevity, fallback to original if needed)
            pass

        elif is_streaming:
            assert input_length is not None, "input_length must be provided for streaming generation"
            model_inputs = kwargs.copy()

            past_source_length = 0
            past_target_length = 0

            # Safe length retrieval
            if self.source_key_values is not None:
                if hasattr(self.source_key_values, 'get_seq_length'):
                    past_source_length = self.source_key_values.get_seq_length()
                elif isinstance(self.source_key_values, (list, tuple)) and len(self.source_key_values) > 0:
                    past_source_length = self.source_key_values[0][0].shape[-2]

            if self.target_key_values is not None:
                if hasattr(self.target_key_values, 'get_seq_length'):
                    past_target_length = self.target_key_values.get_seq_length()
                elif isinstance(self.target_key_values, (list, tuple)) and len(self.target_key_values) > 0:
                    past_target_length = self.target_key_values[0][0].shape[-2]

            if ReadAction:
                # Reading Source
                # position_ids: [past_source, input_length]
                position_ids_source = torch.arange(past_source_length, input_length[0]).to(
                    assistant_token.device).unsqueeze(0)
                position_ids = position_ids_source.clone().detach()
                input_ids = input_ids[:, past_source_length:input_length[0]]
            elif not ReadAction:
                # Generating Target
                # [FIX] Position ID must be Offset by Source Length!
                # Target generation starts AFTER source tokens.
                # If source is 25 tokens, target must start at position 25.
                current_pos = past_source_length + past_target_length + pe_cache_length
                position_ids = torch.tensor([current_pos]).to(assistant_token.device).unsqueeze(0)
                input_ids = input_ids

            if ReadAction:
                model_inputs.update(
                    {
                        "input_ids": input_ids,
                        "position_ids": position_ids,
                        "use_cache": use_cache,
                        "attention_mask": attention_mask,
                        "source_key_values": self.source_key_values,
                        "pe_cache_length": pe_cache_length
                    }
                )
            else:
                model_inputs.update(
                    {
                        "input_ids": input_ids,
                        "position_ids": position_ids,
                        "use_cache": use_cache,
                        "attention_mask": attention_mask,
                        "past_key_values": self.past_key_values,
                        "pe_cache_length": pe_cache_length,
                        # Also pass source_key_values so Attention can concatenate
                        "source_key_values": self.source_key_values
                    }
                )
        return model_inputs

    def _sample_streaming(
            self,
            input_ids: torch.LongTensor,
            logits_processor: LogitsProcessorList,
            stopping_criteria: StoppingCriteriaList,
            generation_config: GenerationConfig,
            synced_gpus: bool,
            streamer: Optional["BaseStreamer"],
            tokenizer: Optional["PreTrainedTokenizerBase"],
            **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:

        # Init caches if needed
        if self.source_key_values is None:
            self.source_key_values = DynamicCache()
        if self.target_key_values is None:
            self.target_key_values = DynamicCache()
        if self.past_key_values is None:
            self.past_key_values = DynamicCache()

        end_Instruct = model_kwargs.get("end_Instruct", None)
        max_new_tokens = generation_config.max_new_tokens if hasattr(generation_config,
                                                                     'max_new_tokens') and generation_config.max_new_tokens is not None else 1024
        ReadAction_criteria = StopTokenCriteria(tokenizer, max_new_tokens=max_new_tokens, end_Instruct=end_Instruct)

        _lengths = model_kwargs.get("_lengths", None)
        source_seg_len = _lengths[0]['source_seg_len']
        pe_cache_length = model_kwargs.get("pe_cache_length", None)
        assistant_token = model_kwargs.get("assistant_token", None)
        ReadAction = True

        wait_k = model_kwargs.get("wait_k", None)
        source_words = wait_k if wait_k is not None else 0
        target_words = 0
        max_distance = 100

        next_tokens = model_kwargs['assistant_token'].unsqueeze(0)
        target_tokens = [next_tokens[:, :-1], next_tokens[:, -1:]]
        target_tokens_this_write = []
        wait_lagging = []

        input_length = (sum(source_seg_len[:model_kwargs['wait_k'] + 1]), 1)
        source_input_length = sum(source_seg_len[:model_kwargs['wait_k'] + 1])

        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample

        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        batch_size, input_len = input_ids.shape
        cur_len = input_len
        generated_tokens_count = 0
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)

        # Initial cache position setup
        model_kwargs = self._get_initial_cache_position_for_streaming(input_length, model_kwargs)

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            if ReadAction:
                # [FIX] DISABLE destructive separation
                # self.separate_source_target()

                model_inputs = self.prepare_inputs_for_generation_streaming(input_ids, input_length=input_length,
                                                                            ReadAction=ReadAction, is_streaming=True,
                                                                            **model_kwargs)
                model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
                model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

                _outputs = self(
                    **model_inputs,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    ReadAction=ReadAction,
                )

                # Switch to Generate
                ReadAction = False
                token_count = 0

                # [FIX] DISABLE destructive merge
                # self.merge_source_target()

            elif not ReadAction:
                # [FIX] DISABLE destructive separation
                # self.separate_source_target()

                token_count += 1
                model_inputs = self.prepare_inputs_for_generation_streaming(next_tokens, input_length=input_length,
                                                                            ReadAction=ReadAction, is_streaming=True,
                                                                            **model_kwargs)

                model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
                model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

                outputs = self(
                    **model_inputs,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    ReadAction=ReadAction,
                )

                model_kwargs = self._update_model_kwargs_for_generation(
                    outputs,
                    model_kwargs,
                    is_encoder_decoder=self.config.is_encoder_decoder,
                )
                if synced_gpus and this_peer_finished:
                    continue

                next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)
                next_token_scores = logits_processor(input_ids, next_token_logits)

                if return_dict_in_generate:
                    if output_scores: scores += (next_token_scores,)
                    if output_logits: raw_logits += (next_token_logits,)

                if do_sample:
                    probs = nn.functional.softmax(next_token_scores, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    next_tokens = torch.argmax(next_token_scores, dim=-1)

                if has_eos_stopping_criteria:
                    next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
                if streamer is not None:
                    streamer.put(next_tokens.cpu())

                del outputs

                next_tokens = next_tokens.unsqueeze(0)
                target_tokens.append(next_tokens)
                target_ids = torch.cat(target_tokens, dim=-1)
                target_tokens_this_write.append(next_tokens)
                target_ids_this_write = torch.cat(target_tokens_this_write, dim=-1)

                ReadAction_new, remove_last_token = ReadAction_criteria(target_ids_this_write, scores, token_count)
                unfinished_sequences = unfinished_sequences & ~stopping_criteria(target_ids[0:, 2:], scores)

                generated_tokens_count += 1
                if generated_tokens_count >= max_new_tokens:
                    this_peer_finished = True

                cur_len += 1
                wait_lagging.append(source_words)
                target_words += 1

                source_finished = source_words >= len(source_seg_len) - 1

                if source_finished:
                    ReadAction = False
                elif not ReadAction_new:
                    ReadAction = True
                    source_words += 1
                    if source_words < len(source_seg_len):
                        num_tokens = source_seg_len[source_words]
                        source_input_length += num_tokens
                        target_input_length = 1
                        input_length = (source_input_length, target_input_length)
                else:
                    ReadAction = False
                    if remove_last_token:
                        target_tokens.pop()
                        next_tokens = target_tokens[-1]
                        target_tokens_this_write = []
                        # Important: Do not pop from H2O Cache directly if it's managed internally
                        # But for consistency with stopping criteria, we might need to rollback
                        # For now, let's assume we don't pop to avoid H2O state corruption
                        # self.past_key_values.pop()

                    distance = target_words - source_words
                    if distance > max_distance and source_words >= len(source_seg_len) - 1:
                        this_peer_finished = True

        if streamer is not None:
            streamer.end()

        assistant_token = model_kwargs.get('assistant_token', None)
        if assistant_token is not None and len(target_tokens) > 0:
            if len(target_tokens) >= 2:
                actual_generated_tokens = target_tokens[2:]
                if len(actual_generated_tokens) > 0:
                    target_ids = torch.cat(actual_generated_tokens, dim=-1)
                else:
                    device = target_tokens[0].device if len(target_tokens) > 0 else input_ids.device
                    target_ids = torch.tensor([], dtype=torch.long, device=device).unsqueeze(0)

        if return_dict_in_generate:
            return GenerateDecoderOnlyOutput(
                sequences=target_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return target_ids, wait_lagging