# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023 The vLLM team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/llm.py

from typing import (Any, Dict, List, Optional, Tuple, Type, Union)

import json
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import PretrainedConfig, PreTrainedTokenizer, PreTrainedTokenizerFast
from verl.workers.rollout.tokenizer import HybridEngineBaseTokenizer
from vllm import LLM
from vllm.outputs import EmbeddingRequestOutput, RequestOutput
from vllm.utils import Counter
from vllm.config import CompilationConfig
from vllm.engine.arg_utils import (HfOverrides, PoolerConfig, TaskOption)

from .arg_utils import EngineArgs
from .llm_engine_sp import LLMEngine




from tqdm import tqdm
from vllm import envs










class LLM(LLM):
    """An LLM for generating texts from given prompts and sampling parameters.

    This class includes a tokenizer, a language model (possibly distributed
    across multiple GPUs), and GPU memory space allocated for intermediate
    states (aka KV cache). Given a batch of prompts and sampling parameters,
    this class generates texts from the model, using an intelligent batching
    mechanism and efficient memory management.

    Args:
        model: The name or path of a HuggingFace Transformers model.
        tokenizer: The name or path of a HuggingFace Transformers tokenizer.
        tokenizer_mode: The tokenizer mode. "auto" will use the fast tokenizer
            if available, and "slow" will always use the slow tokenizer.
        skip_tokenizer_init: If true, skip initialization of tokenizer and
            detokenizer. Expect valid prompt_token_ids and None for prompt
            from the input.
        trust_remote_code: Trust remote code (e.g., from HuggingFace) when
            downloading the model and tokenizer.
        allowed_local_media_path: Allowing API requests to read local images
            or videos from directories specified by the server file system.
            This is a security risk. Should only be enabled in trusted
            environments.
        tensor_parallel_size: The number of GPUs to use for distributed
            execution with tensor parallelism.
        dtype: The data type for the model weights and activations. Currently,
            we support `float32`, `float16`, and `bfloat16`. If `auto`, we use
            the `torch_dtype` attribute specified in the model config file.
            However, if the `torch_dtype` in the config is `float32`, we will
            use `float16` instead.
        quantization: The method used to quantize the model weights. Currently,
            we support "awq", "gptq", and "fp8" (experimental).
            If None, we first check the `quantization_config` attribute in the
            model config file. If that is None, we assume the model weights are
            not quantized and use `dtype` to determine the data type of
            the weights.
        revision: The specific model version to use. It can be a branch name,
            a tag name, or a commit id.
        tokenizer_revision: The specific tokenizer version to use. It can be a
            branch name, a tag name, or a commit id.
        seed: The seed to initialize the random number generator for sampling.
        gpu_memory_utilization: The ratio (between 0 and 1) of GPU memory to
            reserve for the model weights, activations, and KV cache. Higher
            values will increase the KV cache size and thus improve the model's
            throughput. However, if the value is too high, it may cause out-of-
            memory (OOM) errors.
        swap_space: The size (GiB) of CPU memory per GPU to use as swap space.
            This can be used for temporarily storing the states of the requests
            when their `best_of` sampling parameters are larger than 1. If all
            requests will have `best_of=1`, you can safely set this to 0.
            Otherwise, too small values may cause out-of-memory (OOM) errors.
        cpu_offload_gb: The size (GiB) of CPU memory to use for offloading
            the model weights. This virtually increases the GPU memory space
            you can use to hold the model weights, at the cost of CPU-GPU data
            transfer for every forward pass.
        enforce_eager: Whether to enforce eager execution. If True, we will
            disable CUDA graph and always execute the model in eager mode.
            If False, we will use CUDA graph and eager execution in hybrid.
        max_seq_len_to_capture: Maximum sequence len covered by CUDA graphs.
            When a sequence has context length larger than this, we fall back
            to eager mode. Additionally for encoder-decoder models, if the
            sequence length of the encoder input is larger than this, we fall
            back to the eager mode.
        disable_custom_all_reduce: See :class:`~vllm.config.ParallelConfig`
        disable_async_output_proc: Disable async output processing.
            This may result in lower performance.
        hf_overrides: If a dictionary, contains arguments to be forwarded to the
            HuggingFace config. If a callable, it is called to update the
            HuggingFace config.
        compilation_config: Either an integer or a dictionary. If it is an
            integer, it is used as the level of compilation optimization. If it
            is a dictionary, it can specify the full compilation configuration.
        **kwargs: Arguments for :class:`~vllm.EngineArgs`. (See
            :ref:`engine_args`)

    Note:
        This class is intended to be used for offline inference. For online
        serving, use the :class:`~vllm.AsyncLLMEngine` class instead.
    """

    def __init__(
        self,
        model: Union[nn.Module, Dict], # model itself or its parameter dict
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast, HybridEngineBaseTokenizer],
        model_hf_config: PretrainedConfig,
        tokenizer_mode: str = "auto",
        skip_tokenizer_init: bool = False,
        trust_remote_code: bool = False,
        allowed_local_media_path: str = "",
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        seed: int = 0,
        gpu_memory_utilization: float = 0.9,
        swap_space: float = 4,
        cpu_offload_gb: float = 0,
        enforce_eager: Optional[bool] = None,
        max_seq_len_to_capture: int = 8192,
        disable_custom_all_reduce: bool = False,
        disable_async_output_proc: bool = False,
        hf_overrides: Optional[HfOverrides] = None,
        mm_processor_kwargs: Optional[Dict[str, Any]] = None,
        # After positional args are removed, move this right below `model`
        task: TaskOption = "generate",
        override_pooler_config: Optional[PoolerConfig] = None,
        compilation_config: Optional[Union[int, Dict[str, Any]]] = None,
        **kwargs,
    ) -> None:
        '''
        LLM constructor.

        Note: if enforce_eager is unset (enforce_eager is None)
        it defaults to False.
        '''

        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True
        removed_vision_keys = ("image_token_id", "image_feature_size", "image_input_shape", "image_input_type")
        if any(k in kwargs for k in removed_vision_keys):
            raise TypeError("There is no need to pass vision-related arguments anymore.")

        if compilation_config is not None:
            compilation_config_instance = CompilationConfig.from_cli(
                json.dumps(compilation_config))
        else:
            compilation_config_instance = None

        engine_args = EngineArgs(
            model_hf_config=model_hf_config,
            task=task,
            # tokenizer=tokenizer,
            tokenizer_mode=tokenizer_mode,
            skip_tokenizer_init=skip_tokenizer_init,
            trust_remote_code=trust_remote_code,
            allowed_local_media_path=allowed_local_media_path,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            quantization=quantization,
            revision=revision,
            tokenizer_revision=tokenizer_revision,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            swap_space=swap_space,
            cpu_offload_gb=cpu_offload_gb,
            enforce_eager=enforce_eager,
            max_seq_len_to_capture=max_seq_len_to_capture,
            disable_custom_all_reduce=disable_custom_all_reduce,
            disable_async_output_proc=disable_async_output_proc,
            hf_overrides=hf_overrides,
            mm_processor_kwargs=mm_processor_kwargs,
            override_pooler_config=override_pooler_config,
            compilation_config=compilation_config_instance,
            **kwargs,
        )
        tokenizer_cls = (PreTrainedTokenizer, PreTrainedTokenizerFast, HybridEngineBaseTokenizer)
        if not isinstance(tokenizer, tokenizer_cls):
            raise ValueError(
                f"Unexpected tokenizer type: {type(tokenizer)}. Must be"
                "one of the following: PreTrainedTokenizer, PreTrainedTokenizerFast, verl.trainer.ppo.rollout.HybridEngineBaseTokenizer"
            )
        self.engine_class = self.get_engine_class()
        self.llm_engine = self.engine_class.from_engine_args(model, tokenizer, engine_args)  # TODO: check usagecontext
        self.request_counter = Counter()

    @staticmethod
    def get_engine_class() -> Type[LLMEngine]:
        if envs.VLLM_USE_V1:
            # Lazy import: the v1 package isn't distributed
            from vllm.v1.engine.llm_engine import LLMEngine as V1LLMEngine
            return V1LLMEngine  # type: ignore
        return LLMEngine

    def init_cache_engine(self):
        self.llm_engine.init_cache_engine()

    def free_cache_engine(self):
        self.llm_engine.free_cache_engine()

    def get_tokenizer(self) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
        return self.llm_engine.tokenizer

    def set_tokenizer(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    ) -> None:
        self.llm_engine.tokenizer = tokenizer

    def _run_engine(self, *, use_tqdm: bool) -> List[Union[RequestOutput, EmbeddingRequestOutput]]:
        outputs = super()._run_engine(use_tqdm=use_tqdm)
        return self._post_process_outputs(outputs)

    # # NOTE(shengguangming): add for verl
    # # TODO(sgm): we can optimize it by making the dataloader yield List[int] without padding.
    # def _pre_process_inputs(self, prompt_token_ids: torch.Tensor) -> List[int]:
    #     # remove the left padding in the prompt token_id
    #     pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    #     non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    #     token_ids = prompt_token_ids[non_pad_index:].tolist()
    #     return token_ids

    # NOTE(shengguangming): add for verl
    def _post_process_outputs(self, request_outputs: List[RequestOutput]) -> Tuple[torch.Tensor, torch.Tensor]:
        output_token_ids = []
        logprobs = []
        for request_output in request_outputs:  # List[RequestOutput]
            outputs = request_output.outputs
            for output in outputs:  # List[CompletionOutput], usually len == 1
                output_token_ids.append(torch.tensor(output.token_ids))
                # TODO(shengguangming): can be optimzied by rewrite the Sampler._get_logprobs() logits
                logprobs_dicts = output.logprobs
                if logprobs_dicts is not None:
                    logprob = []
                    for logprobs_dict, id in zip(logprobs_dicts, output.token_ids):
                        logprob.append(logprobs_dict[id].logprob)
                    logprobs.append(torch.tensor(logprob))

        pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
        # pad_token_id = 128004
        output_token_ids = pad_sequence(output_token_ids, batch_first=True, padding_value=pad_token_id)
        if len(logprobs) > 0:
            logprobs = pad_sequence(logprobs, batch_first=True, padding_value=pad_token_id)
        return output_token_ids, logprobs

    def sync_model_weights(self, actor_weights: Dict[str, torch.Tensor], load_format: str) -> None:
        self.llm_engine.sync_model_weights(actor_weights=actor_weights, load_format=load_format)

    def offload_model_weights(self) -> None:
        self.llm_engine.offload_model_weights()
