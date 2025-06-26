# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
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

import torch
import torch.distributed
from omegaconf import DictConfig, OmegaConf
from transformers import AutoConfig, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.distributed import parallel_state as vllm_ps

from verl.third_party.vllm import vllm_version
from verl.workers.rollout.base import BaseRollout


class vLLMRollout(BaseRollout):
    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

        if kwargs.get("train_tp", None) is not None:
            # deployed with megatron
            import os

            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            train_tp = kwargs.get("train_tp", None)
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size, num_tp_per_train_tp=num_tp_per_train_tp)
        # before fixing the hf config model context window assertion
        # expected behavior: if the model use rope scaling yarn in its config,
        # it will still raise error since this assertion only checks the `max_position_embeddings` in the HF config
        # assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, "model context length should be greater than total sequence length"
        # after fixing
        # expect behavior: if the model uses yarn position embeddings, it will pass the assertion
        # since it checks the rope scaling factor (if any) to calculate the effective max position embeddings
        rope_scaling_factor = 1.0
        if hasattr(model_hf_config, "rope_scaling") and model_hf_config.rope_scaling is not None:
            # check if the type is yarn
            rope_scaling = model_hf_config.rope_scaling
            rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
            if rope_type == "yarn":
                rope_scaling_factor = rope_scaling.get("factor", 1.0)
        assert model_hf_config.max_position_embeddings * rope_scaling_factor >= config.prompt_length + config.response_length, "model context length should be greater than total sequence length"

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=config.prompt_length + config.response_length,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # # we may detokenize the result all together later
        if vllm_version != "0.3.1":
            kwargs["detokenize"] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

    @torch.no_grad()
    def generate_sequences(self):
        # this function is not used in the initialization of the rollout object
        # thus passed in this test
        pass


def test_vllm_rollout_with_yarn_position_embeddings():
    """
    Test the vLLM rollout with yarn position embeddings.
    """

    config = OmegaConf.create(
        {
            "model_path": "OldKingMeister/Qwen2.5-1.5B-Instruct-YaRN",
            "prompt_length": 32768,
            "response_length": 512,
            "dtype": "bfloat16",
            "enforce_eager": False,
            "gpu_memory_utilization": 0.9,
            "enable_chunked_prefill": True,
            "free_cache_engine": False,
            "disable_log_stats": True,
        }
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    model_hf_config = AutoConfig.from_pretrained(config.model_path)

    vLLMRollout(
        model_path=config.model_path,
        config=config,
        tokenizer=tokenizer,
        model_hf_config=model_hf_config,
    )


if __name__ == "__main__":
    import os

    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    test_vllm_rollout_with_yarn_position_embeddings()
