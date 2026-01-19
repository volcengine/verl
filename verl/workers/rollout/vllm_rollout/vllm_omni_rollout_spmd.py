# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
import inspect
import logging
import os
from typing import Generator

import torch
import torchvision.transforms as T
from tensordict import TensorDict
from torch.distributed.device_mesh import DeviceMesh
from transformers import AutoTokenizer, PreTrainedTokenizer
from vllm.lora.request import LoRARequest
from vllm_omni.entrypoints.omni_diffusion import OmniDiffusion

from verl import DataProto
from verl.third_party.vllm_omni import VLLM_OMNI_SLEEP_LEVEL
from verl.utils.model import get_lora_rank_from_adapter
from verl.utils.profiler import GPUMemoryLogger
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.vllm_rollout.utils import get_vllm_max_lora_rank

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class vLLMOmniRollout(BaseRollout):
    def __init__(self, config: RolloutConfig, model_config: HFModelConfig, device_mesh: DeviceMesh):
        super().__init__(config, model_config, device_mesh)

        if config.layered_summon:
            self.sleep_level = 1
        else:
            self.sleep_level = VLLM_OMNI_SLEEP_LEVEL

        model_path = model_config.local_path
        lora_adapter_path = getattr(model_config, "lora_adapter_path", None)
        if lora_adapter_path is not None:
            lora_rank = get_lora_rank_from_adapter(lora_adapter_path)
        else:
            lora_rank = model_config.lora_rank

        self.lora_kwargs = (
            {"enable_lora": True, "max_loras": 1, "max_lora_rank": get_vllm_max_lora_rank(lora_rank)}
            if model_config.lora_rank > 0
            else {}
        )

        self.inference_engine = OmniDiffusion(model=model_path)
        if (tokenizer_path := self.model_config.tokenizer_path) is None:
            tokenizer_path = model_path
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self._to_tensor = T.PILToTensor()

    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # TODO: the vllm-omni should able to feed tokenized ids directly
        if (prompt := prompts.non_tensor_batch.get("prompt")) is None:
            idx = prompts.batch["input_ids"]
            prompt = self.tokenizer.batch_decode(idx, skip_special_tokens=True)
        else:
            prompt = prompt.tolist()

        batch_size = len(prompt)

        vllm_omni_inputs = prompt

        lora_requests = [None] * batch_size
        if self.lora_kwargs:
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_requests = [
                    LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/simon-stub-path")
                ] * batch_size
        # users can customize different sampling_params at different run
        # TODO: currently vLLM-Omni do not accept batch inference
        outputs = [
            self.inference_engine.generate(x, lora_request=lora_request)
            for x, lora_request in zip(vllm_omni_inputs, lora_requests, strict=False)
        ]

        response = []
        for output in outputs:
            response.append(self._to_tensor(output.images[0]))
        response = torch.stack(response, dim=0)

        batch = TensorDict(
            {"responses": response},
            batch_size=batch_size,
        )

        return DataProto(batch=batch)

    async def resume(self, tags: list[str]):
        """Resume rollout weights or kv cache in GPU memory.

        Args:
            tags: weights or kv_cache.
        """
        if not self.config.free_cache_engine:
            return

        if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
            self.inference_engine.wake_up(tags=tags)
        else:
            self.inference_engine.wake_up()

    async def release(self):
        """Release weights and kv cache in GPU memory."""
        self.inference_engine.reset_prefix_cache()

        if not self.config.free_cache_engine:
            return

        self.inference_engine.sleep(level=self.sleep_level)

    async def update_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None], **kwargs):
        """Update the weights of the rollout model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
        """
        raise NotImplementedError("vLLM-Omni rollout does not support weight update yet.")
