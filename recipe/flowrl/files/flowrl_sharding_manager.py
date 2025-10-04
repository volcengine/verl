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

"""FlowRL Sharding Manager that filters proj_z parameters when syncing to vLLM."""

import logging
from typing import Optional

try:
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor

from verl.workers.sharding_manager.fsdp_vllm import FSDPVLLMShardingManager
from verl.utils.device import get_device_id

logger = logging.getLogger(__name__)


class FlowRLFSDPVLLMShardingManager(FSDPVLLMShardingManager):
    """
    FlowRL version of FSDPVLLMShardingManager that filters out proj_z parameters
    when syncing weights to vLLM inference engine.

    The proj_z module is only needed during training for estimating log Z (partition function).
    It should not be loaded into vLLM since:
    1. vLLM is only used for rollout generation (inference)
    2. It doesn't have the proj_z architecture
    3. Including it would cause weight loading errors
    """

    def update_vllm_weights(self, updated_params: dict, peft_config: Optional[dict] = None):
        """
        Synchronizes parameters from FSDP training model to vLLM inference engine,
        filtering out proj_z parameters.

        Args:
            updated_params (dict): Dictionary of parameter names to tensor values.
            peft_config (optional): PEFT configuration for LoRA adapters.
        """
        # Filter out proj_z parameters before syncing to vLLM
        filtered_params = {
            name: param
            for name, param in updated_params.items()
            if not name.startswith("proj_z")
        }

        num_filtered = len(updated_params) - len(filtered_params)
        if num_filtered > 0:
            logger.info(f"[FlowRL] Filtered {num_filtered} proj_z parameters before syncing to vLLM")

        # Call parent class method with filtered parameters
        super().update_vllm_weights(filtered_params, peft_config)
