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

"""
GRPO Trainer with Atropos Integration

This module implements a GRPO trainer that uses Atropos environments
for computing advantages with token-level overrides.
"""

import logging
from typing import Any, Optional

import torch

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

from .atropos_integration import AtroposConfig, AtroposGRPOComputer

logger = logging.getLogger(__name__)


class RayGRPOAtroposTrainer(RayPPOTrainer):
    """
    Ray-based GRPO trainer with Atropos integration.
    
    This trainer extends the standard PPO trainer to use GRPO with
    Atropos environment feedback for advantage computation.
    """
    
    def __init__(self, config, role_worker_mapping, resource_pool_manager):
        super().__init__(config, role_worker_mapping, resource_pool_manager)
        
        # Initialize Atropos integration
        atropos_config = AtroposConfig(
            api_url=config.trainer.atropos.get("api_url", "http://localhost:9001"),
            timeout=config.trainer.atropos.get("timeout", 30),
            retry_attempts=config.trainer.atropos.get("retry_attempts", 10),
            retry_delay=config.trainer.atropos.get("retry_delay", 0.5),
            max_wait_time=config.trainer.atropos.get("max_wait_time", 30.0),
            use_advantages=config.trainer.atropos.get("use_advantages", True),
            fallback_to_standard=config.trainer.atropos.get("fallback_to_grpo", True)
        )
        
        self.grpo_computer = AtroposGRPOComputer(atropos_config)
        
        # Ensure we're using GRPO
        if config.algorithm.adv_estimator != "grpo":
            logger.warning(f"Overriding adv_estimator from {config.algorithm.adv_estimator} to grpo")
            config.algorithm.adv_estimator = "grpo"
        
        # GRPO doesn't use critic
        self.use_critic = False
        config.algorithm.use_critic = False
        
        logger.info("Initialized RayGRPOAtroposTrainer with Atropos integration")
    
    def _compute_advantages_grpo(self, batch: DataProto) -> tuple[Optional[torch.Tensor], dict[str, Any]]:
        """
        Compute GRPO advantages with Atropos environment overrides.
        
        This method overrides the standard advantage computation to integrate
        with Atropos environments for token-level advantages.
        """
        # Extract data from batch
        input_ids = batch.batch["input_ids"]
        responses = batch.batch["responses"]
        response_mask = batch.batch["response_mask"]
        
        # Get prompt lengths from batch metadata if available, otherwise calculate
        if "prompt_len" in batch.non_tensor_batch:
            prompt_lengths = batch.non_tensor_batch["prompt_len"]
            pad_token_id = getattr(self.tokenizer, "pad_token_id", 0) or 0
            prompts_tensor = torch.nn.utils.rnn.pad_sequence(
                [input_ids[i, :prompt_len] for i, prompt_len in enumerate(prompt_lengths)],
                batch_first=True,
                padding_value=pad_token_id,
            )
            prompts_tensor = prompts_tensor.to(input_ids.device)
        else:
            # Fallback to original calculation
            response_length = responses.shape[1]
            prompt_length = input_ids.shape[1] - response_length
            prompts_tensor = input_ids[:, :prompt_length]
        
        # Get scores if available
        scores = batch.batch.get("token_level_scores")
        if scores is None:
            token_level_rewards = batch.batch.get("token_level_rewards")
            if token_level_rewards is not None:
                scores = token_level_rewards.sum(dim=-1)
            else:
                scores = torch.zeros(
                    response_mask.shape[0],
                    device=response_mask.device,
                    dtype=torch.float32,
                )
        
        # Get tokenizer
        tokenizer = self.tokenizer
        
        # Submit to Atropos and get advantages
        advantages, metrics = self.grpo_computer.compute_advantages_with_overrides(
            prompts=prompts_tensor,
            responses=responses,
            scores=scores,
            tokenizer=tokenizer,
            response_mask=response_mask,
        )
        
        return advantages, metrics
    
    def training_step(self, batch_dict):
        """
        Override training step to use GRPO advantage computation.
        """
        # Clone the batch to avoid in-place mutations that can cause DDP issues
        batch_dict_cloned = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch_dict.items()}
        
        # Convert to DataProto
        batch = DataProto.from_single_dict(batch_dict_cloned)
        
        # Compute advantages using GRPO with Atropos
        advantages, metrics = self._compute_advantages_grpo(batch)
        if metrics:
            logger.info(f"Atropos metrics: {metrics}")
        if advantages is not None:
            batch.batch["token_level_advantages"] = advantages
        
        # Continue with standard training
        return super().training_step(batch.batch)
