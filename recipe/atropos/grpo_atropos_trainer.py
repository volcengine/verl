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
from typing import Dict, Optional, Any
import torch
import numpy as np

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.core_algos import compute_advantage

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
        if config.algorithm.adv_estimator != "grpo_atropos":
            logger.warning(f"Overriding adv_estimator from {config.algorithm.adv_estimator} to grpo_atropos")
            config.algorithm.adv_estimator = "grpo_atropos"
        
        # GRPO doesn't use critic
        self.use_critic = False
        config.algorithm.use_critic = False
        
        logger.info("Initialized RayGRPOAtroposTrainer with Atropos integration")
    
    def _compute_advantages_grpo(self, batch: DataProto) -> torch.Tensor:
        """
        Compute GRPO advantages with Atropos environment overrides.
        
        This method overrides the standard advantage computation to integrate
        with Atropos environments for token-level advantages.
        """
        # Extract data from batch
        input_ids = batch.batch["input_ids"]
        responses = batch.batch["responses"]
        response_mask = batch.batch["response_mask"]
        
        # Get prompts (input_ids minus response)
        response_length = responses.shape[1]
        prompt_length = input_ids.shape[1] - response_length
        prompts = input_ids[:, :prompt_length]
        
        # Get scores if available
        scores = batch.batch.get("token_level_scores")
        if scores is None:
            # Compute simple scores from rewards
            rewards = batch.batch.get("token_level_rewards", torch.zeros_like(response_mask))
            scores = rewards.sum(dim=-1)
        
        # Get tokenizer
        tokenizer = self.tokenizer
        
        # Submit to Atropos and get advantages
        advantages, metrics = self.grpo_computer.compute_advantages_with_overrides(
            prompts=prompts,
            responses=responses,
            scores=scores,
            tokenizer=tokenizer,
            response_mask=response_mask,
            fallback_estimator=lambda s, m: self._compute_standard_grpo_advantages(batch)
        )
        
        # Log metrics
        if metrics:
            logger.info(f"Atropos metrics: {metrics}")
        
        return advantages
    
    def _compute_standard_grpo_advantages(self, batch: DataProto) -> torch.Tensor:
        """Compute standard GRPO advantages as fallback"""
        # Use the registered grpo_atropos estimator
        advantages = compute_advantage(
            rewards=batch.batch.get("token_level_rewards", batch.batch.get("token_level_scores")),
            values=None,  # GRPO doesn't use values
            response_length=batch.batch["responses"].shape[1],
            adv_estimator="grpo_atropos",
            gamma=self.config.algorithm.gamma,
            lam=self.config.algorithm.lam,
            response_mask=batch.batch["response_mask"],
            old_rewards=batch.batch.get("old_rewards"),
            ref_log_probs=batch.batch.get("ref_log_probs"),
            log_probs=batch.batch.get("old_log_probs"),
            kl_penalties=batch.batch.get("kl_penalties"),
            kl_rewards=batch.batch.get("kl_rewards"),
            uid=batch.non_tensor_batch.get("uid"),
            norm_adv_by_std_in_grpo=True,
            epsilon=1e-6
        )
        return advantages
    
    def training_step(self, batch_dict):
        """
        Override training step to use GRPO advantage computation.
        """
        # Convert to DataProto
        batch = DataProto.from_single_dict(batch_dict)
        
        # Compute advantages using GRPO with Atropos
        advantages = self._compute_advantages_grpo(batch)
        batch.batch["advantages"] = advantages
        batch.batch["returns"] = advantages  # GRPO uses advantages as returns
        
        # Continue with standard training
        return super().training_step(batch)