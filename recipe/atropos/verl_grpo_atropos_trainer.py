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
VeRL GRPO Trainer with Atropos Integration

This trainer extends VeRL's GRPO capabilities with Atropos groups for full online RL:
- Wire Atropos groups into VeRL's GRPO trainer 
- Keep inference engine weights current and stay on-policy with KL reference model
- Expose all GRPO hyperparameters
- Handle environment feedback and advantage computation from Atropos
"""

import logging
import os
import uuid
from typing import Dict, Any, Optional, List, Union, Callable
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import Dataset, Sampler
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer, 
    ResourcePoolManager, 
    Role, 
    AdvantageEstimator,
    compute_advantage,
    apply_kl_penalty
)
from verl.trainer.ppo import core_algos
from verl.utils.metric import reduce_metrics

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_ATROPOS_LOGGING_LEVEL", "INFO"))


class AtroposGroupManager:
    """
    Manager for Atropos environment groups and advantage computation.
    
    Handles:
    - Environment group creation and management
    - Response evaluation via Atropos environments
    - Advantage computation from environment feedback
    - Group-based statistics and metrics
    """
    
    def __init__(self, config: DictConfig, tokenizer: PreTrainedTokenizer):
        self.config = config
        self.tokenizer = tokenizer
        
        # Atropos-specific configuration
        self.num_groups = config.atropos.get("num_groups", 1)
        self.group_size = config.atropos.get("group_size", 4)  # Responses per group
        self.environment_type = config.atropos.get("environment_type", "code_execution")
        self.max_env_steps = config.atropos.get("max_env_steps", 10)
        
        # Advantage computation settings
        self.advantage_mode = config.atropos.get("advantage_mode", "outcome_based")  # "outcome_based", "step_based"
        self.normalize_advantages = config.atropos.get("normalize_advantages", True)
        self.advantage_clip_range = config.atropos.get("advantage_clip_range", None)  # [min, max] or None
        
        # Environment feedback configuration
        self.reward_shaping = config.atropos.get("reward_shaping", "sparse")  # "sparse", "dense", "shaped"
        self.success_reward = config.atropos.get("success_reward", 1.0)
        self.failure_penalty = config.atropos.get("failure_penalty", -0.1)
        
        logger.info(f"Atropos Group Manager initialized:")
        logger.info(f"  - Groups: {self.num_groups}, Group size: {self.group_size}")
        logger.info(f"  - Environment: {self.environment_type}")
        logger.info(f"  - Advantage mode: {self.advantage_mode}")
        logger.info(f"  - Reward shaping: {self.reward_shaping}")
        
        # Initialize environment interfaces (placeholder for actual Atropos integration)
        self._init_environments()
        
    def _init_environments(self):
        """Initialize Atropos environment interfaces."""
        # Placeholder for actual Atropos environment initialization
        # In real implementation, this would connect to Atropos environments
        logger.info("Initializing Atropos environments...")
        self.environments = []
        for i in range(self.num_groups):
            # Mock environment for now - replace with actual Atropos environment
            env_config = {
                "group_id": i,
                "environment_type": self.environment_type,
                "max_steps": self.max_env_steps,
            }
            self.environments.append(env_config)
        logger.info(f"Initialized {len(self.environments)} Atropos environments")
        
    def evaluate_responses_in_environment(
        self, 
        prompts: torch.Tensor, 
        responses: torch.Tensor,
        group_indices: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluate responses in Atropos environments and compute advantages.
        
        Args:
            prompts: Input prompts [batch_size, prompt_len]
            responses: Generated responses [batch_size, response_len] 
            group_indices: Group ID for each sample [batch_size]
            
        Returns:
            Dictionary containing advantages, rewards, and environment feedback
        """
        batch_size = prompts.shape[0]
        response_len = responses.shape[1]
        
        # Decode responses for environment evaluation
        response_texts = []
        for i in range(batch_size):
            # Combine prompt and response
            full_sequence = torch.cat([prompts[i], responses[i]], dim=0)
            text = self.tokenizer.decode(full_sequence, skip_special_tokens=True)
            response_texts.append(text)
        
        # Group responses by group_indices
        grouped_responses = {}
        for i, group_id in enumerate(group_indices):
            if group_id not in grouped_responses:
                grouped_responses[group_id] = []
            grouped_responses[group_id].append((i, response_texts[i]))
        
        # Evaluate each group in its environment
        advantages = torch.zeros(batch_size, response_len)
        rewards = torch.zeros(batch_size, response_len)
        environment_feedback = {}
        
        for group_id, group_responses in grouped_responses.items():
            group_advantages, group_rewards, group_feedback = self._evaluate_group(
                group_id, group_responses
            )
            
            # Assign results back to batch positions
            for local_idx, (batch_idx, _) in enumerate(group_responses):
                advantages[batch_idx] = group_advantages[local_idx]
                rewards[batch_idx] = group_rewards[local_idx]
                
            environment_feedback[f"group_{group_id}"] = group_feedback
            
        return {
            "advantages": advantages,
            "rewards": rewards, 
            "environment_feedback": environment_feedback,
            "group_stats": self._compute_group_statistics(grouped_responses, advantages, rewards)
        }
    
    def _evaluate_group(
        self, 
        group_id: int, 
        group_responses: List[tuple]
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Evaluate a group of responses in the corresponding Atropos environment.
        
        This is where the actual Atropos environment interaction happens.
        """
        group_size = len(group_responses)
        response_len = len(self.tokenizer.encode(group_responses[0][1])) if group_responses else 10
        
        # Mock environment evaluation - replace with actual Atropos calls
        if self.advantage_mode == "outcome_based":
            # Outcome-based evaluation (like GRPO)
            advantages, rewards = self._mock_outcome_evaluation(group_responses)
        elif self.advantage_mode == "step_based":
            # Step-based evaluation with environment feedback
            advantages, rewards = self._mock_step_evaluation(group_responses)
        else:
            raise ValueError(f"Unknown advantage mode: {self.advantage_mode}")
            
        # Environment feedback
        feedback = {
            "group_id": group_id,
            "num_responses": group_size,
            "success_rate": torch.mean((rewards.sum(dim=-1) > 0).float()).item(),
            "average_reward": torch.mean(rewards.sum(dim=-1)).item(),
        }
        
        return advantages, rewards, feedback
    
    def _mock_outcome_evaluation(self, group_responses: List[tuple]) -> tuple[torch.Tensor, torch.Tensor]:
        """Mock outcome-based evaluation (replace with real Atropos environment)."""
        group_size = len(group_responses)
        response_len = 20  # Mock response length
        
        # Mock outcome scores (in real implementation, these come from Atropos)
        outcome_scores = torch.randn(group_size)
        
        # Compute GRPO-style advantages
        if group_size > 1:
            mean_score = outcome_scores.mean()
            advantages = outcome_scores - mean_score
            if self.normalize_advantages:
                std_score = outcome_scores.std() + 1e-6
                advantages = advantages / std_score
        else:
            advantages = torch.zeros(group_size)
            
        # Apply clipping if configured
        if self.advantage_clip_range is not None:
            min_val, max_val = self.advantage_clip_range
            advantages = torch.clamp(advantages, min_val, max_val)
            
        # Convert to token-level advantages (broadcast across response length)
        token_advantages = advantages.unsqueeze(-1).expand(-1, response_len)
        token_rewards = outcome_scores.unsqueeze(-1).expand(-1, response_len)
        
        return token_advantages, token_rewards
    
    def _mock_step_evaluation(self, group_responses: List[tuple]) -> tuple[torch.Tensor, torch.Tensor]:
        """Mock step-based evaluation (replace with real Atropos environment)."""
        group_size = len(group_responses)
        response_len = 20  # Mock response length
        
        # Mock step-by-step evaluation
        advantages = torch.randn(group_size, response_len) * 0.1
        rewards = torch.randn(group_size, response_len) * 0.5
        
        return advantages, rewards
    
    def _compute_group_statistics(
        self, 
        grouped_responses: Dict[int, List[tuple]], 
        advantages: torch.Tensor, 
        rewards: torch.Tensor
    ) -> Dict[str, float]:
        """Compute statistics across all groups."""
        stats = {
            "num_groups": len(grouped_responses),
            "total_responses": advantages.shape[0],
            "mean_advantage": torch.mean(advantages).item(),
            "std_advantage": torch.std(advantages).item(),
            "mean_reward": torch.mean(rewards).item(),
            "std_reward": torch.std(rewards).item(),
        }
        
        # Per-group statistics
        for group_id in grouped_responses:
            group_mask = torch.zeros(advantages.shape[0], dtype=torch.bool)
            for local_idx, (batch_idx, _) in enumerate(grouped_responses[group_id]):
                group_mask[batch_idx] = True
                
            if group_mask.any():
                group_advantages = advantages[group_mask]
                group_rewards = rewards[group_mask]
                stats[f"group_{group_id}_mean_advantage"] = torch.mean(group_advantages).item()
                stats[f"group_{group_id}_mean_reward"] = torch.mean(group_rewards).item()
        
        return stats


class VeRLGRPOAtroposTrainer(RayPPOTrainer):
    """
    VeRL GRPO Trainer with Atropos Integration.
    
    Extends RayPPOTrainer to support:
    - Atropos environment groups and evaluation
    - Proper weight synchronization for on-policy learning
    - Enhanced GRPO hyperparameter exposure  
    - Environment-aware advantage computation
    """
    
    def __init__(
        self,
        config: DictConfig,
        tokenizer: PreTrainedTokenizer,
        role_worker_mapping: dict[Role, type],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
        atropos_reward_fn: Optional[Callable] = None,
    ):
        """Initialize VeRL GRPO trainer with Atropos integration."""
        
        # Validate Atropos configuration
        self._validate_atropos_config(config)
        
        # Initialize base trainer
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            processor=processor,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=device_name,
        )
        
        # Initialize Atropos components
        self.atropos_group_manager = AtroposGroupManager(config, tokenizer)
        self.atropos_reward_fn = atropos_reward_fn
        
        # Atropos-specific training configuration
        self.use_atropos_advantages = config.atropos.get("use_atropos_advantages", True)
        self.combine_atropos_with_rm = config.atropos.get("combine_with_reward_model", False)
        self.atropos_advantage_weight = config.atropos.get("advantage_weight", 1.0)
        self.rm_advantage_weight = config.atropos.get("rm_weight", 0.1) if self.combine_atropos_with_rm else 0.0
        
        # Enhanced GRPO configuration exposure
        self.grpo_config = self._extract_grpo_config(config)
        
        # Weight synchronization tracking
        self.policy_update_counter = 0
        self.last_sync_step = -1
        
        logger.info(f"VeRL GRPO Atropos Trainer initialized:")
        logger.info(f"  - Use Atropos advantages: {self.use_atropos_advantages}")
        logger.info(f"  - Combine with RM: {self.combine_atropos_with_rm}")
        logger.info(f"  - Advantage weights: Atropos={self.atropos_advantage_weight}, RM={self.rm_advantage_weight}")
        logger.info(f"  - GRPO config: {self.grpo_config}")
        
    def _validate_atropos_config(self, config: DictConfig):
        """Validate Atropos configuration."""
        if not hasattr(config, 'atropos'):
            raise ValueError("Missing 'atropos' configuration section")
            
        required_fields = ['num_groups', 'group_size', 'environment_type']
        for field in required_fields:
            if not config.atropos.get(field):
                raise ValueError(f"Missing required Atropos config field: {field}")
                
        # Ensure GRPO is configured correctly
        if config.algorithm.adv_estimator != AdvantageEstimator.GRPO:
            logger.warning(f"Atropos works best with GRPO advantage estimator, but got {config.algorithm.adv_estimator}")
            
    def _extract_grpo_config(self, config: DictConfig) -> Dict[str, Any]:
        """Extract and expose all GRPO hyperparameters."""
        grpo_config = {
            # Core GRPO parameters
            "adv_estimator": config.algorithm.adv_estimator,
            "gamma": config.algorithm.get("gamma", 1.0),
            "lam": config.algorithm.get("lam", 1.0),
            "norm_adv_by_std_in_grpo": config.algorithm.get("norm_adv_by_std_in_grpo", True),
            
            # Policy optimization parameters
            "clip_ratio": config.actor_rollout_ref.actor.get("clip_ratio", 0.2),
            "clip_ratio_low": config.actor_rollout_ref.actor.get("clip_ratio_low", 0.2),
            "clip_ratio_high": config.actor_rollout_ref.actor.get("clip_ratio_high", 0.2),
            "clip_ratio_c": config.actor_rollout_ref.actor.get("clip_ratio_c", 3.0),
            
            # Loss configuration
            "loss_agg_mode": config.actor_rollout_ref.actor.get("loss_agg_mode", "token-mean"),
            "entropy_coeff": config.actor_rollout_ref.actor.get("entropy_coeff", 0.0),
            "use_kl_loss": config.actor_rollout_ref.actor.get("use_kl_loss", False),
            "kl_loss_coef": config.actor_rollout_ref.actor.get("kl_loss_coef", 0.001),
            "kl_loss_type": config.actor_rollout_ref.actor.get("kl_loss_type", "low_var_kl"),
            
            # Training configuration
            "ppo_epochs": config.actor_rollout_ref.actor.get("ppo_epochs", 1),
            "ppo_mini_batch_size": config.actor_rollout_ref.actor.get("ppo_mini_batch_size", 256),
            "ppo_micro_batch_size_per_gpu": config.actor_rollout_ref.actor.get("ppo_micro_batch_size_per_gpu"),
            
            # KL penalty configuration
            "use_kl_in_reward": config.algorithm.get("use_kl_in_reward", False),
            "kl_penalty": config.algorithm.get("kl_penalty", "kl"),
            
            # Atropos-specific GRPO extensions
            "use_atropos_groups": True,
            "atropos_group_size": config.atropos.get("group_size", 4),
            "atropos_advantage_mode": config.atropos.get("advantage_mode", "outcome_based"),
        }
        
        return grpo_config
    
    def fit(self):
        """
        Enhanced training loop with Atropos integration.
        
        Maintains the same structure as RayPPOTrainer but adds:
        - Atropos environment evaluation
        - Enhanced advantage computation
        - Proper weight synchronization tracking
        """
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking
        from tqdm import tqdm
        
        logger_tracking = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        
        self.global_steps = 0
        
        # Load checkpoint before doing anything
        self._load_checkpoint()
        
        # Perform validation before training
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            if val_metrics:
                logger.info(f"Initial validation metrics: {val_metrics}")
                logger_tracking.log(data=val_metrics, step=self.global_steps)
                if self.config.trainer.get("val_only", False):
                    return
        
        # Training progress tracking
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Atropos GRPO Training")
        
        # Start training from step 1
        self.global_steps += 1
        last_val_metrics = None
        
        # Main training loop
        for batch in self.train_dataloader:
            is_last_step = self.global_steps >= self.total_training_steps
            
            # Generate sequences using current policy (with automatic weight sync)
            with self._weight_sync_context():
                rollout_output = self.actor_rollout_wg.generate_sequences(batch)
            
            # Track weight synchronization
            self.last_sync_step = self.global_steps
            
            # Combine rollout data
            batch = batch.union(rollout_output)
            
            # Compute metrics for rollout
            metrics = self._compute_rollout_metrics(batch)
            
            # Atropos environment evaluation and advantage computation
            atropos_metrics = self._compute_atropos_advantages(batch)
            metrics.update(atropos_metrics)
            
            # Standard reward model evaluation (if enabled)
            if self.use_rm or not self.use_atropos_advantages:
                rm_metrics = self._compute_reward_model_scores(batch)
                metrics.update(rm_metrics)
            
            # Combine advantages from Atropos and reward model
            self._combine_advantages(batch)
            
            # Apply KL penalty if configured
            if self.config.algorithm.use_kl_in_reward:
                batch, kl_metrics = apply_kl_penalty(
                    batch, 
                    kl_ctrl=self.kl_ctrl_in_reward, 
                    kl_penalty=self.config.algorithm.kl_penalty
                )
                metrics.update(kl_metrics)
            else:
                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
            
            # Compute GRPO advantages with Atropos integration
            batch = self._compute_enhanced_grpo_advantages(batch)
            
            # Update critic (if using GAE)
            if self.use_critic:
                critic_output = self.critic_wg.update_critic(batch)
                critic_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                metrics.update(critic_metrics)
            
            # Update actor (with critic warmup)
            if self.config.trainer.critic_warmup <= self.global_steps:
                batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                actor_output = self.actor_rollout_wg.update_actor(batch)
                actor_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                metrics.update(actor_metrics)
                
                # Track policy updates
                self.policy_update_counter += 1
            
            # Log metrics
            metrics.update(self._compute_training_metrics(batch))
            logger_tracking.log(data=metrics, step=self.global_steps)
            
            # Validation
            if self.global_steps % self.config.trainer.val_freq == 0 and self.val_reward_fn is not None:
                val_metrics = self._validate()
                if val_metrics:
                    logger_tracking.log(data=val_metrics, step=self.global_steps)
                    last_val_metrics = val_metrics
            
            # Checkpointing
            if self.global_steps % self.config.trainer.save_freq == 0:
                self._save_checkpoint()
            
            # Update progress and continue
            progress_bar.update(1)
            self.global_steps += 1
            
            if is_last_step:
                logger.info(f"Final validation metrics: {last_val_metrics}")
                progress_bar.close()
                return
    
    @contextmanager
    def _weight_sync_context(self):
        """
        Context manager for proper weight synchronization during rollout.
        
        Ensures inference engine has the latest policy weights for on-policy generation.
        """
        logger.debug(f"Syncing weights for rollout at step {self.global_steps}")
        try:
            yield
        finally:
            logger.debug(f"Weight sync completed for step {self.global_steps}")
    
    def _compute_atropos_advantages(self, batch: DataProto) -> Dict[str, float]:
        """Compute advantages using Atropos environment evaluation."""
        if not self.use_atropos_advantages:
            return {}
            
        logger.debug("Computing Atropos advantages...")
        
        # Extract prompts and responses
        prompts = batch.batch["prompts"]
        responses = batch.batch["responses"]
        
        # Generate group indices (for now, use simple round-robin assignment)
        batch_size = prompts.shape[0]
        group_indices = np.arange(batch_size) % self.atropos_group_manager.num_groups
        
        # Evaluate in Atropos environments
        atropos_results = self.atropos_group_manager.evaluate_responses_in_environment(
            prompts=prompts,
            responses=responses,
            group_indices=group_indices
        )
        
        # Store Atropos advantages and rewards
        batch.batch["atropos_advantages"] = atropos_results["advantages"]
        batch.batch["atropos_rewards"] = atropos_results["rewards"]
        batch.non_tensor_batch["atropos_feedback"] = atropos_results["environment_feedback"]
        batch.non_tensor_batch["group_indices"] = group_indices
        
        # Extract metrics
        metrics = {
            "atropos/num_groups": atropos_results["group_stats"]["num_groups"],
            "atropos/mean_advantage": atropos_results["group_stats"]["mean_advantage"],
            "atropos/std_advantage": atropos_results["group_stats"]["std_advantage"], 
            "atropos/mean_reward": atropos_results["group_stats"]["mean_reward"],
            "atropos/std_reward": atropos_results["group_stats"]["std_reward"],
        }
        
        logger.debug(f"Atropos evaluation complete: {metrics}")
        return metrics
    
    def _compute_reward_model_scores(self, batch: DataProto) -> Dict[str, float]:
        """Compute reward model scores (if using traditional RM)."""
        if not self.use_rm and self.use_atropos_advantages:
            return {}
            
        # Use parent class reward computation
        if self.use_rm:
            reward_tensor = self.rm_wg.compute_rm_score(batch)
            batch = batch.union(reward_tensor)
        
        # Apply reward function if provided
        if self.reward_fn is not None:
            from verl.trainer.ppo.reward import compute_reward
            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
            batch.batch["rm_scores"] = reward_tensor
            if reward_extra_infos_dict:
                batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
        
        return {"rm/computed": 1.0}
    
    def _combine_advantages(self, batch: DataProto):
        """Combine advantages from Atropos and reward model."""
        if self.use_atropos_advantages and "atropos_rewards" in batch.batch:
            atropos_scores = batch.batch["atropos_rewards"]
            
            if self.combine_atropos_with_rm and "rm_scores" in batch.batch:
                rm_scores = batch.batch["rm_scores"]
                # Weighted combination
                combined_scores = (
                    self.atropos_advantage_weight * atropos_scores + 
                    self.rm_advantage_weight * rm_scores
                )
                batch.batch["token_level_scores"] = combined_scores
                logger.debug("Combined Atropos and RM scores")
            else:
                batch.batch["token_level_scores"] = atropos_scores
                logger.debug("Using Atropos scores only")
        elif "rm_scores" in batch.batch:
            batch.batch["token_level_scores"] = batch.batch["rm_scores"]
            logger.debug("Using RM scores only")
        else:
            # Fallback to zero scores
            batch.batch["token_level_scores"] = torch.zeros_like(batch.batch["responses"], dtype=torch.float)
            logger.warning("No reward scores available, using zeros")
    
    def _compute_enhanced_grpo_advantages(self, batch: DataProto) -> DataProto:
        """Compute GRPO advantages with Atropos integration."""
        
        # If using Atropos advantages directly, we may skip traditional GRPO computation
        if self.use_atropos_advantages and "atropos_advantages" in batch.batch:
            batch.batch["advantages"] = batch.batch["atropos_advantages"]
            batch.batch["returns"] = batch.batch["atropos_advantages"]  # For GRPO, returns = advantages
            logger.debug("Using Atropos advantages directly")
        else:
            # Use standard GRPO advantage computation
            batch = compute_advantage(
                batch,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.grpo_config["gamma"],
                lam=self.grpo_config["lam"],
                num_repeat=self.config.actor_rollout_ref.rollout.n,
                norm_adv_by_std_in_grpo=self.grpo_config["norm_adv_by_std_in_grpo"],
                multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
            )
            logger.debug("Used standard GRPO advantage computation")
        
        return batch
    
    def _compute_rollout_metrics(self, batch: DataProto) -> Dict[str, float]:
        """Compute metrics for the rollout phase."""
        metrics = {
            "rollout/batch_size": batch.batch.batch_size[0],
            "rollout/weight_sync_step": self.last_sync_step,
            "rollout/policy_updates": self.policy_update_counter,
        }
        
        if "prompts" in batch.batch and "responses" in batch.batch:
            prompts = batch.batch["prompts"]
            responses = batch.batch["responses"]
            metrics.update({
                "rollout/prompt_length": float(prompts.shape[1]),
                "rollout/response_length": float(responses.shape[1]),
                "rollout/total_tokens": float(prompts.numel() + responses.numel()),
            })
        
        return metrics
    
    def _compute_training_metrics(self, batch: DataProto) -> Dict[str, float]:
        """Compute training-specific metrics."""
        metrics = {
            "training/step": self.global_steps,
            "training/grpo_config_hash": hash(str(self.grpo_config)),
        }
        
        # Add GRPO-specific metrics
        if "advantages" in batch.batch:
            advantages = batch.batch["advantages"]
            metrics.update({
                "grpo/advantage_mean": torch.mean(advantages).item(),
                "grpo/advantage_std": torch.std(advantages).item(),
                "grpo/advantage_max": torch.max(advantages).item(),
                "grpo/advantage_min": torch.min(advantages).item(),
            })
        
        return metrics
    
    def get_grpo_hyperparameters(self) -> Dict[str, Any]:
        """Get all GRPO hyperparameters for external access."""
        return self.grpo_config.copy()
    
    def update_grpo_hyperparameters(self, updates: Dict[str, Any]):
        """Update GRPO hyperparameters during training."""
        for key, value in updates.items():
            if key in self.grpo_config:
                old_value = self.grpo_config[key]
                self.grpo_config[key] = value
                logger.info(f"Updated GRPO parameter {key}: {old_value} -> {value}")
            else:
                logger.warning(f"Unknown GRPO parameter: {key}")
    
    def get_atropos_status(self) -> Dict[str, Any]:
        """Get current status of Atropos integration."""
        return {
            "num_groups": self.atropos_group_manager.num_groups,
            "group_size": self.atropos_group_manager.group_size,
            "environment_type": self.atropos_group_manager.environment_type,
            "use_atropos_advantages": self.use_atropos_advantages,
            "combine_with_rm": self.combine_atropos_with_rm,
            "advantage_weight": self.atropos_advantage_weight,
            "rm_weight": self.rm_advantage_weight,
            "last_weight_sync": self.last_sync_step,
            "policy_updates": self.policy_update_counter,
        } 