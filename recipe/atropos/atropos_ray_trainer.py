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
Atropos Ray Trainer with Environment Integration

This trainer integrates with Atropos environments to provide:
- Real environment feedback for advantages
- Token-level advantage overrides
- Proper rollout management through Atropos API
"""

import logging
import uuid
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import compute_advantage
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.debug import marked_timer

from .atropos_api_client import AtroposAPIClient, AtroposAPIError, AtroposConfig

logger = logging.getLogger(__name__)


class RayAtroposTrainer(RayPPOTrainer):
    """
    Ray-based trainer for Atropos integration.

    This trainer extends RayPPOTrainer to integrate with Atropos environments,
    using real environment feedback for advantage computation.
    """

    def __init__(self, config, role_worker_mapping, resource_pool_manager):
        super().__init__(config, role_worker_mapping, resource_pool_manager)

        # Initialize Atropos API client
        atropos_config = AtroposConfig(
            api_url=config.trainer.atropos.get("api_url", "http://localhost:9001"),
            timeout=config.trainer.atropos.get("timeout", 30),
            batch_size=config.actor_rollout_ref.rollout.batch_size,
            max_token_len=config.actor_rollout_ref.rollout.max_length,
            wandb_group=config.trainer.project_name,
            wandb_project=config.trainer.experiment_name,
            checkpoint_dir=config.trainer.default_local_dir,
            save_checkpoint_interval=config.trainer.save_freq,
            retry_attempts=config.trainer.atropos.get("retry_attempts", 10),
            retry_delay=config.trainer.atropos.get("retry_delay", 0.5),
            max_wait_time=config.trainer.atropos.get("max_wait_time", 30.0),
        )
        self.atropos_client = AtroposAPIClient(atropos_config)

        # Test connectivity
        if not self.atropos_client.test_connectivity():
            raise AtroposAPIError(f"Cannot connect to Atropos API at {atropos_config.api_url}. Please ensure the Atropos server is running.")

        # Register trainer with Atropos
        try:
            success = self.atropos_client.register_trainer(starting_step=0, num_steps=self.total_training_steps)
            if not success:
                raise AtroposAPIError("Failed to register trainer with Atropos")

            # Get environment information
            env_info = self.atropos_client.get_environment_info()
            logger.info(f"Available Atropos environments: {env_info}")

        except AtroposAPIError as e:
            logger.error(f"Atropos setup failed: {e}")
            raise

        # Atropos-specific configuration
        self.use_atropos_advantages = config.trainer.atropos.get("use_advantages", True)
        self.fallback_to_grpo = config.trainer.atropos.get("fallback_to_grpo", True)

        logger.info(f"Atropos trainer initialized with API at {atropos_config.api_url}")

    def fit(self):
        """
        Training loop with Atropos integration.

        We override the base fit method to inject Atropos advantage computation
        while maintaining all the GPU training infrastructure.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger_instance = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # Load checkpoint
        self._load_checkpoint()

        # Validation before training
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            logger_instance.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # Progress bar
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Atropos Training")

        self.global_steps += 1

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                do_profile = self.global_steps in (self.config.trainer.profile_steps or [])

                if do_profile:
                    self._start_profiling()

                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # Pop keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "interaction_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("interaction_kwargs")

                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                with marked_timer("step", timing_raw):
                    # Generate rollouts using GPU-based workers
                    with marked_timer("gen", timing_raw, color="red"):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info.get("timing", {}))
                        gen_batch_output.meta_info.pop("timing", None)

                    # Add unique IDs for tracking
                    batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)

                    # Repeat to align with repeated responses
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    # Compute response mask
                    from verl.trainer.ppo.ray_trainer import compute_response_mask

                    batch.batch["response_mask"] = compute_response_mask(batch)

                    # Balance batch if needed
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # Compute rewards and advantages
                    with marked_timer("reward_and_advantages", timing_raw, color="yellow"):
                        # Compute reward model score if using RM
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # Use reward function if provided
                        if self.reward_fn is not None:
                            from verl.trainer.ppo.reward import compute_reward

                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
                            batch.batch["token_level_scores"] = reward_tensor
                            if reward_extra_infos_dict:
                                batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # Submit to Atropos and get advantages
                        advantages = self._compute_advantages_with_atropos(batch)
                        batch.batch["advantages"] = advantages

                    # Compute old log probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)

                    # Compute reference log probs if needed
                    if self.use_reference_policy:
                        with marked_timer("ref", timing_raw, color="olive"):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # Compute values if using critic
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    # Apply KL penalty if configured
                    if self.config.algorithm.use_kl_in_reward and self.use_reference_policy:
                        from verl.trainer.ppo.ray_trainer import apply_kl_penalty

                        batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                        metrics.update(kl_metrics)
                    else:
                        batch.batch["token_level_rewards"] = batch.batch.get("token_level_scores", torch.zeros_like(batch.batch["advantages"]))

                    # Update policy using GPU workers
                    with marked_timer("update", timing_raw, color="green"):
                        update_info = self.actor_rollout_wg.update_policy(batch)
                        update_info.meta_info.pop("timing", None)

                    # Compute metrics
                    from verl.trainer.ppo.metric_utils import (
                        compute_data_metrics,
                    )

                    data_metrics = compute_data_metrics(batch, log_ptx_loss=self.config.algorithm.enable_ptx_loss)
                    metrics.update(data_metrics)
                    metrics.update(update_info.meta_info)

                    # Add Atropos-specific metrics
                    if hasattr(self, "_last_atropos_batch_data"):
                        atropos_metrics = self._compute_atropos_metrics(self._last_atropos_batch_data)
                        metrics.update(atropos_metrics)

                    # Log metrics
                    logger_instance.log(data=metrics, step=self.global_steps)

                # Update progress
                progress_bar.update(1)
                self.global_steps += 1

                # Validation
                if self.global_steps % self.config.trainer.eval_freq == 0:
                    val_metrics = self._validate()
                    if val_metrics:
                        logger_instance.log(data=val_metrics, step=self.global_steps)

                # Save checkpoint
                if self.global_steps % self.config.trainer.save_freq == 0:
                    self._save_checkpoint()

                if do_profile:
                    self._stop_profiling()

                if self.global_steps >= self.total_training_steps:
                    break

            if self.global_steps >= self.total_training_steps:
                break

        progress_bar.close()
        logger_instance.finish()

    def _compute_advantages_with_atropos(self, batch: DataProto) -> torch.Tensor:
        """
        Compute advantages using Atropos environments.

        This method:
        1. Extracts prompts and responses from batch
        2. Submits to Atropos API
        3. Retrieves token-level advantages
        4. Falls back to standard computation if needed
        """
        # Extract necessary data
        input_ids = batch.batch["input_ids"]
        responses = batch.batch["responses"]
        attention_mask = batch.batch["attention_mask"]

        # Calculate prompt length
        response_length = responses.shape[1]
        prompt_length = input_ids.shape[1] - response_length
        prompts = input_ids[:, :prompt_length]

        # Get tokenizer from actor worker group
        tokenizer = self.tokenizer

        # Get scores if available
        scores = None
        if "token_level_scores" in batch.batch:
            # Sum token-level scores to get response-level scores
            scores = batch.batch["token_level_scores"].sum(dim=-1).tolist()

        try:
            # Process rollout through Atropos
            result = self.atropos_client.process_rollout_data(
                prompts=prompts,
                responses=responses,
                tokenizer=tokenizer,
                scores=scores,
                log_probs=batch.batch.get("response_log_probs"),
                ref_model=None,  # We compute ref log probs separately
            )

            if result is not None and self.use_atropos_advantages:
                advantages = result["advantages"]
                logger.info(f"Retrieved advantages from Atropos for {result['processed_count']} sequences")

                # Store batch data for metrics
                self._last_atropos_batch_data = result["batch_data"]

                # Ensure advantages are on the correct device
                device = input_ids.device
                if advantages.device != device:
                    advantages = advantages.to(device)

                return advantages
            else:
                logger.warning("No advantages received from Atropos, falling back to standard computation")

        except Exception as e:
            logger.error(f"Error computing advantages with Atropos: {e}")

        # Fallback to standard advantage computation
        if self.fallback_to_grpo:
            logger.info("Using fallback advantage computation")
            # Use the grpo_atropos estimator
            advantages = compute_advantage(
                rewards=batch.batch.get("token_level_rewards", batch.batch.get("token_level_scores", torch.zeros_like(attention_mask, dtype=torch.float32))),
                values=batch.batch.get("values") if self.use_critic else None,
                response_length=response_length,
                advantages=None,
                gamma=self.config.algorithm.gamma,
                gae_lambda=self.config.algorithm.lam,
                adv_estimator="grpo_atropos",
                old_rewards=batch.batch.get("old_rewards"),
                ref_log_probs=batch.batch.get("ref_log_probs"),
                log_probs=batch.batch.get("old_log_probs"),
                kl_penalties=batch.batch.get("kl_penalties"),
                kl_rewards=batch.batch.get("kl_rewards"),
                response_mask=batch.batch.get("response_mask"),
                uid=batch.non_tensor_batch.get("uid"),
            )
            return advantages
        else:
            raise AtroposAPIError("Failed to compute advantages with Atropos and fallback disabled")

    def _compute_atropos_metrics(self, batch_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute metrics specific to Atropos integration"""
        metrics = {}

        if not batch_data:
            return metrics

        # Extract environment-specific metrics
        env_scores = defaultdict(list)
        env_advantages = defaultdict(list)

        for item in batch_data:
            if "env_name" in item:
                env_name = item["env_name"]
                if "scores" in item:
                    env_scores[env_name].append(item["scores"])
                if "advantages" in item and item["advantages"] is not None:
                    env_advantages[env_name].extend(item["advantages"])

        # Aggregate by environment
        for env_name, scores in env_scores.items():
            if scores:
                metrics[f"atropos/{env_name}/mean_score"] = np.mean(scores)
                metrics[f"atropos/{env_name}/std_score"] = np.std(scores)

        for env_name, advantages in env_advantages.items():
            if advantages:
                metrics[f"atropos/{env_name}/mean_advantage"] = np.mean(advantages)
                metrics[f"atropos/{env_name}/std_advantage"] = np.std(advantages)

        # Overall metrics
        all_scores = [s for scores in env_scores.values() for s in scores]
        if all_scores:
            metrics["atropos/mean_score"] = np.mean(all_scores)
            metrics["atropos/num_environments"] = len(env_scores)

        return metrics

    def _start_profiling(self):
        """Start profiling on all worker groups"""
        self.actor_rollout_wg.start_profile()
        if self.use_reference_policy:
            self.ref_policy_wg.start_profile()
        if self.use_critic:
            self.critic_wg.start_profile()
        if self.use_rm:
            self.rm_wg.start_profile()

    def _stop_profiling(self):
        """Stop profiling on all worker groups"""
        self.actor_rollout_wg.stop_profile()
        if self.use_reference_policy:
            self.ref_policy_wg.stop_profile()
        if self.use_critic:
            self.critic_wg.stop_profile()
        if self.use_rm:
            self.rm_wg.stop_profile()
