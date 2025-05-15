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
An naive implementation of split placment example
"""

import uuid
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    _timer,
    apply_kl_penalty,
    compute_advantage,
    compute_data_metrics,
    compute_timing_metrics,
)
from verl.utils.metric import reduce_metrics


def fit(self):
    """
    The training loop of PPO.
    The driver process only need to call the compute functions of the worker group through RPC
    to construct the PPO dataflow.
    The light-weight advantage computation is done on the driver process.
    """
    from omegaconf import OmegaConf

    from verl.utils.tracking import Tracking

    logger = Tracking(
        project_name=self.config.trainer.project_name,
        experiment_name=self.config.trainer.experiment_name,
        default_backend=self.config.trainer.logger,
        config=OmegaConf.to_container(self.config, resolve=True),
    )

    self.global_steps = 0

    # load checkpoint before doing anything
    self._load_checkpoint()

    # perform validation before training
    # currently, we only support validation using the reward_function.
    if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
        val_metrics = self._validate()
        pprint(f"Initial validation metrics: {val_metrics}")
        logger.log(data=val_metrics, step=self.global_steps)
        if self.config.trainer.get("val_only", False):
            return

    # we start from step 1
    self.global_steps += 1
    last_val_metrics = None

    for epoch in range(self.config.trainer.total_epochs):
        for batch_dict in self.train_dataloader:
            metrics = {}
            timing_raw = {}

            batch: DataProto = DataProto.from_single_dict(batch_dict)

            # pop those keys for generation
            gen_batch = batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"])
            is_last_step = self.global_steps >= self.total_training_steps

            with _timer("step", timing_raw):
                # generate a batch
                with _timer("gen", timing_raw):
                    gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                    with _timer("gen_max", timing_raw):
                        gen_baseline_batch = deepcopy(gen_batch)
                        gen_baseline_batch.meta_info["do_sample"] = False
                        gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                        batch = batch.union(gen_baseline_output)
                        reward_baseline_tensor = self.reward_fn(batch)
                        reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                        batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                        batch.batch["reward_baselines"] = reward_baseline_tensor

                        del gen_baseline_batch, gen_baseline_output

                batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                # repeat to align with repeated responses in rollout
                batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                batch = batch.union(gen_batch_output)

                # balance the number of valid tokens on each dp rank.
                # Note that this breaks the order of data inside the batch.
                # Please take care when you implement group based adv computation such as GRPO and rloo
                self._balance_batch(batch, metrics=metrics)

                # compute global_valid tokens
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                # recompute old_log_probs
                with _timer("old_log_prob", timing_raw):
                    old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                    batch = batch.union(old_log_prob)

                if self.use_reference_policy:
                    # compute reference log_prob
                    with _timer("ref", timing_raw):
                        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                        batch = batch.union(ref_log_prob)

                # compute values
                if self.use_critic:
                    with _timer("values", timing_raw):
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)

                with _timer("adv", timing_raw):
                    # compute scores. Support both model and function-based.
                    # We first compute the scores using reward model. Then, we call reward_fn to combine
                    # the results from reward model and rule-based results.
                    if self.use_rm:
                        # we first compute reward model score
                        reward_tensor = self.rm_wg.compute_rm_score(batch)
                        batch = batch.union(reward_tensor)

                    # we combine with rule-based rm
                    reward_tensor = self.reward_fn(batch)
                    batch.batch["token_level_scores"] = reward_tensor

                    # compute rewards. apply_kl_penalty if available
                    if self.config.algorithm.use_kl_in_reward:
                        batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                        metrics.update(kl_metrics)
                    else:
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                    # compute advantages, executed on the driver process
                    norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                    batch = compute_advantage(
                        batch,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                        num_repeat=self.config.actor_rollout_ref.rollout.n,
                        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                    )

                # update critic
                if self.use_critic:
                    with _timer("update_critic_call", timing_raw):
                        critic_output = self.critic_wg.update_critic(batch)

                # implement critic warmup
                if self.config.trainer.critic_warmup <= self.global_steps:
                    # update actor
                    with _timer("update_actor_call", timing_raw):
                        actor_output = self.actor_rollout_wg.update_actor(batch)

                # NOTE: make sure you set blocking=False in update_actor and update_crtic in the worker class
                with _timer("update_actor_critic", timing_raw):
                    critic_output = critic_output.get()
                    critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                    metrics.update(critic_output_metrics)

                    actor_output = actor_output.get()
                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_output_metrics)

                # validate
                if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                    with _timer("testing", timing_raw):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                    with _timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

            # collect metrics
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

            # TODO: make a canonical logger that supports various backend
            logger.log(data=metrics, step=self.global_steps)

            if self.global_steps >= self.total_training_steps:
                pprint(f"Final validation metrics: {last_val_metrics}")
                return

            self.global_steps += 1
