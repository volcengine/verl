# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    apply_kl_penalty,
    compute_advantage,
)
from verl.trainer.ppo.reward import compute_reward
from verl.utils.metric import reduce_metrics
from verl.utils.profiler import marked_timer
from verl.utils.rollout_skip import RolloutSkip


def dapo_ray_trainer_fit(self):
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
    self.gen_steps = 0

    # load checkpoint before doing anything
    self._load_checkpoint()

    # perform validation before training
    # currently, we only support validation using the reward_function.
    if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
        val_metrics = self._validate()
        assert val_metrics, f"{val_metrics=}"
        pprint(f"Initial validation metrics: {val_metrics}")
        logger.log(data=val_metrics, step=self.global_steps)
        if self.config.trainer.get("val_only", False):
            return

    if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
        rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
        rollout_skip.wrap_generate_sequences()

    # add tqdm
    progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

    # we start from step 1
    self.global_steps += 1
    self.gen_steps += 1
    last_val_metrics = None

    partial_batch: Optional[DataProto] = None  # samples whose rollout is not finished yet
    staged_batch: Optional[DataProto] = None  # samples whose rollout has been finished but not yet trained on

    prev_step_profile = False
    curr_step_profile = (
        self.global_steps in self.config.global_profiler.steps
        if self.config.global_profiler.steps is not None
        else False
    )
    next_step_profile = False

    timing_raw = defaultdict(float)
    batch = None
    num_prompt_in_batch = 0
    num_gen_batches = 0
    for epoch in range(self.config.trainer.total_epochs):
        for batch_dict in self.train_dataloader:
            metrics = {}

            with marked_timer("start_profile", timing_raw):
                self._start_profiling(
                    not prev_step_profile and curr_step_profile
                    if self.config.global_profiler.profile_continuous_steps
                    else curr_step_profile
                )

            new_batch: DataProto = DataProto.from_single_dict(batch_dict)
            new_batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
            )
            # repeat to align with repeated responses in rollout
            new_batch = new_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
            )  # gen_prompt_bsz * n
            new_batch.non_tensor_batch["age"] = np.ones(len(new_batch.batch), dtype=int)
            new_batch.non_tensor_batch["raw_response_ids"] = np.fromiter(
                ([] for _ in range(len(new_batch.batch))), dtype=object
            )
            new_batch = DataProto.concat([partial_batch, new_batch]) if partial_batch is not None else new_batch

            num_gen_batches += 1
            # pop those keys for generation

            if "multi_modal_data" in new_batch.non_tensor_batch.keys():
                gen_batch = new_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data", "raw_response_ids"],
                )
            else:
                gen_batch = new_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids", "raw_response_ids"],
                )

            gen_batch.non_tensor_batch["age"] = new_batch.non_tensor_batch["age"]

            is_last_step = self.global_steps >= self.total_training_steps

            with marked_timer("step", timing_raw):
                # generate a batch
                with marked_timer("gen", timing_raw, "red"):
                    if self.aggregator:
                        self.aggregator.set_sample_num.remote(len(gen_batch))
                        self.aggregator.clear.remote()
                    if not self.async_rollout_mode:
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                    else:
                        gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                    timing_raw.update(gen_batch_output.meta_info["timing"])
                    gen_batch_output.meta_info.pop("timing", None)
                with marked_timer("filter", timing_raw):
                    new_batch = new_batch.union(gen_batch_output)

                    finished_mask = new_batch.non_tensor_batch.pop("finished")

                    if self.config.actor_rollout_ref.rollout.partial_rollout_mode == "sync":
                        finished_mask = (
                            new_batch.non_tensor_batch["age"] == self.config.algorithm.partial_rollout_max_split
                        ) | finished_mask
                    if self.config.actor_rollout_ref.rollout.partial_rollout_mode == "async":
                        finished_mask = (
                            [
                                len(response) >= self.config.actor_rollout_ref.rollout.response_length
                                for response in gen_batch_output.non_tensor_batch["raw_response_ids"]
                            ]
                        ) | finished_mask

                    staged_out, partial_batch = DataProto.split_data(new_batch, finished_mask)


                    staged_out.non_tensor_batch.pop("raw_prompt_ids")
                    staged_out.non_tensor_batch.pop("raw_response_ids")

                    if len(partial_batch.batch) > 0:
                        raw_prompt_ids = partial_batch.non_tensor_batch["raw_prompt_ids"]
                        raw_response_ids = partial_batch.non_tensor_batch["raw_response_ids"]

                        # breakpoint()
                        for key in ("input_ids", "attention_mask", "position_ids"):
                            tmp = partial_batch.batch.pop(key, None)
                            partial_batch.batch[key] = tmp[:, : self.config.data.max_prompt_length]  # TODO ?

                        for key in ("prompts", "responses", "rollout_log_probs"):
                            # we don't support rollout_log_probs in this feature branch yet
                            if key in partial_batch.batch:
                                partial_batch.batch.pop(key)
                    else:
                        partial_batch = None

                    # note that we no longer ensure the order of samples in staged_batch

                    staged_batch = (
                        DataProto.concat([staged_batch, staged_out]) if staged_batch is not None else staged_out
                    )

                    # prompts whose number of finished rollout is enough can be trained on
                    # while filtering, we ensure sample number is divisible by n_gpus_per_node and as large as possible
                    can_train_mask = np.zeros(len(staged_batch.batch), dtype=bool)
                    id2count = defaultdict(int)
                    required_rollouts = self.config.actor_rollout_ref.rollout.n
                    divisor = self.config.actor_rollout_ref.actor.ppo_mini_batch_size * required_rollouts

                    for uid in staged_batch.non_tensor_batch["uid"]:
                        id2count[uid] += 1
                    assert not id2count or max(id2count.values()) <= required_rollouts, (
                        "max number of responses exceeds rollout n"
                    )

                    complete_uids = [uid for uid, count in id2count.items() if count == required_rollouts]

                    total_complete_samples = len(complete_uids) * required_rollouts
                    max_usable_groups = (total_complete_samples // divisor) * divisor // required_rollouts
                    can_train_count = max_usable_groups * required_rollouts

                    if can_train_count == 0:
                        print(f"{total_complete_samples=}, no complete uid groups available. Keep generating...")
                        continue

                    selected_uids = set(complete_uids[:max_usable_groups])

                    for i, uid in enumerate(staged_batch.non_tensor_batch["uid"]):
                        if uid in selected_uids:
                            can_train_mask[i] = True

                    new_batch, staged_batch = DataProto.split_data(staged_batch, can_train_mask)

                if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                    with marked_timer("gen_max", timing_raw, "red"):
                        gen_baseline_batch = deepcopy(gen_batch)
                        gen_baseline_batch.meta_info["do_sample"] = False
                        gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)

                        new_batch = new_batch.union(gen_baseline_output)
                        # compute reward model score on new_batch
                        rm_scores = None
                        if self.use_rm and "rm_scores" not in new_batch.batch.keys():
                            rm_scores = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(rm_scores)
                        reward_baseline_tensor, _ = compute_reward(new_batch, self.reward_fn)
                        reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                        keys_to_pop = set(gen_baseline_output.batch.keys())
                        if rm_scores is not None:
                            keys_to_pop.update(rm_scores.batch.keys())
                        new_batch.pop(batch_keys=list(keys_to_pop))

                        new_batch.batch["reward_baselines"] = reward_baseline_tensor

                        del rm_scores, gen_baseline_batch, gen_baseline_output

                if self.config.algorithm.use_kl_in_reward:
                    # We need these metrics for apply_kl_penalty if using kl in reward
                    new_batch = self.compute_kl_related_metrics(new_batch, metrics, timing_raw)
                    # otherwise, we will compute those after dynamic sampling

                with marked_timer("reward", timing_raw, "yellow"):
                    # compute scores. Support both model and function-based.
                    # We first compute the scores using reward model. Then, we call reward_fn to combine
                    # the results from reward model and rule-based results.
                    if self.use_rm and "rm_scores" not in new_batch.batch.keys():
                        # we first compute reward model score
                        reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                        new_batch = new_batch.union(reward_tensor)

                    # we combine with rule-based rm
                    reward_tensor, reward_extra_infos_dict = compute_reward(new_batch, self.reward_fn)

                    new_batch.batch["token_level_scores"] = reward_tensor

                    if reward_extra_infos_dict:
                        new_batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                    # compute rewards. apply_kl_penalty if available
                    if self.config.algorithm.use_kl_in_reward:
                        new_batch, kl_metrics = apply_kl_penalty(
                            new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                        )
                        metrics.update(kl_metrics)  # TODO: This will be cleared if we use multiple genenration batches
                    else:
                        new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

                if not self.config.algorithm.filter_groups.enable:
                    batch = new_batch
                else:  # NOTE: When prompts after filtering is less than train batch size,
                    # we skip to the next generation batch
                    metric_name = self.config.algorithm.filter_groups.metric
                    if metric_name == "seq_final_reward":
                        # Turn to numpy for easier filtering
                        new_batch.non_tensor_batch["seq_final_reward"] = (
                            new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                        )
                    elif metric_name == "seq_reward":
                        new_batch.non_tensor_batch["seq_reward"] = (
                            new_batch.batch["token_level_scores"].sum(dim=-1).numpy()
                        )

                    # Collect the sequence reward for each trajectory
                    prompt_uid2metric_vals = defaultdict(list)
                    for uid, metric_val in zip(
                        new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name], strict=True
                    ):
                        prompt_uid2metric_vals[uid].append(metric_val)

                    prompt_uid2metric_std = {}
                    for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                        prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

                    kept_prompt_uids = [
                        uid
                        for uid, std in prompt_uid2metric_std.items()
                        if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
                    ]
                    num_prompt_in_batch += len(kept_prompt_uids)

                    kept_traj_idxs = []
                    for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
                        if traj_from_prompt_uid in kept_prompt_uids:
                            kept_traj_idxs.append(idx)

                    new_batch = new_batch[kept_traj_idxs]
                    batch = new_batch if batch is None else DataProto.concat([batch, new_batch])

                    prompt_bsz = self.config.data.train_batch_size
                    if num_prompt_in_batch < prompt_bsz:
                        print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
                        max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                        if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                            print(f"{num_gen_batches=}. Keep generating...")
                            self.gen_steps += 1
                            is_last_step = self.global_steps >= self.total_training_steps
                            continue
                        else:
                            raise ValueError(
                                f"{num_gen_batches=} >= {max_num_gen_batches=}."
                                + " Generated too many. Please check if your data are too difficult."
                                + " You could also try set max_num_gen_batches=0 to enable endless trials."
                            )
                    else:
                        # Align the batch
                        traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                        batch = batch[:traj_bsz]
                        if partial_batch:
                            partial_batch.non_tensor_batch["age"] += 1

                # === Updating ===
                # Balance the number of valid tokens across DP ranks.
                # NOTE: This usually changes the order of data in the `batch`,
                # which won't affect the advantage calculation (since it's based on uid),
                # but might affect the loss calculation (due to the change of mini-batching).
                # TODO: Decouple the DP balancing and mini-batching.
                if self.config.trainer.balance_batch:
                    self._balance_batch(batch, metrics=metrics)

                # compute global_valid tokens
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                if not self.config.algorithm.use_kl_in_reward:
                    batch = self.compute_kl_related_metrics(batch, metrics, timing_raw)

                # compute values
                if self.use_critic:
                    with marked_timer("values", timing_raw, "cyan"):
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)

                # Compute rollout correction weights and off-policy metrics (inherited from RayPPOTrainer)
                from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                if rollout_corr_config is not None and "rollout_log_probs" in batch.batch:
                    batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                    # IS and off-policy metrics already have rollout_corr/ prefix
                    metrics.update(is_metrics)

                with marked_timer("adv", timing_raw, "brown"):
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
                    with marked_timer("update_critic", timing_raw, "pink"):
                        critic_output = self.critic_wg.update_critic(batch)
                    critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                    metrics.update(critic_output_metrics)

                # implement critic warmup
                if self.config.trainer.critic_warmup <= self.global_steps:
                    # update actor
                    with marked_timer("update_actor", timing_raw, "red"):
                        actor_output = self.actor_rollout_wg.update_actor(batch)
                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_output_metrics)

                # Log rollout generations if enabled
                rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                if rollout_data_dir:
                    self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

            # validate
            if (
                self.val_reward_fn is not None
                and self.config.trainer.test_freq > 0
                and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
            ):
                with marked_timer("testing", timing_raw, "green"):
                    val_metrics: dict = self._validate()
                    if is_last_step:
                        last_val_metrics = val_metrics
                metrics.update(val_metrics)

            if self.config.trainer.save_freq > 0 and (
                is_last_step or self.global_steps % self.config.trainer.save_freq == 0
            ):
                with marked_timer("save_checkpoint", timing_raw, "green"):
                    self._save_checkpoint()

            with marked_timer("stop_profile", timing_raw):
                next_step_profile = (
                    self.global_steps + 1 in self.config.global_profiler.steps
                    if self.config.global_profiler.steps is not None
                    else False
                )
                self._stop_profiling(
                    curr_step_profile and not next_step_profile
                    if self.config.global_profiler.profile_continuous_steps
                    else curr_step_profile
                )
                prev_step_profile = curr_step_profile
                curr_step_profile = next_step_profile
            if self.enable_partial_rollout:
                metrics.update(
                    {
                        "training/can_train_count": can_train_count,
                        "training/total_complete_samples": total_complete_samples,
                    }
                )

            # collect metrics
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            # TODO: implement actual tflpo and theoretical tflpo
            n_gpus = self.resource_pool_manager.get_n_gpus()
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
            timing_raw = defaultdict(float)  # clear timing

            metrics["train/num_gen_batches"] = num_gen_batches
            batch = None
            num_prompt_in_batch = 0
            num_gen_batches = 0

            # TODO: make a canonical logger that supports various backend
            logger.log(data=metrics, step=self.global_steps)

            if is_last_step:
                pprint(f"Final validation metrics: {last_val_metrics}")
                progress_bar.close()
                return

            progress_bar.update(1)
            self.global_steps += 1
            self.gen_steps += 1
    # check if last step checkpint exists
    checkpoint_dir = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
    if not os.path.exists(checkpoint_dir):
        # save last step checkpoint
        timing_raw = defaultdict(float)
        with marked_timer("save_checkpoint", timing_raw, "green"):
            self._save_checkpoint()
        metrics = {f"timing/{k}": v for k, v in timing_raw.items()}
        logger.log(data=metrics, step=self.global_steps)
