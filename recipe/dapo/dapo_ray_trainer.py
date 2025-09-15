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
from collections import defaultdict, deque
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.utils.profiler import marked_timer
from verl.utils.rollout_skip import RolloutSkip

class DataBuffer:
    """Buffer to manage sample-level data conservation using original batch format"""
    
    def __init__(self):
        self.buffer = deque()  # Store DataProto objects
        self.stats = {
            'total_samples_buffered': 0,
            'total_samples_reused': 0,
        }
    
    def get_samples(self, num_needed):
        """Get samples from buffer, return as DataProto"""
        
        if len(self.buffer) == 0:
            return None
        
        collected_samples = []
        samples_collected = 0
        
        while samples_collected < num_needed and len(self.buffer) > 0:
            buffered_data = self.buffer.popleft()
            samples_in_data = len(buffered_data)
            
            if samples_collected + samples_in_data <= num_needed:
                # Use entire buffered data
                collected_samples.append(buffered_data)
                samples_collected += samples_in_data
                self.stats['total_samples_reused'] += samples_in_data
            else:
                # Take partial data and put remainder back
                samples_to_take = num_needed - samples_collected
                used_data = buffered_data[:samples_to_take]
                remainder_data = buffered_data[samples_to_take:]
                
                collected_samples.append(used_data)
                self.buffer.appendleft(remainder_data)  # Put remainder back at front
                samples_collected += samples_to_take
                self.stats['total_samples_reused'] += samples_to_take
                break
        
        if collected_samples:
            result = DataProto.concat(collected_samples)
            return result
        
        return None

    def get_samples_excluding(self, num_needed, exclude_uids):
        """Get up to num_needed samples excluding any whose uid is in exclude_uids."""
        if len(self.buffer) == 0 or num_needed <= 0:
            return None

        collected = []
        reused = 0
        remaining_needed = num_needed

        while remaining_needed > 0 and self.buffer:
            dp = self.buffer.popleft()
            uids = dp.non_tensor_batch.get("uid", None)
            if uids is None:
                # No UID info; treat all as keep
                keep_indices = list(range(len(dp)))
            else:
                uid_list = uids.tolist() if hasattr(uids, "tolist") else list(uids)
                keep_indices = [i for i, uid in enumerate(uid_list) if uid not in exclude_uids]

            if not keep_indices:
                # All items excluded; drop this chunk permanently
                continue

            kept_dp = dp[keep_indices]
            kept_len = len(kept_dp)

            if kept_len <= remaining_needed:
                collected.append(kept_dp)
                reused += kept_len
                remaining_needed -= kept_len
            else:
                used = kept_dp[:remaining_needed]
                remainder = kept_dp[remaining_needed:]
                # Put the remainder back to the front for future calls
                self.buffer.appendleft(remainder)
                collected.append(used)
                reused += len(used)
                remaining_needed = 0
                break

        if collected:
            result = DataProto.concat(collected) if len(collected) > 1 else collected[0]
            self.stats['total_samples_reused'] += reused
            return result

        return None
    
    def size(self):
        total = sum(len(data) for data in self.buffer)
        return total
    
    def get_stats(self):
        return self.stats.copy()

class SmartDataLoader:
    """
    DataLoader wrapper that handles sample-level buffering using DataProto operations

    """
    
    def __init__(self, dataloader, config):
        self.config = config
        self.dataloader = dataloader
        self.dataloader_iter = iter(dataloader)
        self.dataloader_exhausted = False
        self.reached_epoch_end = False         # mark that we hit StopIteration (before refresh)
        if not self.config.enable:
            return
        self.data_history = set()  # track seen data UIDs
        self.unused_buffer = DataBuffer()             # successful but unused samples
        if self.config.keep_filtered:
            self.filtered_buffer = DataBuffer()    # filtered-out samples
        self.current_batch_remainder = None
        print("DEBUG: SmartDataLoader initialized")
        
    def _refresh_dataloader_iter(self):
        """Reset dataloader iterator when exhausted"""
        self.dataloader_iter = iter(self.dataloader)
        self.reached_epoch_end = False

    def _get_fresh_batch(self, allow_refresh=False):
        """
        Try to get a fresh batch.
        - If StopIteration occurs and allow_refresh is False: do NOT refresh; mark reached_epoch_end and return None.
        - If allow_refresh is True: attempt a refresh once; if still empty, mark dataloader_exhausted and return None.
        """
        if self.dataloader_exhausted:
            return None

        try:
            batch_dict = next(self.dataloader_iter)
        except StopIteration:
            self.reached_epoch_end = True
            if not allow_refresh:
                return None
            # Attempt a single refresh
            self._refresh_dataloader_iter()
            try:
                batch_dict = next(self.dataloader_iter)
            except StopIteration:
                print("DEBUG: Dataloader exhausted permanently after refresh")
                self.dataloader_exhausted = True
                return None

        data_proto = DataProto.from_single_dict(batch_dict)
        trajectory_uids = [str(uuid.uuid4()) for _ in range(len(data_proto))]
        data_proto.non_tensor_batch["uid"] = np.array(trajectory_uids, dtype=object)
        return data_proto

    def get_batch_for_generation(self, batch_size):
        """
        Get exactly batch_size samples for generation.
        Loads data in the following order:
        1) Primary buffer (unused trimmed samples from previous calls)
        2) Data from the dataloader
        3) Filtered buffer (filtered-out samples)
            - Will only be used once per epoch before refreshing dataloader
        """

        if not self.config.enable:
            return self._get_fresh_batch(allow_refresh=True)

        collected_data = []
        samples_needed = batch_size
        
        # 1) First, try primary buffer
        buffered_data = self.unused_buffer.get_samples(samples_needed)
        if buffered_data is not None:
            collected_data.append(buffered_data)
            samples_needed -= len(buffered_data)
        
        # 2) Use remainder if available
        if self.current_batch_remainder is not None and samples_needed > 0:
            remainder_size = len(self.current_batch_remainder)
            samples_to_take = min(samples_needed, remainder_size)
            
            if samples_to_take == remainder_size:
                # Use entire remainder
                collected_data.append(self.current_batch_remainder)
                self.current_batch_remainder = None
                samples_needed -= samples_to_take
            else:
                # Use part of remainder
                used_part = self.current_batch_remainder[:samples_to_take]
                self.current_batch_remainder = self.current_batch_remainder[samples_to_take:]
                collected_data.append(used_part)
                samples_needed -= samples_to_take
        
        # 3) Pull fresh data; on epoch end, prefer filtered buffer before refresh
        while samples_needed > 0 and not self.dataloader_exhausted:
            fresh_data = self._get_fresh_batch(allow_refresh=False)
            if fresh_data is None:
                # Hit end of epoch; first consume from filtered buffer excluding seen UIDs
                filtered_data = self.filtered_buffer.get_samples_excluding(samples_needed, self.data_history)
                if filtered_data is not None:
                    collected_data.append(filtered_data)
                    # ensures that the filtered data is only used once per epoch. Discards after second failure
                    self.data_history.update(
                        sample.non_tensor_batch["uid"] 
                        for sample in filtered_data
                    ) if filtered_data else None
                    samples_needed -= len(filtered_data)

                # If still need more, now refresh and try to get fresh data
                if samples_needed > 0:
                    fresh_data = self._get_fresh_batch(allow_refresh=True)
                    if fresh_data is None:
                        # Permanently exhausted; break to final fallback below
                        break

            if fresh_data is not None and samples_needed > 0:
                fresh_size = len(fresh_data)
                if samples_needed >= fresh_size:
                    collected_data.append(fresh_data)
                    samples_needed -= fresh_size
                else:
                    used_part = fresh_data[:samples_needed]
                    self.current_batch_remainder = fresh_data[samples_needed:]
                    collected_data.append(used_part)
                    samples_needed = 0

        if not collected_data:
            print("No data available (primary buffer empty, no remainder, dataloader empty, filtered buffer empty)")
            return None

        result = collected_data[0] if len(collected_data) == 1 else DataProto.concat(collected_data)
        return result

    def return_unused_samples(self, returning_batch):
        """Return unused samples to buffer based on returning_batch directly. Returns to the FRONT of the deque"""
        if returning_batch is None or len(returning_batch) == 0:
            return
        self.unused_buffer.buffer.appendleft(returning_batch)
        self.unused_buffer.stats['total_samples_buffered'] += len(returning_batch)

    def return_filtered_samples(self, filtered_batch):
        """Return filtered-out originals to the filtered buffer. Adds to the BACK of the deque"""
        if filtered_batch is None or len(filtered_batch) == 0:
            return
        self.filtered_buffer.buffer.append(filtered_batch)
        self.filtered_buffer.stats['total_samples_buffered'] += len(filtered_batch)

    def get_stats(self):

        if self.config.keep_filtered:
            return {
                "unused_buffer": self.unused_buffer.get_stats(),
                "filtered_buffer": self.filtered_buffer.get_stats(),
            }
        else:
            return {
                "unused_buffer": self.unused_buffer.get_stats(),
            }

class RayDAPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

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
        self.gen_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()
        # create smart_dataloader
        self.smart_dataloader = SmartDataLoader(self.train_dataloader, self.config.data.databuffer)

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
            while True:
                original_data_proto = self.smart_dataloader.get_batch_for_generation(
                    self.config.data.train_batch_size,
                )
                
                if original_data_proto is None:
                    print("DEBUG: SmartDataLoader returned None, ending epoch")
                    break  # End of epoch

                metrics = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                new_batch: DataProto = deepcopy(original_data_proto)
                num_gen_batches += 1
                # pop those keys for generation
                if "multi_modal_data" in new_batch.non_tensor_batch.keys():
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                    )
                else:
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, "red"):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, "red"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            new_batch = new_batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(new_batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            new_batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    # The new_batch already has UIDs from SmartDataLoader
                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)

                    with marked_timer("reward", timing_raw, "yellow"):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        try:
                            reward_result = self.reward_fn(new_batch, return_dict=True)
                            reward_tensor = reward_result["reward_tensor"]
                            reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
                        except Exception as e:
                            print(f"Error in reward_fn: {e}")
                            reward_tensor = self.reward_fn(new_batch)
                            reward_extra_infos_dict = {}

                        new_batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(
                                new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(
                                kl_metrics
                            )  # TODO: This will be cleared if we use multiple genenration batches
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

                        # Return filtered-out originals to filtered buffer
                        if self.config.data.databuffer.get("keep_filtered", False):
                            filtered_prompt_uids = set(prompt_uid2metric_vals.keys()) - set(kept_prompt_uids)
                            if filtered_prompt_uids:
                                filtered_indices = [
                                    i for i, uid in enumerate(original_data_proto.non_tensor_batch["uid"]) 
                                    if uid in filtered_prompt_uids
                                ]
                                filtered_original_subset = original_data_proto[filtered_indices]
                                self.smart_dataloader.return_filtered_samples(filtered_original_subset)

                        new_batch = new_batch[kept_traj_idxs]
                        batch = new_batch if batch is None else DataProto.concat([batch, new_batch])

                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:
                            print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f"{num_gen_batches=}. Keep generating...")
                                progress_bar.update(1)
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
                            if len(batch) > traj_bsz:
                                returning_batch = batch[traj_bsz:]  # These are the samples being trimmed off
                                batch = batch[:traj_bsz]  # Keep only what we need for training
                                batch.meta_info["global_steps"] = self.global_steps

                                # Use UIDs to select directly from original_data_proto and buffer those originals.
                                returning_uids = set(returning_batch.non_tensor_batch["uid"])
                                print(f"Returning {len(returning_uids)} trimmed samples to buffer")

                                # Indices in the original (non-repeated, non-unioned) data that match the trimmed UIDs
                                returning_indices = [
                                    i for i, uid in enumerate(original_data_proto.non_tensor_batch["uid"])
                                    if uid in returning_uids
                                ]

                                if returning_indices:
                                    returning_original_subset = original_data_proto[returning_indices]
                                    # Put the untouched original samples back into the buffer; no gen_batch needed
                                    self.smart_dataloader.return_unused_samples(returning_original_subset)
                            else:
                                print("DEBUG: No samples to return (batch size exactly matches required size)")
                            
                            
                            # Log smart dataloader statistics
                            dl_stats = self.smart_dataloader.get_stats()
                            total_samples_buffered = dl_stats['unused_buffer']['total_samples_buffered']
                            total_samples_reused = dl_stats['unused_buffer']['total_samples_reused']
                            metrics.update({
                                'data_con/buffer_size': self.smart_dataloader.unused_buffer.size(),
                                'data_con/total_samples_buffered': total_samples_buffered,
                                'data_con/total_samples_reused': total_samples_reused,
                            })

                            if self.config.data.databuffer.get("keep_filtered", False):
                                total_samples_buffered = dl_stats['filtered_buffer']['total_samples_buffered']
                                total_samples_reused = dl_stats['filtered_buffer']['total_samples_reused']
                                metrics.update({
                                    'data_con/filtered_buffer_size': self.smart_dataloader.filtered_buffer.size(),
                                    'data_con/filtered_total_samples_buffered': total_samples_buffered,
                                    'data_con/filtered_total_samples_reused': total_samples_reused,
                                })

                    # === Updating ===

                    batch.batch["response_mask"] = compute_response_mask(batch)

                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, "blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, "olive"):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, "cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

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
