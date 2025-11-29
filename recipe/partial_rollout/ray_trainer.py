# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Optional
import threading

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.trainer.ppo.ray_trainer import apply_kl_penalty, compute_advantage, compute_response_mask, ResourcePoolManager
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger


class AggregatorActor:
    def __init__(self, threshold_rate):
        self.total = 0
        self.stopped = False
        self.sample_num = 0
        self.threshold_rate = threshold_rate
        self.threshold = 0
        self.prefetch_request_index_lock = threading.Lock()

    def set_sample_num(self, value):
        self.sample_num = value
        self.threshold = int(self.threshold_rate * self.sample_num)

    def clear(self):
        self.total = 0
        self.stopped = False

    def add(self, value):
        with self.prefetch_request_index_lock:
            self.total += value
        if self.total >= self.threshold and not self.stopped:
            self.stopped = True
        return self.stopped

    def is_stopped(self):
        return self.stopped

    def get_total(self):
        return self.total

    def get_threshold(self):
        return self.threshold


def ray_trainer__init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
):
    """
    Initialize distributed PPO trainer with Ray backend.
    Note that this trainer runs on the driver process on a single CPU/GPU node.

    Args:
        config: Configuration object containing training parameters.
        tokenizer: Tokenizer used for encoding and decoding text.
        role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
        resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
        ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
        processor: Optional data processor, used for multimodal data
        reward_fn: Function for computing rewards during training.
        val_reward_fn: Function for computing rewards during validation.
        train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
        val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
        collate_fn: Function to collate data samples into batches.
        train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
        device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
    """

    # Store the tokenizer for text processing
    self.tokenizer = tokenizer
    self.processor = processor
    self.config = config
    self.reward_fn = reward_fn
    self.val_reward_fn = val_reward_fn

    self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
    assert self.hybrid_engine, "Currently, only support hybrid engine"

    if self.hybrid_engine:
        assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

    self.role_worker_mapping = role_worker_mapping
    self.resource_pool_manager = resource_pool_manager
    self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
    self.use_rm = need_reward_model(self.role_worker_mapping)
    self.use_critic = need_critic(self.config)
    self.ray_worker_group_cls = ray_worker_group_cls
    self.device_name = device_name if device_name else self.config.trainer.device
    self.validation_generations_logger = ValidationGenerationsLogger(
        project_name=self.config.trainer.project_name,
        experiment_name=self.config.trainer.experiment_name,
    )

    # if ref_in_actor is True, the reference policy will be actor without lora applied
    self.ref_in_actor = (
            config.actor_rollout_ref.model.get("lora_rank", 0) > 0
            or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
    )

    # define in-reward KL control
    # kl loss control currently not suppoorted
    if self.config.algorithm.use_kl_in_reward:
        self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

    self.aggregator = None
    self.enable_partial_rollout: bool = self.config.actor_rollout_ref.rollout.partial_rollout_max_split > 1
    # check partial rollout config
    assert (config.data.max_response_length %
            config.actor_rollout_ref.rollout.partial_rollout_max_split == 0), \
        "max_response_length must be divisible by partial_rollout_max_split"

    self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)


def ray_trainer_init_workers(self):
    """Initialize distributed training workers using Ray backend.

    Creates:
    1. Ray resource pools from configuration
    2. Worker groups for each role (actor, critic, etc.)
    """
    self.resource_pool_manager.create_resource_pool()

    self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

    if Role.Aggregator in self.role_worker_mapping:
        self.aggregator = self.role_worker_mapping[Role.Aggregator].options(name="aggregator_actor").remote(
            threshold_rate=self.config.actor_rollout_ref.rollout.rollout_threshold_rate)

    # create actor and rollout
    if self.hybrid_engine:
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
        actor_rollout_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.ActorRollout],
            config=self.config.actor_rollout_ref,
            role=str(Role.ActorRollout),
        )
        self.resource_pool_to_cls[resource_pool][str(Role.ActorRollout)] = actor_rollout_cls
    else:
        raise NotImplementedError

    # create critic
    if self.use_critic:
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
        critic_cfg = omega_conf_to_dataclass(self.config.critic)
        critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
        self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

    # create reference policy if needed
    if self.use_reference_policy:
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
        ref_policy_cls = RayClassWithInitArgs(
            self.role_worker_mapping[Role.RefPolicy],
            config=self.config.actor_rollout_ref,
            role=str(Role.RefPolicy),
        )
        self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

    # create a reward model if reward_fn is None
    if self.use_rm:
        # we create a RM here
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
        rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
        self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls

    # initialize WorkerGroup
    # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
    # you should not use `create_colocated_worker_cls`.
    # Instead, directly pass different resource pool to different worker groups.
    # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
    all_wg = {}
    wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
    if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
        wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
    if OmegaConf.select(self.config.global_profiler, "steps") is not None:
        wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
        # Only require nsight worker options when tool is nsys
        if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
            assert (
                OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                is not None
            ), "worker_nsight_options must be set when using nsys with profile_steps"
            wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
            )
    wg_kwargs["device_name"] = self.device_name

    for resource_pool, class_dict in self.resource_pool_to_cls.items():
        worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
        wg_dict = self.ray_worker_group_cls(
            resource_pool=resource_pool,
            ray_cls_with_init=worker_dict_cls,
            **wg_kwargs,
        )
        spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
        all_wg.update(spawn_wg)

    if self.use_critic:
        self.critic_wg = all_wg[str(Role.Critic)]
        self.critic_wg.init_model()

    if self.use_reference_policy and not self.ref_in_actor:
        self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
        self.ref_policy_wg.init_model()

    self.rm_wg = None
    # initalization of rm_wg will be deprecated in the future
    if self.use_rm:
        self.rm_wg = all_wg[str(Role.RewardModel)]
        self.rm_wg.init_model()

    # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
    self.actor_rollout_wg = all_wg[str(Role.ActorRollout)]
    self.actor_rollout_wg.init_model()

    # create async rollout manager and request scheduler
    self.async_rollout_mode = False
    if self.config.actor_rollout_ref.rollout.mode == "async":
        from verl.experimental.agent_loop import AgentLoopManager

        self.async_rollout_mode = True
        self.async_rollout_manager = AgentLoopManager(
            config=self.config, worker_group=self.actor_rollout_wg, rm_wg=self.rm_wg
        )


def ray_trainer_fit(self):
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

    current_epoch = self.global_steps // len(self.train_dataloader)

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
    last_val_metrics = None
    self.max_steps_duration = 0
    partial_batch: Optional[DataProto] = None  # samples whose rollout is not finished yet
    staged_batch: Optional[DataProto] = None  # samples whose rollout has been finished but not yet trained on
    prev_step_profile = False
    curr_step_profile = (
        self.global_steps in self.config.global_profiler.steps
        if self.config.global_profiler.steps is not None
        else False
    )
    next_step_profile = False

    for epoch in range(current_epoch, self.config.trainer.total_epochs):
        for batch_dict in self.train_dataloader:
            metrics = {}
            timing_raw = {}

            with marked_timer("start_profile", timing_raw):
                self._start_profiling(
                    not prev_step_profile and curr_step_profile
                    if self.config.global_profiler.profile_continuous_steps
                    else curr_step_profile
                )
            batch: DataProto = DataProto.from_single_dict(batch_dict)
            batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                     dtype=object)
            # repeat to align with repeated responses in rollout
            batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
            batch.non_tensor_batch["age"] = np.ones(len(batch.batch), dtype=int)
            batch.non_tensor_batch["raw_response_ids"] = np.fromiter(([] for _ in range(len(batch.batch))),
                                                                     dtype=object)

            batch = DataProto.concat([partial_batch, batch]) if partial_batch is not None else batch
            # add uid to batch
            batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
            )

            gen_batch = self._get_gen_batch(batch)

            # pass global_steps to trace
            gen_batch.meta_info["global_steps"] = self.global_steps
            gen_batch_output = gen_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
            )

            is_last_step = self.global_steps >= self.total_training_steps
            with marked_timer("step", timing_raw):
                # generate a batch
                with marked_timer("gen", timing_raw, color="red"):
                    if self.aggregator:
                        self.aggregator.set_sample_num.remote(len(gen_batch))
                        self.aggregator.clear.remote()

                    if not self.async_rollout_mode:
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                    else:
                        gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

                    timing_raw.update(gen_batch_output.meta_info["timing"])
                    gen_batch_output.meta_info.pop("timing", None)
                with marked_timer("filter", timing_raw):
                    batch = batch.union(gen_batch_output)

                    finished_mask = batch.non_tensor_batch.pop("finished")
                    if self.config.actor_rollout_ref.rollout.partial_rollout_mode == "sync":
                        finished_mask = (batch.non_tensor_batch[
                                             "age"] == self.config.actor_rollout_ref.rollout.partial_rollout_max_split) | finished_mask
                    if self.config.actor_rollout_ref.rollout.partial_rollout_mode == "async":
                        finished_mask = ([len(response) >= self.config.actor_rollout_ref.rollout.response_length
                                          for response in
                                          gen_batch_output.non_tensor_batch['raw_response_ids']]) | finished_mask
                    staged_out, partial_batch = DataProto.split_data(batch, finished_mask)
                    staged_out.non_tensor_batch.pop("raw_prompt_ids")
                    staged_out.non_tensor_batch.pop("raw_response_ids")

                    if len(partial_batch.batch) > 0:
                        for key in ("input_ids", "attention_mask", "position_ids"):
                            tmp = partial_batch.batch.pop(key, None)
                            partial_batch.batch[key] = tmp[:, : self.config.data.max_prompt_length]

                        for key in ("prompts", "responses", "rollout_log_probs"):
                            # we don't support rollout_log_probs in this feature branch yet
                            if key in partial_batch.batch:
                                partial_batch.batch.pop(key)
                    else:
                        partial_batch = None

                    # note that we no longer ensure the order of samples in staged_batch
                    staged_batch = DataProto.concat(
                        [staged_batch, staged_out]) if staged_batch is not None else staged_out

                    # prompts whose number of finished rollout is enough can be trained on
                    # while filtering, we ensure sample number is divisible by n_gpus_per_node and as large as possible
                    can_train_mask = np.zeros(len(staged_batch.batch), dtype=bool)
                    id2count = defaultdict(int)
                    required_rollouts = self.config.actor_rollout_ref.rollout.n
                    divisor = self.config.actor_rollout_ref.actor.ppo_mini_batch_size * required_rollouts

                    for uid in staged_batch.non_tensor_batch["uid"]:
                        id2count[uid] += 1
                    assert not id2count or max(
                        id2count.values()) <= required_rollouts, "max number of responses exceeds rollout n"

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

                    batch, staged_batch = DataProto.split_data(staged_batch, can_train_mask)
                    if partial_batch:
                        partial_batch.non_tensor_batch["age"] += 1
                if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                    if self.reward_fn is None:
                        raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                    with marked_timer("gen_max", timing_raw, color="purple"):
                        gen_baseline_batch = deepcopy(gen_batch)
                        gen_baseline_batch.meta_info["do_sample"] = False
                        if not self.async_rollout_mode:
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                        else:
                            gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                        batch = batch.union(gen_baseline_output)
                        # compute reward model score on batch
                        rm_scores = None
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            rm_scores = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(rm_scores)
                        reward_baseline_tensor, _ = compute_reward(batch, self.reward_fn)
                        reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                        keys_to_pop = set(gen_baseline_output.batch.keys())
                        if rm_scores is not None:
                            keys_to_pop.update(rm_scores.batch.keys())
                        batch.pop(batch_keys=list(keys_to_pop))

                        batch.batch["reward_baselines"] = reward_baseline_tensor

                        del rm_scores, gen_baseline_batch, gen_baseline_output


                if "response_mask" not in batch.batch.keys():
                    batch.batch["response_mask"] = compute_response_mask(batch)
                # Balance the number of valid tokens across DP ranks.
                # NOTE: This usually changes the order of data in the `batch`,
                # which won't affect the advantage calculation (since it's based on uid),
                # but might affect the loss calculation (due to the change of mini-batching).
                if self.config.trainer.balance_batch:
                    self._balance_batch(batch, metrics=metrics)

                # compute global_valid tokens
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                with marked_timer("reward", timing_raw, color="yellow"):
                    # compute reward model score
                    if self.use_rm and "rm_scores" not in batch.batch.keys():
                        reward_tensor = self.rm_wg.compute_rm_score(batch)
                        batch = batch.union(reward_tensor)

                    if self.config.reward_model.launch_reward_fn_async:
                        future_reward = compute_reward_async.remote(
                            data=batch, config=self.config, tokenizer=self.tokenizer
                        )
                    else:
                        reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                # Operating Mode Selection:
                # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                    from verl.trainer.ppo.rollout_corr_helper import apply_rollout_correction

                    apply_rollout_correction(
                        batch=batch,
                        rollout_corr_config=rollout_corr_config,
                        policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                    )
                else:  # Recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(
                            loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode
                        )
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)
                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            from verl.utils.debug.metrics import calculate_debug_metrics

                            metrics.update(calculate_debug_metrics(batch))

                assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                if self.use_reference_policy:
                    # compute reference log_prob
                    with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                        if not self.ref_in_actor:
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                        else:
                            ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                        batch = batch.union(ref_log_prob)

                # compute values
                if self.use_critic:
                    with marked_timer("values", timing_raw, color="cyan"):
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)

                with marked_timer("adv", timing_raw, color="brown"):
                    # we combine with rule-based rm
                    reward_extra_infos_dict: dict[str, list]
                    if self.config.reward_model.launch_reward_fn_async:
                        reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                    batch.batch["token_level_scores"] = reward_tensor

                    if reward_extra_infos_dict:
                        batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                    # compute rewards. apply_kl_penalty if available
                    if self.config.algorithm.use_kl_in_reward:
                        batch, kl_metrics = apply_kl_penalty(
                            batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                        )
                        metrics.update(kl_metrics)
                    else:
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                    # Compute rollout correction: IS weights, rejection sampling, and metrics
                    # Only runs in decoupled mode (computes once per batch using stable π_old)
                    # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                    if (
                        rollout_corr_config is not None
                        and "rollout_log_probs" in batch.batch
                        and not bypass_recomputing_logprobs  # Only in decoupled mode
                    ):
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                        # Compute IS weights, apply rejection sampling, compute metrics
                        batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                        # IS and off-policy metrics already have rollout_corr/ prefix
                        metrics.update(is_metrics)

                    # compute advantages, executed on the driver process
                    norm_adv_by_std_in_grpo = self.config.algorithm.get(
                        "norm_adv_by_std_in_grpo", True
                    )  # GRPO adv normalization factor

                    batch = compute_advantage(
                        batch,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                        num_repeat=self.config.actor_rollout_ref.rollout.n,
                        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        config=self.config.algorithm,
                    )

                # update critic
                if self.use_critic:
                    with marked_timer("update_critic", timing_raw, color="pink"):
                        critic_output = self.critic_wg.update_critic(batch)
                    critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                    metrics.update(critic_output_metrics)

                # implement critic warmup
                if self.config.trainer.critic_warmup <= self.global_steps:
                    # update actor
                    with marked_timer("update_actor", timing_raw, color="red"):
                        rollout_config = self.config.actor_rollout_ref.rollout
                        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
                        # TODO: Make "temperature" single source of truth from generation.
                        batch.meta_info["temperature"] = rollout_config.temperature
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
                with marked_timer("testing", timing_raw, color="green"):
                    val_metrics: dict = self._validate()
                    if is_last_step:
                        last_val_metrics = val_metrics
                metrics.update(val_metrics)

            # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
            esi_close_to_expiration = should_save_ckpt_esi(
                max_steps_duration=self.max_steps_duration,
                redundant_time=self.config.trainer.esi_redundant_time,
            )
            # Check if the conditions for saving a checkpoint are met.
            # The conditions include a mandatory condition (1) and
            # one of the following optional conditions (2/3/4):
            # 1. The save frequency is set to a positive value.
            # 2. It's the last training step.
            # 3. The current step number is a multiple of the save frequency.
            # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
            if self.config.trainer.save_freq > 0 and (
                is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
            ):
                if esi_close_to_expiration:
                    print("Force saving checkpoint: ESI instance expiration approaching.")
                with marked_timer("save_checkpoint", timing_raw, color="green"):
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

            steps_duration = timing_raw["step"]
            self.max_steps_duration = max(self.max_steps_duration, steps_duration)

            # training metrics
            metrics.update(
                {
                    "training/global_step": self.global_steps,
                    "training/epoch": epoch,
                }
            )
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
            # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

            # this is experimental and may be changed/removed in the future in favor of a general-purpose one
            if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                self.train_dataloader.sampler.update(batch=batch)

            # TODO: make a canonical logger that supports various backend
            logger.log(data=metrics, step=self.global_steps)

            progress_bar.update(1)
            self.global_steps += 1

            if (
                hasattr(self.config.actor_rollout_ref.actor, "profiler")
                and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
            ):
                self.actor_rollout_wg.dump_memory_snapshot(
                    tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                )

            if is_last_step:
                pprint(f"Final validation metrics: {last_val_metrics}")
                progress_bar.close()
                return

            # this is experimental and may be changed/removed in the future
            # in favor of a general-purpose data buffer pool
            if hasattr(self.train_dataset, "on_batch_end"):
                # The dataset may be changed after each training batch
                self.train_dataset.on_batch_end(batch=batch)






class RayPPOTrainer111:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = (
            config.actor_rollout_ref.model.get("lora_rank", 0) > 0
            or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        )

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.aggregator = None
        self.enable_partial_rollout: bool = self.config.actor_rollout_ref.rollout.partial_rollout_max_split > 1
        # check partial rollout config
        assert (config.data.max_response_length %
                config.actor_rollout_ref.rollout.partial_rollout_max_split == 0), \
            "max_response_length must be divisible by partial_rollout_max_split"

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        if Role.Aggregator in self.role_worker_mapping:
            self.aggregator = self.role_worker_mapping[Role.Aggregator].options(name="aggregator_actor").remote(
                threshold_rate=self.config.actor_rollout_ref.rollout.rollout_threshold_rate)

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role=str(Role.ActorRollout),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.ActorRollout)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
            self.ref_policy_wg.init_model()

        self.rm_wg = None
        # initalization of rm_wg will be deprecated in the future
        if self.use_rm:
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(Role.ActorRollout)]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config, worker_group=self.actor_rollout_wg, rm_wg=self.rm_wg
            )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            # NOTE: while there is no checkpoint to load, we still need to offload the model and optimizer to CPU
            self.actor_rollout_wg.load_checkpoint(None)
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                self.actor_rollout_wg.load_checkpoint(None)
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        workload_lst = calculate_workload(global_seqlen_lst)
        world_size = self.actor_rollout_wg.world_size
        if keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(workload_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(world_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    workload_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=world_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(
                workload_lst, k_partitions=world_size, equal_size=True
            )
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        for idx, partition in enumerate(global_partition_lst):
            partition.sort(key=lambda x: (workload_lst[x], x))
            ordered_partition = partition[::2] + partition[1::2][::-1]
            global_partition_lst[idx] = ordered_partition
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

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

        current_epoch = self.global_steps // len(self.train_dataloader)

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
        last_val_metrics = None
        self.max_steps_duration = 0
        partial_batch: Optional[DataProto] = None  # samples whose rollout is not finished yet
        staged_batch: Optional[DataProto] = None  # samples whose rollout has been finished but not yet trained on
        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                         dtype=object)
                # repeat to align with repeated responses in rollout
                batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                batch.non_tensor_batch["age"] = np.ones(len(batch.batch), dtype=int)
                batch.non_tensor_batch["raw_response_ids"] = np.fromiter(([] for _ in range(len(batch.batch))),
                                                                         dtype=object)

                batch = DataProto.concat([partial_batch, batch]) if partial_batch is not None else batch
                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if self.aggregator:
                            self.aggregator.set_sample_num.remote(len(gen_batch))
                            self.aggregator.clear.remote()

                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)
                    with marked_timer("filter", timing_raw):
                        batch = batch.union(gen_batch_output)

                        finished_mask = batch.non_tensor_batch.pop("finished")
                        if self.config.actor_rollout_ref.rollout.partial_rollout_mode == "sync":
                            finished_mask = (batch.non_tensor_batch[
                                                 "age"] == self.config.algorithm.partial_rollout_max_split) | finished_mask
                        if self.config.actor_rollout_ref.rollout.partial_rollout_mode == "async":
                            finished_mask = ([len(response) >= self.config.actor_rollout_ref.rollout.response_length
                                              for response in
                                              gen_batch_output.non_tensor_batch['raw_response_ids']]) | finished_mask
                        staged_out, partial_batch = DataProto.split_data(batch, finished_mask)
                        staged_out.non_tensor_batch.pop("raw_prompt_ids")
                        staged_out.non_tensor_batch.pop("raw_response_ids")

                        if len(partial_batch.batch) > 0:
                            for key in ("input_ids", "attention_mask", "position_ids"):
                                tmp = partial_batch.batch.pop(key, None)
                                partial_batch.batch[key] = tmp[:, : self.config.data.max_prompt_length]

                            for key in ("prompts", "responses", "rollout_log_probs"):
                                # we don't support rollout_log_probs in this feature branch yet
                                if key in partial_batch.batch:
                                    partial_batch.batch.pop(key)
                        else:
                            partial_batch = None

                        # note that we no longer ensure the order of samples in staged_batch
                        staged_batch = DataProto.concat(
                            [staged_batch, staged_out]) if staged_batch is not None else staged_out

                        # prompts whose number of finished rollout is enough can be trained on
                        # while filtering, we ensure sample number is divisible by n_gpus_per_node and as large as possible
                        can_train_mask = np.zeros(len(staged_batch.batch), dtype=bool)
                        id2count = defaultdict(int)
                        required_rollouts = self.config.actor_rollout_ref.rollout.n
                        divisor = self.config.actor_rollout_ref.actor.ppo_mini_batch_size * required_rollouts

                        for uid in staged_batch.non_tensor_batch["uid"]:
                            id2count[uid] += 1
                        assert not id2count or max(
                            id2count.values()) <= required_rollouts, "max number of responses exceeds rollout n"

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

                        batch, staged_batch = DataProto.split_data(staged_batch, can_train_mask)
                        if partial_batch:
                            partial_batch.non_tensor_batch["age"] += 1
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                rm_scores = self.rm_wg.compute_rm_score(batch)
                                batch = batch.union(rm_scores)
                            reward_baseline_tensor, _ = compute_reward(batch, self.reward_fn)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output


                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                data=batch, config=self.config, tokenizer=self.tokenizer
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_rollout_correction

                        apply_rollout_correction(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # Recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                            entropy_agg = agg_loss(
                                loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode
                            )
                            old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            # Compute IS weights, apply rejection sampling, compute metrics
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            rollout_config = self.config.actor_rollout_ref.rollout
                            batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
                            # TODO: Make "temperature" single source of truth from generation.
                            batch.meta_info["temperature"] = rollout_config.temperature
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
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
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

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
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
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
