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

import threading
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, apply_kl_penalty, compute_advantage, compute_response_mask
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
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
    assert config.data.max_response_length % config.actor_rollout_ref.rollout.partial_rollout_max_split == 0, (
        "max_response_length must be divisible by partial_rollout_max_split"
    )

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
        self.aggregator = (
            self.role_worker_mapping[Role.Aggregator]
            .options(name="aggregator_actor")
            .remote(threshold_rate=self.config.actor_rollout_ref.rollout.rollout_threshold_rate)
        )

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
            batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
            # repeat to align with repeated responses in rollout
            batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
            batch.non_tensor_batch["age"] = np.ones(len(batch.batch), dtype=int)
            batch.non_tensor_batch["raw_response_ids"] = np.fromiter(
                ([] for _ in range(len(batch.batch))), dtype=object
            )

            batch = DataProto.concat([partial_batch, batch]) if partial_batch is not None else batch
            # add uid to batch
            batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)

            gen_batch = self._get_gen_batch(batch)

            # pass global_steps to trace
            gen_batch.meta_info["global_steps"] = self.global_steps
            gen_batch_output = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

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
                        finished_mask = (
                            batch.non_tensor_batch["age"]
                            == self.config.actor_rollout_ref.rollout.partial_rollout_max_split
                        ) | finished_mask
                    if self.config.actor_rollout_ref.rollout.partial_rollout_mode == "async":
                        finished_mask = (
                            [
                                len(response) >= self.config.actor_rollout_ref.rollout.response_length
                                for response in gen_batch_output.non_tensor_batch["raw_response_ids"]
                            ]
                        ) | finished_mask
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
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
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
