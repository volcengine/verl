# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
# Copyright 2025 Meituan Ltd. and/or its affiliates
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
This trainer supports model-agonistic model initialization with huggingface
"""

import uuid
from pprint import pprint

import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from recipe.one_step_off_policy.utils import need_critic
from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    ResourcePoolManager,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_reference_policy, need_reward_model
from verl.utils.debug import marked_timer
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.tracking import ValidationGenerationsLogger


class GenerationBatchFuture:
    """
    Wrapper class for encapsulating batch generation results
    """

    def __init__(self, epoch, batch, gen_batch_output, future_reward=None):
        """
        :param epoch: current epoch
        :param batch: Input batch data
        :param gen_batch_output: Generated sequences from the main model (DataProtoFuture)
        :param future_reward: Future for reward computation (optional)
        """
        self.epoch = epoch
        self.batch = batch
        self.gen_batch_output = gen_batch_output
        self.future_reward = future_reward

    def get(self):
        """
        Get the actual results by calling get() method on gen_batch_output

        Returns:
            tuple: (epoch, batch, gen_batch_result, future_reward)
                - epoch: Current epoch
                - batch: Original input batch data
                - gen_batch_result: Result from gen_batch_output.get() or gen_batch_output itself
                - future_reward: Future for reward computation if available, else None
        """
        # Call get() method on gen_batch_output if available
        if hasattr(self.gen_batch_output, "get"):
            gen_batch_result = self.gen_batch_output.get()
        else:
            gen_batch_result = self.gen_batch_output

        return self.epoch, self.batch, gen_batch_result, self.future_reward


class OneStepOffRayTrainer(RayPPOTrainer):
    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Dataset | None = None,
        val_dataset: Dataset | None = None,
        collate_fn=None,
        train_sampler: Sampler | None = None,
        device_name="cuda",
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
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to "cuda".
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine

        assert not self.hybrid_engine

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_critic = need_critic(config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _validate(self):
        self.actor_rollout_wg = self.rollout_wg
        ret = super()._validate()
        self.actor_rollout_wg = self.actor_wg
        return ret

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        for role, role_name in [(Role.Actor, "actor"), (Role.Rollout, "rollout")]:
            resource_pool = self.resource_pool_manager.get_resource_pool(role)
            role_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[role],
                config=self.config.actor_rollout_ref,
                role=role_name,
            )
            self.resource_pool_to_cls[resource_pool][role_name] = role_cls

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

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
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.trainer, "steps")
            assert (
                OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                is not None
            ), "worker_nsight_options must be set when profile_steps is set"
            wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
            )

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                device_name=self.device_name,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        self.actor_wg = all_wg["actor"]
        self.rollout_wg = all_wg["rollout"]
        self.actor_wg.init_model()
        self.rollout_wg.init_model()
        self.actor_rollout_wg = self.actor_wg  # to be compatible with the functions that not be modified
        weights_info = self.actor_wg.get_actor_weights_info()[0]
        self.rollout_wg.set_actor_weights_info(weights_info)

        self.create_weight_sync_group()
        self.sync_rollout_weights()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async" and self._is_rollout:
            from verl.workers.rollout.async_server import AsyncLLMServerManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AsyncLLMServerManager(
                config=self.config,
                worker_group=self.rollout_wg,
            )

    def create_weight_sync_group(self):
        master_address = ray.get(self.actor_wg.workers[0]._get_node_ip.remote())
        master_port = ray.get(self.actor_wg.workers[0]._get_free_port.remote())
        world_size = len(self.actor_wg.workers + self.rollout_wg.workers)
        self.actor_wg.create_weight_sync_group(
            master_address,
            master_port,
            0,
            world_size,
        )
        ray.get(
            self.rollout_wg.create_weight_sync_group(
                master_address,
                master_port,
                len(self.actor_wg.workers),
                world_size,
            )
        )

    def sync_rollout_weights(self):
        if not self.hybrid_engine:
            self.actor_wg.sync_rollout_weights()
            ray.get(self.rollout_wg.sync_rollout_weights())

    def _create_continuous_iterator(self):
        """
        Create a continuous data iterator across epoch
        """
        for epoch in range(self.config.trainer.total_epochs):
            iterator = iter(self.train_dataloader)
            for batch_dict in iterator:
                yield epoch, batch_dict

    def _async_gen_next_batch(self, continuous_iterator):
        """
        Call parameter synchronization and asynchronous sequence generation.
        """
        try:
            epoch, batch_dict = next(continuous_iterator)
        except StopIteration:
            return None
        except Exception as e:
            print(f"Error in async_gen_next_batch: {e}")
            return None

        # Create the initial batch from the data loader
        batch = DataProto.from_single_dict(batch_dict)

        # pop those keys for generation
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
        gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

        # sync weights from actor to rollout
        self.sync_rollout_weights()

        # async generation
        gen_batch_output = self.rollout_wg.async_generate_sequences(gen_batch)

        # Launch individual reward computations as each generation completes
        future_reward = None
        if self.config.reward_model.launch_reward_fn_async:
            # Store the object reference and set up callback
            future_reward = self._launch_individual_rewards.remote(
                gen_batch_output, self.config, self.tokenizer, batch.non_tensor_batch
            )

        # Return the original, now-modified `batch` and the `future_reward`
        return GenerationBatchFuture(epoch, batch, gen_batch_output, future_reward)

    @staticmethod
    @ray.remote
    def _launch_individual_rewards(gen_batch_output, config, tokenizer, original_non_tensor_batch):
        # Get generation results
        gen_batch_result = gen_batch_output.get()

        # Repeat non_tensor_batch to match the number of responses
        n = config.actor_rollout_ref.rollout.n
        repeated_non_tensor_batch = {}
        for key, value in original_non_tensor_batch.items():
            repeated_non_tensor_batch[key] = np.repeat(value, n, axis=0)

        # Split into individual responses with preserved non_tensor_batch
        responses_split = []
        for i in range(len(gen_batch_result)):
            response_data = gen_batch_result[i : i + 1]  # Get single response
            # Add repeated non_tensor_batch values
            for key in repeated_non_tensor_batch:
                response_data.non_tensor_batch[key] = repeated_non_tensor_batch[key][i : i + 1]
            responses_split.append(response_data)

        # Launch async reward computation
        reward_futures = [
            compute_reward_async.remote(response_data, config, tokenizer) for response_data in responses_split
        ]

        # Wait for results and combine
        results = ray.get(reward_futures)
        rewards_list = [r[0] for r in results]
        extras_list = [r[1] for r in results]

        combined_reward_tensor = torch.cat(rewards_list, dim=0)
        combined_extras_dict = {}
        if extras_list and extras_list[0]:
            for key in extras_list[0].keys():
                combined_extras_dict[key] = [d[key] for d in extras_list if key in d]

        return combined_reward_tensor, combined_extras_dict

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
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        # across epoch iterator
        continuous_iterator = self._create_continuous_iterator()

        # Start the first asynchronous generation task.
        batch_data_future = self._async_gen_next_batch(continuous_iterator)

        while batch_data_future is not None:
            do_profile = (
                self.global_steps in self.config.global_profiler.steps
                if self.config.global_profiler.steps is not None
                else False
            )
            if do_profile:
                self.actor_wg.start_profile()
                if not self.hybrid_engine:
                    self.rollout_wg.start_profile()
                if self.use_reference_policy:
                    self.ref_policy_wg.start_profile()
                if self.use_critic:
                    self.critic_wg.start_profile()
                if self.use_rm:
                    self.rm_wg.start_profile()

            metrics = {}
            timing_raw = {}
            is_last_step = self.global_steps >= self.total_training_steps

            with marked_timer("step", timing_raw):
                # wait for the previous batch
                with marked_timer("wait_prev_gen", timing_raw, color="red"):
                    epoch, batch, gen_batch_output, future_reward = batch_data_future.get()
                    timing_raw.update(gen_batch_output.meta_info["timing"])
                    gen_batch_output.meta_info.pop("timing", None)

                # asys next generation (with syns weights from actor to rollout)
                with marked_timer("sync_rollout_weights", timing_raw, color="purple"):
                    if not is_last_step:
                        batch_data_future = self._async_gen_next_batch(continuous_iterator)

                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )
                # repeat to align with repeated responses in rollout
                batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                batch = batch.union(gen_batch_output)

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

                with marked_timer("reward", timing_raw, color="yellow"):
                    # compute reward model score
                    if self.use_rm:
                        reward_tensor = self.rm_wg.compute_rm_score(batch)
                        batch = batch.union(reward_tensor)

                    # Use the pre-launched future reward if available
                    if self.config.reward_model.launch_reward_fn_async:
                        # future_reward was already started in _async_gen_next_batch
                        reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                    else:
                        reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                # recompute old_log_probs
                with marked_timer("old_log_prob", timing_raw, color="blue"):
                    old_log_prob = self.actor_wg.compute_log_prob(batch)
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
                        rollout_old_log_probs = batch.batch["rollout_log_probs"]
                        actor_old_log_probs = batch.batch["old_log_probs"]
                        attention_mask = batch.batch["attention_mask"]
                        responses = batch.batch["responses"]
                        response_length = responses.size(1)
                        response_mask = attention_mask[:, -response_length:]

                        rollout_probs = torch.exp(rollout_old_log_probs)
                        actor_probs = torch.exp(actor_old_log_probs)
                        rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                        rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                        rollout_probs_diff_max = torch.max(rollout_probs_diff)
                        rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                        rollout_probs_diff_std = torch.std(rollout_probs_diff)
                        metrics.update(
                            {
                                "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                            }
                        )

                if self.use_reference_policy:
                    # compute reference log_prob
                    with marked_timer("ref", timing_raw, color="olive"):
                        if not self.ref_in_actor:
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                        else:
                            ref_log_prob = self.actor_wg.compute_ref_log_prob(batch)
                        batch = batch.union(ref_log_prob)

                # compute values
                if self.use_critic:
                    with marked_timer("values", timing_raw, color="cyan"):
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)

                with marked_timer("adv", timing_raw, color="brown"):
                    # we combine with rule-based rm
                    reward_extra_infos_dict: dict[str, list]
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

                    # Compute rollout IS weights and mismatch metrics (inherited from RayPPOTrainer)
                    batch, is_metrics = self.compute_rollout_importance_weights_and_add_to_batch(batch)
                    # IS and mismatch metrics already have mismatch/ prefix
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
                        batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                        actor_output = self.actor_wg.update_actor(batch)
                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_output_metrics)

                # Log rollout generations if enabled
                rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                if rollout_data_dir:
                    with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                        inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                        outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                        scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                        self._dump_generations(
                            inputs=inputs,
                            outputs=outputs,
                            scores=scores,
                            reward_extra_infos_dict=reward_extra_infos_dict,
                            dump_path=rollout_data_dir,
                        )

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

            if self.config.trainer.save_freq > 0 and (
                is_last_step or self.global_steps % self.config.trainer.save_freq == 0
            ):
                with marked_timer("save_checkpoint", timing_raw, color="green"):
                    self._save_checkpoint()

            # training metrics
            metrics.update(
                {
                    "training/global_step": self.global_steps,
                    "training/epoch": epoch,
                }
            )
            # collect metrics
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            # TODO: implement actual tflpo and theoretical tflpo
            n_gpus = self.resource_pool_manager.get_n_gpus()
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

            # TODO: make a canonical logger that supports various backend
            logger.log(data=metrics, step=self.global_steps)

            progress_bar.update(1)
            self.global_steps += 1

            if do_profile:
                self.actor_wg.stop_profile()
                if not self.hybrid_engine:
                    self.rollout_wg.stop_profile()
                if self.use_reference_policy:
                    self.ref_policy_wg.stop_profile()
                if self.use_critic:
                    self.critic_wg.stop_profile()
                if self.use_rm:
                    self.rm_wg.stop_profile()

            if is_last_step:
                pprint(f"Final validation metrics: {last_val_metrics}")
                progress_bar.close()
                return
