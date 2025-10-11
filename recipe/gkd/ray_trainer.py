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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import time
import uuid
from collections import deque
from copy import deepcopy
from typing import Optional, Type
from functools import lru_cache

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto, DataProtoFuture
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.metric_utils import (
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.tracking import ValidationGenerationsLogger

from recipe.gkd.teacher import TeacherClient
from recipe.gkd.teacher_utils import get_teacher_knowledge

from verl.trainer.ppo.ray_trainer import Role, ResourcePoolManager, RayPPOTrainer
from verl.utils.torch_dtypes import PrecisionType

WorkerType = Type[Worker]

class GenerationBatchFuture:
    """
    Wrapper class for encapsulating batch generation results
    """

    def __init__(self, epoch, batch, gen_batch_output):
        """
        :param epoch: current epoch
        :param batch: Input batch data
        :param gen_batch_output: Generated sequences from the main model (DataProtoFuture)
        """
        self.epoch = epoch
        self.batch = batch
        self.gen_batch_output = gen_batch_output
        self.teacher_batch_output = None

    def set_teacher_batch_output(self, teacher_batch_output):
        """Set the teacher batch output for this generation batch.

        Args:
            teacher_batch_output: The teacher model's output (DataProtoFuture or raw output)
                to be associated with this generation batch. This will be used for
                distillation or guidance during training.
        """
        self.teacher_batch_output = teacher_batch_output

    def get(self):
        """
        Get the actual results by calling get() method on gen_batch_output

        Returns:
            tuple: (batch, gen_batch_result)
                - batch: Original input batch data
                - gen_batch_result: Result from gen_batch_output.get() or gen_batch_output itself
        """
        # Call get() method on gen_batch_output if available
        if hasattr(self.gen_batch_output, "get"):
            gen_batch_result = self.gen_batch_output.get()
            self.gen_batch_output = gen_batch_result
        else:
            gen_batch_result = self.gen_batch_output

        if self.teacher_batch_output is None:
            return self.epoch, self.batch, gen_batch_result

        if hasattr(self.teacher_batch_output, "get"):
            teacher_batch_result = self.teacher_batch_output.get()
            self.teacher_batch_output = teacher_batch_result
        else:
            teacher_batch_result = self.teacher_batch_output

        return self.epoch, self.batch, gen_batch_result, teacher_batch_result


class OnPolicyDistillTrainer(RayPPOTrainer):
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, and vLLM integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
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
        self.config = config

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert not self.hybrid_engine

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()
        self.use_critic = False
        
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)
        self.teacher_config = self.config.actor_rollout_ref.teacher
        self.n_server_workers = self.teacher_config.n_server_workers
        self.teacher_client = TeacherClient(self.teacher_config.server_ip, self.teacher_config.server_port, 
                                            n_server_workers=self.n_server_workers)

        self.params_dtype = PrecisionType.to_dtype("bfloat16")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_sampler

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

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"

        if self.val_dataset:
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

            assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

            print(
                f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
                f"{len(self.val_dataloader)}"
            )
        else:
            print(f"Size of train dataloader: {len(self.train_dataloader)}")

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = min(self.config.trainer.total_training_steps, total_training_steps)

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

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        # Build Ray classes per pool
        resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # Rollout group
        rollout_pool = self.resource_pool_manager.get_resource_pool(Role.Rollout)
        rollout_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.Rollout],
            config=self.config.actor_rollout_ref,
            role="rollout",
        )
        resource_pool_to_cls[rollout_pool]["rollout"] = rollout_cls

        # Actor group
        actor_pool = self.resource_pool_manager.get_resource_pool(Role.Actor)
        actor_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.Actor],
            config=self.config.actor_rollout_ref,
            role="actor",
        )
        resource_pool_to_cls[actor_pool]["actor"] = actor_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.trainer, "profile_steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.trainer, "profile_steps")
            assert OmegaConf.select(self.config.trainer, "worker_nsight_options") is not None, (
                "worker_nsight_options must be set when profile_steps is set"
            )
            wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                OmegaConf.select(self.config.trainer, "worker_nsight_options")
            )

        for resource_pool, class_dict in resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                device_name=self.device_name,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

            # avoid port conflict
            for role in spawn_wg.values():
                role.init_model()

        self.rollout_wg = all_wg["rollout"]
        self.actor_wg = all_wg["actor"]

        # Initialize both groups
        # self.rollout_wg.init_model()
        # self.actor_wg.init_model()
        weights_info = self.actor_wg.get_actor_weights_info()[0]
        self.rollout_wg.set_actor_weights_info(weights_info)
        from ray.util.collective import collective

        actor_rollout_workers = self.actor_wg.workers + self.rollout_wg.workers
        collective.create_collective_group(
            actor_rollout_workers,
            len(actor_rollout_workers),
            list(range(0, len(actor_rollout_workers))),
            backend="nccl",
            group_name="actor_rollout",
        )

    def sync_rollout_weights(self):
        assert not self.hybrid_engine
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

    def _async_gen_next_batch(self, epoch, batch_dict, sync_before_generation=True):
        """
        Call parameter synchronization and asynchronous sequence generation.
        """
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
        gen_batch.meta_info["global_steps"] = self.global_steps
        gen_batch.meta_info["sampling_params"] = {
            "top_p": self.config.actor_rollout_ref.rollout.temperature
        }
        # sync weights from actor to rollout
        if sync_before_generation:
            self.sync_rollout_weights()
        # Call non-blocking rollout (worker method registered with blocking=False)
        gen_batch_output = self.rollout_wg.async_generate_sequences(gen_batch)
        return GenerationBatchFuture(epoch, batch, gen_batch_output)

    def _async_get_teacher_knowledge(self, future: GenerationBatchFuture):
        """Asynchronously obtain teacher model knowledge for generated sequences.
        
        This method retrieves generated sequences from the future object, adds response length metadata,
        and asynchronously queries the teacher model for knowledge distillation. The teacher model's output
        is set in the future object for subsequent processing.
        
        Args:
            future (GenerationBatchFuture): Future object containing generated sequences and metadata
            
        Returns:
            GenerationBatchFuture: The same future object with teacher knowledge set
            
        Raises:
            RuntimeError: If teacher client initialization fails or knowledge retrieval fails
        """
        _, _, gen_batch_output = future.get()
        gen_batch_output.meta_info["response_length"] = self.config.data.max_response_length

        future.set_teacher_batch_output(
            get_teacher_knowledge(gen_batch_output, self.teacher_client, 
                                  self.n_server_workers, is_async=True))
        return future
    
    def _maybe_reload_rollout_from_ckpt(self, last_ckpt_path: Optional[str]):
        """
        Optionally reload rollout weights from the latest actor checkpoint.
        """
        if not last_ckpt_path:
            return
        try:
            ok = self.rollout_wg.reload_from_checkpoint(last_ckpt_path)
            if not ok:
                print(f"[WARN] Rollout reload_from_checkpoint failed for {last_ckpt_path}")
        except Exception as e:
            print(f"[WARN] Exception reloading rollout from checkpoint: {e}")

    def one_step_off_scheduler(self, continuous_iterator):
        """One-step-off scheduler implementation (version 1) for GKD training with improved pipeline.
        
        This scheduler optimizes the training pipeline by:
        1. Overlapping rollout weight synchronization with teacher knowledge processing
        2. Maintaining consistent timing measurement across iterations
        3. Reducing idle time between generation and knowledge distillation phases
        
        The scheduler maintains the following timing metrics:
        - sync_rollout_weights: Time taken to synchronize rollout weights
        - wait_prev_gen: Time waiting for previous generation to complete
        - wait_prev_teacher: Time waiting for teacher knowledge to be ready
        
        Args:
            continuous_iterator: Iterator providing (epoch, batch_dict) tuples for training
            
        Yields:
            tuple: Contains (batch, gen_batch_output, teacher_batch_output, timing_metrics)
                - batch: Original input batch data
                - gen_batch_output: Generated sequences from main model
                - teacher_batch_output: Knowledge distillation from teacher model
                - timing_metrics: Dictionary of timing measurements
        """
        timing = {}
        for i, (epoch, batch_dict) in enumerate(continuous_iterator):
            if i == 0:
                # sync weights and start first async rollout
                with marked_timer("sync_rollout_weights", timing):
                    fut = self._async_gen_next_batch(epoch, batch_dict)
                # wait for previous rollout finish and start async generate teacher knowledge
                with marked_timer("wait_prev_gen", timing):
                    prev_fut = self._async_get_teacher_knowledge(fut)
                # no yield here, so we will continue to the next loop and enter `else` block
            if i == 1:
                # we don't need to sync weights here because we have not trained the actor yet
                # start second async rollout
                fut = self._async_gen_next_batch(epoch, batch_dict, sync_before_generation=False)
                # wait for generating teacher knowledge finish
                # and get previous result including rollout and teacher knowledge
                with marked_timer("wait_prev_teacher", timing):
                    prev_result = prev_fut.get()
                yield *prev_result, timing

                # start next step from here
                timing = {}
                # wait for previous rollout finish and start async generate teacher knowledge
                with marked_timer("wait_prev_gen", timing):
                    prev_fut = self._async_get_teacher_knowledge(fut)
            else:
                # sync weights and start next async rollout
                with marked_timer("sync_rollout_weights", timing):
                    fut = self._async_gen_next_batch(epoch, batch_dict)
                # wait for generating teacher knowledge finish
                # and get previous result including rollout and teacher knowledge
                with marked_timer("wait_prev_teacher", timing):
                    prev_result = prev_fut.get()
                yield *prev_result, timing

                # start next step from here
                timing = {}
                # wait for previous rollout finish and start async generate teacher knowledge
                with marked_timer("wait_prev_gen", timing):
                    prev_fut = self._async_get_teacher_knowledge(fut)
        
        # for last step
        with marked_timer("wait_prev_teacher", timing):
            prev_result = prev_fut.get()
        yield *prev_result, timing

    def two_step_off_scheduler(self, continuous_iterator):
        """Two-step-off scheduler implementation for GKD training with optimized pipeline.
        
        This scheduler implements a double-buffered pipeline that overlaps:
        1. Sequence generation with teacher knowledge distillation
        2. Weight synchronization with previous batch processing
        
        Key features:
        - Maintains two parallel processing streams (current and previous batches)
        - Overlaps computation and communication where possible
        - Provides consistent timing metrics across iterations
        
        Pipeline stages:
        1. Initialization: Start first generation without teacher processing
        2. Steady state: Alternate between processing teacher knowledge and starting new generation
        3. Final state: Process last batch of teacher knowledge
        
        Timing metrics collected:
        - sync_rollout_weights: Time for weight synchronization between actor and rollout workers
        - wait_prev_prev_teacher: Time waiting for teacher knowledge from two batches ago
        - wait_prev_gen: Time waiting for previous generation to complete
        
        Args:
            continuous_iterator: Iterator providing (epoch, batch_dict) tuples for training
            
        Yields:
            tuple: Contains (batch, gen_batch_output, teacher_batch_output, timing_metrics)
                - batch: Original input batch data
                - gen_batch_output: Generated sequences from main model
                - teacher_batch_output: Knowledge distillation from teacher model
                - timing_metrics: Dictionary of timing measurements
        """
        timing = {}
        for i, (epoch, batch_dict) in enumerate(continuous_iterator):
            if i == 0:
                with marked_timer("sync_rollout_weights", timing):
                    prev_fut = self._async_gen_next_batch(epoch, batch_dict)
                continue
            elif i == 1:
                prev_prev_fut = self._async_get_teacher_knowledge(prev_fut)
                prev_fut = self._async_gen_next_batch(epoch, batch_dict, sync_before_generation=False)
                continue
            elif i == 2:
                with marked_timer("wait_prev_prev_teacher", timing):
                    prev_prev_result = prev_prev_fut.get()
                with marked_timer("wait_prev_gen", timing):
                    prev_prev_fut = self._async_get_teacher_knowledge(prev_fut)
                prev_fut = self._async_gen_next_batch(epoch, batch_dict, sync_before_generation=False)
                yield *prev_prev_result, timing
                timing = {}
            else:
                with marked_timer("wait_prev_prev_teacher", timing):
                    prev_prev_result = prev_prev_fut.get()
                with marked_timer("wait_prev_gen", timing):
                    prev_prev_fut = self._async_get_teacher_knowledge(prev_fut)
                with marked_timer("sync_rollout_weights", timing):
                    prev_fut = self._async_gen_next_batch(epoch, batch_dict)
                yield *prev_prev_result, timing
                timing = {}

        # for second to last step
        with marked_timer("wait_prev_prev_teacher", timing):
            prev_prev_result = prev_prev_fut.get()
        with marked_timer("wait_prev_gen", timing):
            prev_prev_fut = self._async_get_teacher_knowledge(prev_fut)
        yield *prev_prev_result, timing

        # for last step
        with marked_timer("wait_prev_prev_teacher", timing):
            prev_prev_result = prev_prev_fut.get()
        yield *prev_prev_result, timing

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

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, 
                            desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        max_steps_duration = 0

        last_saved_ckpt_path = None

        # Pre-warm: submit the first rollout
        continuous_iterator = self._create_continuous_iterator()

        scheduler_type = self.config.trainer.scheduler

        if scheduler_type == "one_step_off":
            scheduler = self.one_step_off_scheduler(continuous_iterator)
        elif scheduler_type == "two_step_off":
            scheduler = self.two_step_off_scheduler(continuous_iterator)
        else:
            raise TypeError(f"unrecognized scheduler type: {scheduler_type}")

        # Main loop
        while True:
            do_profile = (
                self.global_steps in self.config.trainer.profile_steps
                if self.config.trainer.profile_steps is not None
                else False
            )
            if do_profile:
                self.rollout_wg.start_profile()
                self.actor_wg.start_profile()

            metrics = {}
            timing_raw = {}
            is_last_step = self.global_steps >= self.total_training_steps
                
            with marked_timer("step", timing_raw):
                _, batch, gen_batch_output, teacher_batch_output, schedule_timing = next(scheduler)
                timing_raw.update(schedule_timing)

                gen_timing = gen_batch_output.meta_info.pop("timing", {})
                for k, v in gen_timing.items():
                    if isinstance(v, list):
                        array_v = np.array(v)
                        timing_raw[k + "_mean"] = array_v.mean().item()
                        timing_raw[k + "_min"] = array_v.min().item()
                        timing_raw[k + "_max"] = array_v.max().item()
                        timing_raw[k] = array_v.max().item()
                    else:
                        timing_raw[k] = v

                timing_raw.update(teacher_batch_output.meta_info.pop("timing"))
                
                # Compute statistics of generated response lengths distribution
                response_lens = (gen_batch_output.batch["responses"] != 
                    self.tokenizer.pad_token_id).sum(dim=-1).tolist()
                metrics.update(
                    {
                        "response_seq_len/average": sum(response_lens) / len(response_lens),
                        "response_seq_len/max": max(response_lens),
                        "response_seq_len/min": min(response_lens),
                        "response_seq_len/max_count": response_lens.count(max(response_lens)),
                        "response_seq_len/min_count": response_lens.count(min(response_lens)),
                    }
                )
                
                # Merge generated outputs back
                batch = batch.union(gen_batch_output)

                # Debug print
                one_attention_mask = batch.batch["attention_mask"][0].to(torch.bool)
                one_sentence = batch.batch["input_ids"][0]
                print("INFO:", "generate text done.")
                print("DEBUG:", self.tokenizer.decode(one_sentence[one_attention_mask].tolist()))

                # compute global_valid tokens
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                
                batch = batch.union(teacher_batch_output)

                # # update actor
                # with marked_timer("send_teacher_knowledge", timing_raw, color="red"):
                #     self.actor_wg.send_teacher_knowledge(teacher_batch_output)
                
                # update actor
                with marked_timer("update_actor", timing_raw, color="red"):
                    actor_output = self.actor_wg.update_actor(batch)

                print("INFO:", "update actor done.")
                actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                metrics.update(actor_output_metrics)

                # save model
                if self.config.trainer.save_freq > 0 and (
                    is_last_step
                    or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        try:
                            last_saved_ckpt_path = self.actor_wg.save_checkpoint()
                            self._maybe_reload_rollout_from_ckpt(last_saved_ckpt_path)
                        except Exception as e:
                            print(f"[WARN] Failed to save or reload checkpoint: {e}")

            # Metrics and bookkeeping
            steps_duration = timing_raw["step"]
            max_steps_duration = max(max_steps_duration, steps_duration)
            # training metrics
            metrics["training/global_step"] = self.global_steps
            # collect metrics
            # metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            n_gpus = self.resource_pool_manager.get_n_gpus()
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            # TODO: implement actual tflpo and theoretical tflpo
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

            # this is experimental and may be changed/removed in the future in favor of a general-purpose one
            if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                self.train_dataloader.sampler.update(batch=batch)

            # TODO: make a canonical logger that supports various backend
            logger.log(data=metrics, step=self.global_steps)

            progress_bar.update(1)
            self.global_steps += 1

            if do_profile:
                self.rollout_wg.stop_profile()
                self.actor_wg.stop_profile()

            if is_last_step:
                progress_bar.close()
                return
