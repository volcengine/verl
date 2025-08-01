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

import logging
import time
import warnings
from pprint import pprint

import numpy as np
import ray
from omegaconf import OmegaConf
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from recipe.fully_async_policy.message_queue import MessageQueueClient, QueueSample
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    ResourcePoolManager,
    Role,
    WorkerType,
)
from verl.utils.debug import marked_timer
from verl.utils.tracking import ValidationGenerationsLogger

logger = logging.getLogger(__name__)


@ray.remote
class FullyAsyncTrainer(RayPPOTrainer):
    """
    完全异步的PPO训练器，从MessageQueue获取样本进行训练
    基于OneStepOffRayTrainer的成熟实现改进
    """

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
        assert not self.hybrid_engine

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        if config.critic.enable is not None:
            self.use_critic = bool(config.critic.enable)
        elif self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            warnings.warn(
                "Disabled critic as algorithm.adv_estimator != gae. "
                "If it is not intended, please set critic.enable=True",
                stacklevel=2,
            )
            self.use_critic = False

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)
        self.message_queue_client = None

    def set_message_queue_client(self, message_queue_client: MessageQueueClient):
        """设置消息队列客户端"""
        self.message_queue_client = message_queue_client

    def _validate(self):
        """执行验证 - 参考OneStepOffRayTrainer的验证逻辑"""
        return None

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # 创建actor worker
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.Actor)
        actor_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.Actor],
            config=self.config.actor_rollout_ref,
            role="actor",
        )
        self.resource_pool_to_cls[resource_pool]["actor"] = actor_cls

        # 创建critic worker
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # 创建reference policy worker
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # 创建reward model worker
        if self.use_rm:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # 初始化WorkerGroup - 参考OneStepOffRayTrainer的实现
        all_wg = {}
        wg_kwargs = {}
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

        # 分配worker groups
        self.actor_wg = all_wg["actor"]
        self.actor_wg.init_model()

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        logger.info("FullyAsyncTrainer workers initialized successfully")

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        logger.info("Starting Trainer...")

        if self.message_queue_client is None:
            raise ValueError("MessageQueue client not set. Call set_message_queue_client() first.")

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
        self.max_steps_duration = 0

        # across epoch iterator
        continuous_iterator = self._create_continuous_iterator()

        # Start the first asynchronous generation task.
        batch_data_future = self._async_gen_next_batch(continuous_iterator)

        while batch_data_future is not None:
            metrics = {}
            timing_raw = {}

            do_profile = (
                self.global_steps in self.config.trainer.profile_steps
                if self.config.trainer.profile_steps is not None
                else False
            )
            self._start_profiling(do_profile, timing_raw)

            is_last_step = self.global_steps >= self.total_training_steps

            with marked_timer("step", timing_raw):
                # wait for the previous batch
                with marked_timer("wait_prev_gen", timing_raw, color="red"):
                    epoch, batch, gen_batch_output = batch_data_future.get()
                    timing_raw.update(gen_batch_output.meta_info["timing"])
                    gen_batch_output.meta_info.pop("timing", None)

                # asys next generation (with syns weights from actor to rollout)
                with marked_timer("sync_rollout_weights", timing_raw, color="purple"):
                    if not is_last_step:
                        batch_data_future = self._async_gen_next_batch(continuous_iterator)

                batch = self._post_generate_batch(batch, gen_batch_output, metrics)
                batch, reward_extra_infos_dict = self._process_batch_common(batch, metrics, timing_raw)
                self._log_rollout(batch, reward_extra_infos_dict, timing_raw)
                last_val_metrics = self._validate_metrics(is_last_step, last_val_metrics, metrics, timing_raw)
                self._check_save_checkpoint(is_last_step, timing_raw)

            self._stop_profiling(do_profile, timing_raw)
            self._collect_metrics(batch, epoch, metrics, timing_raw)
            self._post_batch_processing(batch)

            # TODO: make a canonical logger that supports various backend
            logger.log(data=metrics, step=self.global_steps)

            progress_bar.update(1)
            self.global_steps += 1

            if is_last_step:
                pprint(f"Final validation metrics: {last_val_metrics}")
                progress_bar.close()
                return

    def get_statistics(self) -> dict:
        """获取训练统计信息"""
        queue_stats = self.message_queue_client.get_statistics() if self.message_queue_client else {}
        return {
            "global_steps": self.global_steps,
            "processed_samples": self.processed_samples,
            "stale_samples_processed": self.stale_samples_processed,
            "current_param_version": self.current_param_version,
            "param_sync_count": self.param_sync_count,
            "queue_size": queue_stats.get("queue_size", 0),
            "queue_total_produced": queue_stats.get("total_produced", 0),
            "queue_total_consumed": queue_stats.get("total_consumed", 0),
            "queue_dropped_samples": queue_stats.get("dropped_samples", 0),
        }

    def _compute_sample_freshness_metrics(self, batch_samples: list[QueueSample]) -> dict:
        """计算样本新鲜度指标"""
        sample_ages = [self.current_param_version - sample.param_version for sample in batch_samples]
        current_time = time.time()
        sample_latencies = [current_time - sample.timestamp for sample in batch_samples]

        return {
            "freshness/avg_sample_age": np.mean(sample_ages),
            "freshness/max_sample_age": max(sample_ages),
            "freshness/min_sample_age": min(sample_ages),
            "freshness/avg_sample_latency": np.mean(sample_latencies),
            "freshness/max_sample_latency": max(sample_latencies),
            "freshness/stale_samples_ratio": sum(1 for age in sample_ages if age > 1) / len(sample_ages),
        }
