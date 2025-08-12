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
from typing import Any

import numpy as np
import ray
from omegaconf import OmegaConf

from recipe.fully_async_policy.message_queue import MessageQueueClient, QueueSample
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    ResourcePoolManager,
    Role,
    WorkerType,
)
from verl.utils.debug import marked_timer

logger = logging.getLogger(__name__)


@ray.remote
class FullyAsyncTrainer(RayPPOTrainer):
    """
    A fully asynchronous PPO trainer that obtains samples from a MessageQueue for training.
    Based on an improved implementation of OneStepOffRayTrainer
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
        device_name=None,
    ):
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

        self.message_queue_client = None
        self.param_synchronizer = None

        # Statistics
        self.processed_samples = 0
        self.stale_samples_processed = 0
        self.current_param_version = 0

    def set_message_queue_client(self, message_queue_client: MessageQueueClient):
        """Set message queue client"""
        self.message_queue_client = message_queue_client

    def set_parameter_synchronizer(self, param_synchronizer):
        """Set parameter synchronizer"""
        self.param_synchronizer = param_synchronizer

    def get_actor_wg(self):
        """Get actor worker group"""
        return self.actor_wg

    def _get_samples_from_queue(self) -> tuple[None, None] | tuple[int, Any]:
        """
        Get samples from message queue and compose gen_batch_output

        Returns:
            tuple: (epoch, batch_dict, gen_batch_output)
        """

        # Calculate the number of samples needed
        n_responses_per_prompt = self.config.actor_rollout_ref.rollout.n
        batch_size = self.config.data.train_batch_size
        required_samples = n_responses_per_prompt * batch_size

        print(
            f"Requesting {required_samples} samples from queue (n={n_responses_per_prompt}, batch_size={batch_size})",
            flush=True,
        )

        # Get samples from queue
        consumer_start = time.time()
        queue_samples = self.message_queue_client.get_samples(min_batch_count=required_samples)
        consumer_end = time.time()

        if not queue_samples or len(queue_samples) == 0:
            logger.warning("required_samples is empty")
            return None, None

        print(f"Retrieved {len(queue_samples)} samples from queue. wait time {consumer_end - consumer_start}")

        queue_samples = [ray.cloudpickle.loads(x) for x in queue_samples]
        print(queue_samples)

        # Assemble batch
        batch = self._assemble_gen_batch_output_from_queue_samples(queue_samples)

        print("=" * 200)
        print(batch)

        return 0, batch

    def _assemble_gen_batch_output_from_queue_samples(self, queue_samples: list[QueueSample]):
        """
        Assemble gen_batch_output from queue samples

        Args:
            queue_samples: List of samples from queue
            n_responses_per_prompt: Number of responses per prompt
            batch_size: Batch size

        Returns:
            DataProto: Assembled gen_batch_output
        """
        import numpy as np

        from verl.protocol import DataProto

        if not queue_samples:
            raise ValueError("Empty queue_samples provided for batch assembly")

        print(f"Assembling batch from {len(queue_samples)} queue samples")

        # Extract data and metadata from all samples
        sample_data_list = []
        rollout_metadata_list = []
        timing_info = {}

        for i, sample in enumerate(queue_samples):
            sample_data_list.append(sample.data)
            rollout_metadata_list.append(sample.rollout_metadata)

        batch = DataProto.from_items(sample_data_list)

        # Collect timing information and metadata
        param_versions = []
        sample_timestamps = []
        for metadata in rollout_metadata_list:
            # Extract parameter version and timestamp
            param_versions.append(metadata.get("rollout_param_version", 0))
            sample_timestamps.append(metadata.get("generation_timestamp", time.time()))
            if "timing" in metadata:
                for timing_key, timing_value in metadata["timing"].items():
                    if timing_key not in timing_info:
                        timing_info[timing_key] = []
                    # if isinstance(timing_value, (int, float)):
                    #     timing_info[timing_key].append(timing_value)
        # Calculate average timing
        avg_timing = {}
        for key, values in timing_info.items():
            if values and len(values) > 0:
                avg_timing[key] = sum(values) / len(values)

        # Create meta_info
        meta_info = {
            "timing": avg_timing,
            "queue_sample_count": len(queue_samples),
            "rollout_param_versions": param_versions,
            "sample_timestamps": sample_timestamps,
            "param_version_diversity": len(set(param_versions)),
            "avg_sample_age": np.mean([time.time() - ts for ts in sample_timestamps]),
        }

        print(meta_info)

        return batch

    def _create_actor_rollout_classes(self):
        # create actor
        for role in [Role.Actor]:
            resource_pool = self.resource_pool_manager.get_resource_pool(role)
            role_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[role],
                config=self.config.actor_rollout_ref,
                role=str(role),
            )
            self.resource_pool_to_cls[resource_pool][str(role)] = role_cls

    def _init_models(self):
        if self.use_critic:
            self.critic_wg = self.all_wg[str(Role.Critic)]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = self.all_wg[str(Role.RefPolicy)]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = self.all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        self.actor_wg = self.all_wg[str(Role.Actor)]
        self.actor_wg.init_model()
        self.actor_rollout_wg = self.actor_wg  # to be compatible with the functions that not be modified

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        print("Starting FullyAsyncTrainer...")
        if self.message_queue_client is None:
            raise ValueError("MessageQueue client not set. Call set_message_queue_client() first.")
        if self.param_synchronizer is None:
            raise ValueError("param_synchronizer client not set. Call set_parameter_synchronizer() first.")

        from verl.utils.tracking import Tracking

        self.logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # we start from step 1
        self.global_steps += 1
        self.max_steps_duration = 0

        # Use queue mode, no need for traditional dataloader iterator
        # Initialize to get the first batch of data
        while True:
            print("while True", flush=True)

            # Check queue status
            if self.message_queue_client:
                queue_stats = self.message_queue_client.get_statistics()
                print(f"Queue status before getting samples: {queue_stats}")

                if queue_stats.get("queue_size", 0) == 0:
                    print("WARNING: Queue is empty, will block waiting for samples")

            metrics = {}
            timing_raw = {}

            is_last_step = False

            with marked_timer("step", timing_raw):
                with marked_timer("gen", timing_raw, color="red"):
                    epoch, batch = self._get_samples_from_queue()
                    if batch is None:
                        break

                print("_get_samples_from_queue end")

                # # 更新统计信息
                #     self.processed_samples += len(batch) if isinstance(batch, list) else 1
                #
                #     # 从meta_info中获取参数版本信息
                #     if hasattr(batch, "meta_info") and batch.meta_info:
                #         rollout_param_versions = batch.meta_info.get("rollout_param_versions", [])
                #         if rollout_param_versions:
                #             # 统计陈旧样本
                #             stale_count = sum(1 for v in rollout_param_versions if self.current_param_version - v > 1)
                #             self.stale_samples_processed += stale_count
                #
                #         # 添加新鲜度指标到metrics
                #         if rollout_param_versions:
                #             param_version_diversity = batch.meta_info.get("param_version_diversity", 0)
                #             avg_sample_age = batch.meta_info.get("avg_sample_age", 0)
                #
                #             metrics.update(
                #                 {
                #                     "freshness/param_version_diversity": param_version_diversity,
                #                     "freshness/avg_sample_age": avg_sample_age,
                #                     "freshness/stale_samples_ratio": stale_count / len(rollout_param_versions)
                #                     if rollout_param_versions
                #                     else 0,
                #                     "statistics/processed_samples": self.processed_samples,
                #                     "statistics/stale_samples_processed": self.stale_samples_processed,
                #                     "statistics/current_param_version": self.current_param_version,
                #                 }
                #             )
                print("_process_batch_common")
                batch, reward_extra_infos_dict = self._process_batch_common(batch, metrics, timing_raw)
                print("_log_rollout")
                self._log_rollout(batch, reward_extra_infos_dict, timing_raw)
                print("_check_save_checkpoint")
                self._check_save_checkpoint(is_last_step, timing_raw)

            print("_collect_metrics")
            # self._collect_metrics(batch, epoch, metrics, timing_raw)

            # Trigger parameter synchronization after training step
            print("_trigger_parameter_sync_after_step")
            self._trigger_parameter_sync_after_step()
            print(f"global_steps: {self.global_steps}")
            self.global_steps += 1

    def get_statistics(self) -> dict:
        """Get training statistics"""
        queue_stats = self.message_queue_client.get_statistics() if self.message_queue_client else {}
        return {
            "global_steps": self.global_steps,
            "processed_samples": self.processed_samples,
            "stale_samples_processed": self.stale_samples_processed,
            "current_param_version": self.current_param_version,
            "queue_size": queue_stats.get("queue_size", 0),
            "queue_total_produced": queue_stats.get("total_produced", 0),
            "queue_total_consumed": queue_stats.get("total_consumed", 0),
            "queue_dropped_samples": queue_stats.get("dropped_samples", 0),
        }

    def _trigger_parameter_sync_after_step(self):
        """
        Trigger parameter synchronization after training step
        This ensures rollouter always uses the latest trained parameters
        """
        self.current_param_version = self.current_param_version + 1
        print(
            f"[TRAINER] Triggering parameter sync after "
            f"training step {self.global_steps}, version: {self.current_param_version}"
        )
        logger.info(
            f"Triggering parameter sync after training step {self.global_steps}, version: {self.current_param_version}"
        )
        ray.get(self.param_synchronizer.sync_weights.remote(self.current_param_version))

    def _compute_sample_freshness_metrics(self, batch_samples: list[QueueSample]) -> dict:
        """
        Compute sample freshness metrics

        Args:
            batch_samples: List of queue samples

        Returns:
            dict: Dictionary of freshness metrics
        """
        if not batch_samples:
            return {}

        try:
            # Extract parameter versions and timestamps
            sample_ages = []
            sample_latencies = []
            current_time = time.time()

            for sample in batch_samples:
                # Get information from rollout_metadata
                if hasattr(sample, "rollout_metadata") and sample.rollout_metadata:
                    rollout_version = sample.rollout_metadata.get("rollout_param_version", 0)
                    generation_time = sample.rollout_metadata.get("generation_timestamp", current_time)
                else:
                    rollout_version = 0
                    generation_time = current_time

                age = max(0, self.current_param_version - rollout_version)
                latency = max(0, current_time - generation_time)

                sample_ages.append(age)
                sample_latencies.append(latency)

            if not sample_ages:
                return {}

            return {
                "freshness/avg_sample_age": np.mean(sample_ages),
                "freshness/max_sample_age": max(sample_ages),
                "freshness/min_sample_age": min(sample_ages),
                "freshness/avg_sample_latency": np.mean(sample_latencies),
                "freshness/max_sample_latency": max(sample_latencies),
                "freshness/min_sample_latency": min(sample_latencies),
                "freshness/stale_samples_ratio": sum(1 for age in sample_ages if age > 1) / len(sample_ages),
                "freshness/sample_count": len(sample_ages),
            }

        except Exception as e:
            logger.error(f"Error computing freshness metrics: {e}")
            return {"freshness/error": str(e)}
