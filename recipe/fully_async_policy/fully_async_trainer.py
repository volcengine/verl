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

from recipe.fully_async_policy.message_queue import MessageQueueClient, RolloutSample
from recipe.fully_async_policy.utils import calculate_one_step_size
from verl.experimental.agent_loop.agent_loop import postprocess_agent_loop_outputs
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


@ray.remote(num_cpus=10)
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

        self.required_samples = calculate_one_step_size(
            self.minimal_bsz, config.actor_rollout_ref.actor.ppo_mini_batch_size
        )

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
        Uses a loop to continuously collect samples until enough are gathered

        Returns:
            tuple: (epoch, batch_dict, gen_batch_output)
        """
        print(
            f"[FullyAsyncTrainer] Requesting {self.required_samples} samples from queue",
            flush=True,
        )

        # Collect samples using a simple loop calling get_sample
        consumer_start = time.time()
        queue_samples = []

        while len(queue_samples) < self.required_samples:
            # 获取单个样本，会一直等待直到有样本或收到None
            sample = self.message_queue_client.get_sample_sync()

            if sample is None:
                # 检测到结束信号（None），立即退出
                logger.info(
                    f"Detected termination signal (None), stopping sample collection. "
                    f"Collected {len(queue_samples)}/{self.required_samples} samples"
                )
                break

            queue_samples.append(sample)

            if len(queue_samples) % 10 == 0 or len(queue_samples) >= self.required_samples:
                print(f"[FullyAsyncTrainer] Collected {len(queue_samples)}/{self.required_samples} samples")

        consumer_end = time.time()

        if not queue_samples or len(queue_samples) < self.required_samples:
            logger.warning("not enough samples collected after loop")
            return None, None

        print(
            f"[FullyAsyncTrainer] Loop collection completed: {len(queue_samples)}/{self.required_samples} samples, "
            f"total wait time: {consumer_end - consumer_start:.2f} seconds"
        )

        queue_samples = [ray.cloudpickle.loads(x) for x in queue_samples]
        # Assemble batch - now working directly with RolloutSample objects
        batch = self._assemble_gen_batch_output_from_queue_samples(queue_samples)

        return 0, batch

    def _assemble_gen_batch_output_from_queue_samples(self, rollout_samples: list[RolloutSample]):
        """
        Assemble gen_batch_output from RolloutSample objects
        从 RolloutSample 对象中组装批次，类似 ray_trainer 的 _post_generate_batch 逻辑

        Args:
            rollout_samples: List of RolloutSample objects

        Returns:
            DataProto: Assembled gen_batch_output
        """
        start_time = time.time()

        import numpy as np
        import torch

        from verl import DataProto
        from verl.trainer.ppo.ray_trainer import compute_response_mask

        if not rollout_samples:
            raise ValueError("Empty rollout_samples provided for batch assembly")

        print(f"[FullyAsyncTrainer] Assembling batch from {len(rollout_samples)} RolloutSample objects")

        # 直接处理 RolloutSample 对象
        processing_times = [rs.processing_time for rs in rollout_samples]

        # 第一步：从 AgentLoopOutput 创建生成结果的 DataProto
        agent_loop_outputs = [rs.agent_loop_output for rs in rollout_samples]
        gen_batch_output = postprocess_agent_loop_outputs(agent_loop_outputs, self.tokenizer, self.config)

        # 第二步：重建原始 batch 信息
        # 每个 RolloutSample 都是独立的，直接按顺序重建原始数据
        original_batch_list = []
        for rs in rollout_samples:
            original_batch_dict = rs.original_batch_dict

            # 重建 DataProto
            original_batch_item = DataProto.from_single_dict(
                {
                    **{k: v for k, v in original_batch_dict["batch"].items()},
                    **{f"__{k}": v for k, v in original_batch_dict["non_tensor_batch"].items()},
                }
            )
            original_batch_item.meta_info.update(original_batch_dict["meta_info"])
            original_batch_list.append(original_batch_item)

        # 合并所有原始样本为一个批次
        if original_batch_list:
            original_batch = DataProto.from_items(original_batch_list)
        else:
            # 如果没有原始数据，创建空的 DataProto
            original_batch = DataProto.from_single_dict({})

        # 添加 UID
        uids = []
        for rs in rollout_samples:
            uids.append(f"uid_{rs.sample_id}")
        original_batch.non_tensor_batch["uid"] = np.array(uids, dtype=object)

        # 直接合并原始数据和生成结果，不需要 repeat
        # 因为队列中的每个 RolloutSample 都已经是独立的样本
        final_batch = original_batch.union(gen_batch_output)

        # 计算 response_mask（如果不存在）
        if "response_mask" not in final_batch.batch.keys():
            final_batch.batch["response_mask"] = compute_response_mask(final_batch)

        # 平衡批次（如果配置了）
        if self.config.trainer.balance_batch:
            self._balance_batch(final_batch, metrics={})

        # 计算全局有效 token 数
        if "attention_mask" in final_batch.batch:
            final_batch.meta_info["global_token_num"] = torch.sum(final_batch.batch["attention_mask"], dim=-1).tolist()

        # 收集统计信息和元数据（直接从 RolloutSample 中获取）
        param_versions = [rs.param_version for rs in rollout_samples]
        sample_timestamps = [rs.generation_timestamp for rs in rollout_samples]

        # 创建 meta_info
        final_batch.meta_info.update(
            {
                "rollout_param_versions": param_versions,
                "sample_timestamps": sample_timestamps,
                "avg_processing_time": np.mean(processing_times) if processing_times else 0,
                "max_processing_time": np.max(processing_times) if processing_times else 0,
                "param_version_diversity": len(set(param_versions)) if param_versions else 0,
                "avg_sample_age": np.mean([time.time() - ts for ts in sample_timestamps]) if sample_timestamps else 0,
                "assembly_time": time.time() - start_time,
            }
        )

        print(f"[FullyAsyncTrainer] Batch assembly completed in {time.time() - start_time:.2f}s")
        print(f"[FullyAsyncTrainer] {final_batch}")

        return final_batch

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

    def _init_async_rollout_manager(self):
        pass

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        print("[FullyAsyncTrainer] Starting FullyAsyncTrainer...")
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
            metrics = {}
            timing_raw = {}

            is_last_step = False

            with marked_timer("step", timing_raw):
                with marked_timer("gen", timing_raw, color="red"):
                    epoch, batch = self._get_samples_from_queue()
                    if batch is None:
                        break

                    # 更新统计信息
                    self.processed_samples += len(batch) if isinstance(batch, list) else 1

                    # 从meta_info中获取参数版本信息
                    if hasattr(batch, "meta_info") and batch.meta_info:
                        rollout_param_versions = batch.meta_info.get("rollout_param_versions", [])
                        if rollout_param_versions:
                            # 统计陈旧样本
                            stale_count = sum(1 for v in rollout_param_versions if self.current_param_version - v > 1)
                            self.stale_samples_processed += stale_count

                        # 添加新鲜度指标到metrics
                        if rollout_param_versions:
                            param_version_diversity = batch.meta_info.get("param_version_diversity", 0)
                            avg_sample_age = batch.meta_info.get("avg_sample_age", 0)

                            metrics.update(
                                {
                                    "freshness/param_version_diversity": param_version_diversity,
                                    "freshness/avg_sample_age": avg_sample_age,
                                    "freshness/stale_samples_ratio": stale_count / len(rollout_param_versions)
                                    if rollout_param_versions
                                    else 0,
                                    "statistics/processed_samples": self.processed_samples,
                                    "statistics/stale_samples_processed": self.stale_samples_processed,
                                    "statistics/current_param_version": self.current_param_version,
                                }
                            )
                # batch, reward_extra_infos_dict = self._process_batch_common(batch, metrics, timing_raw)
                # self._log_rollout(batch, reward_extra_infos_dict, timing_raw)
                # self._check_save_checkpoint(is_last_step, timing_raw)

            # self._collect_metrics(batch, epoch, metrics, timing_raw)

            # Trigger parameter synchronization after training step
            # self._trigger_parameter_sync_after_step()
            print(f"[FullyAsyncTrainer] global_steps: {self.global_steps}")
            self.global_steps += 1

    def get_statistics(self) -> dict:
        """Get training statistics"""
        queue_stats = self.message_queue_client.get_statistics_sync() if self.message_queue_client else {}
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
            f"[FullyAsyncTrainer] Triggering parameter sync after "
            f"training step {self.global_steps}, version: {self.current_param_version}"
        )
        ray.get(self.param_synchronizer.sync_weights.remote(self.current_param_version))

    def _compute_sample_freshness_metrics(self, rollout_samples: list[RolloutSample]) -> dict:
        """
        Compute sample freshness metrics

        Args:
            rollout_samples: List of RolloutSample objects

        Returns:
            dict: Dictionary of freshness metrics
        """
        if not rollout_samples:
            return {}

        try:
            # Extract parameter versions and timestamps directly from RolloutSample
            sample_ages = []
            sample_latencies = []
            current_time = time.time()

            for sample in rollout_samples:
                # Get information directly from RolloutSample
                rollout_version = sample.param_version
                generation_time = sample.generation_timestamp

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
