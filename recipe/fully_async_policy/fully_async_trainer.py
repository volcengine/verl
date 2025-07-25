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
from pprint import pprint

import numpy as np
import ray
from omegaconf import OmegaConf
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from recipe.fully_async_policy.message_queue import BatchSample, MessageQueueClient
from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    ResourcePoolManager,
    Role,
    WorkerType,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.tracking import ValidationGenerationsLogger

logger = logging.getLogger(__name__)


class FullyAsyncTrainer:
    """
    完全异步的PPO训练器，从MessageQueue获取样本进行训练
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
        device_name="cuda",
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()

        # 数据相关
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.collate_fn = collate_fn
        self.train_sampler = train_sampler

        # 角色配置
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.use_critic = Role.Critic in role_worker_mapping
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # Worker groups
        self.actor_wg = None
        self.critic_wg = None
        self.ref_policy_wg = None
        self.rm_wg = None

        # 训练状态
        self.global_steps = 0
        self.current_param_version = 0
        self.total_training_steps = config.trainer.total_training_steps

        # MessageQueue客户端
        self.message_queue_client = None

        # 与Rollouter的通信
        self.rollouter_actor = None

        # 统计信息
        self.processed_samples = 0
        self.stale_samples_processed = 0

    def set_message_queue_client(self, message_queue_client: MessageQueueClient):
        """设置消息队列客户端"""
        self.message_queue_client = message_queue_client

    def set_rollouter_actor(self, rollouter_actor):
        """设置Rollouter Actor的引用"""
        self.rollouter_actor = rollouter_actor

    def init_workers(self):
        """初始化训练workers"""
        logger.info("Initializing FullyAsyncTrainer workers...")

        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # 创建actor worker
        actor_resource_pool = self.resource_pool_manager.get_resource_pool(Role.Actor)
        actor_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.Actor],
            config=self.config.actor_rollout_ref,
            role="actor",
        )
        self.resource_pool_to_cls[actor_resource_pool]["actor"] = actor_cls

        # 创建critic worker
        if self.use_critic:
            critic_resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[critic_resource_pool]["critic"] = critic_cls

        # 创建reference policy worker
        if self.use_reference_policy:
            ref_resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
            )
            self.resource_pool_to_cls[ref_resource_pool]["ref"] = ref_policy_cls

        # 创建reward model worker
        if self.use_rm:
            rm_resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model
            )
            self.resource_pool_to_cls[rm_resource_pool]["rm"] = rm_cls

        # 初始化WorkerGroup
        all_wg = {}
        wg_kwargs = {}
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

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

    def _load_checkpoint(self):
        """加载检查点"""
        # 简化的检查点加载逻辑
        pass

    def _validate(self):
        """执行验证"""
        if self.val_reward_fn is None:
            return None

        # 简化的验证逻辑
        logger.info("Validation step skipped in async trainer")
        return {"val_reward": 0.0}

    def _save_checkpoint(self):
        """保存检查点"""
        # 简化的检查点保存逻辑
        pass

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path):
        """保存生成结果"""
        # 简化的生成结果保存逻辑
        pass

    def _update_param_version_and_sync(self):
        """更新参数版本并同步到Rollouter"""
        self.current_param_version += 1

        # 通知MessageQueue更新参数版本
        self.message_queue_client.update_param_version(self.current_param_version)

        # 通知Rollouter更新参数
        if self.rollouter_actor is not None:
            ray.get(self.rollouter_actor.update_rollout_weights.remote(self.current_param_version))

    def _process_batch_samples(self, batch_samples: list[BatchSample]) -> DataProto:
        """处理从队列获取的batch样本"""
        if len(batch_samples) == 1:
            return batch_samples[0].data

        # 如果有多个batch，需要合并
        all_batches = [sample.data for sample in batch_samples]
        return DataProto.concat(all_batches)

    def fit(self):
        """主训练循环"""
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # 加载检查点
        self._load_checkpoint()

        # 验证
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            if val_metrics:
                pprint(f"Initial validation metrics: {val_metrics}")
                logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # 进度条
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        self.global_steps += 1
        last_val_metrics = None

        if self.message_queue_client is None:
            raise ValueError("MessageQueue client not set. Call set_message_queue_client() first.")

        logger.info("Starting fully async training loop...")

        while self.global_steps <= self.total_training_steps:
            do_profile = (
                self.global_steps in self.config.trainer.profile_steps
                if self.config.trainer.profile_steps is not None
                else False
            )

            if do_profile:
                self.actor_wg.start_profile()
                if self.use_reference_policy:
                    self.ref_policy_wg.start_profile()
                if self.use_critic:
                    self.critic_wg.start_profile()
                if self.use_rm:
                    self.rm_wg.start_profile()

            metrics = {}
            timing_raw = {}
            # is_last_step = self.global_steps >= self.total_training_steps

            with marked_timer("step", timing_raw):
                # 从队列获取样本
                with marked_timer("get_batch_from_queue", timing_raw, color="blue"):
                    min_batch_count = self.config.async_training.get("min_batch_count", 1)
                    batch_timeout = self.config.async_training.get("batch_timeout", 30.0)

                    batch_samples = self.message_queue_client.get_batch(
                        min_batch_count=min_batch_count, timeout=batch_timeout
                    )

                    if batch_samples is None:
                        logger.warning("Timeout waiting for batch samples, continuing...")
                        continue

                # 处理获取的样本
                batch = self._process_batch_samples(batch_samples)

                # 计算样本的新鲜度
                sample_ages = [self.current_param_version - sample.param_version for sample in batch_samples]
                avg_sample_age = np.mean(sample_ages)
                max_sample_age = max(sample_ages)

                logger.info(
                    f"Processing batch with {len(batch_samples)} samples, "
                    f"avg_age={avg_sample_age:.1f}, max_age={max_sample_age}"
                )

                # 添加响应掩码
                batch.batch["response_mask"] = compute_response_mask(batch)

                # 计算奖励
                with marked_timer("compute_reward", timing_raw, color="yellow"):
                    if self.reward_fn is not None:
                        batch, reward_extra_infos_dict = compute_reward(
                            batch, reward_fn=self.reward_fn, tokenizer=self.tokenizer
                        )
                    elif self.use_rm:
                        batch, reward_extra_infos_dict = compute_reward_async(
                            batch, rm_wg=self.rm_wg, tokenizer=self.tokenizer
                        )
                    else:
                        raise ValueError("No reward function or reward model provided")

                # 计算reference log probabilities
                if self.use_reference_policy:
                    with marked_timer("compute_ref_log_prob", timing_raw, color="green"):
                        if self.ref_in_actor:
                            ref_log_prob_output = self.actor_wg.compute_ref_log_prob(batch)
                        else:
                            ref_log_prob_output = self.ref_policy_wg.compute_log_prob(batch)
                        batch = batch.union(ref_log_prob_output)

                # 计算actor log probabilities
                with marked_timer("compute_log_prob", timing_raw, color="cyan"):
                    log_prob_output = self.actor_wg.compute_log_prob(batch)
                    batch = batch.union(log_prob_output)

                # 应用KL惩罚
                if self.use_reference_policy:
                    batch = apply_kl_penalty(batch, self.config.algorithm)

                # 计算优势
                if self.use_critic:
                    with marked_timer("compute_values", timing_raw, color="magenta"):
                        values_output = self.critic_wg.compute_values(batch)
                        batch = batch.union(values_output)

                with marked_timer("compute_advantage", timing_raw, color="orange"):
                    batch = compute_advantage(batch, self.config.algorithm)

                # 更新critic
                if self.use_critic:
                    with marked_timer("update_critic", timing_raw, color="pink"):
                        critic_output = self.critic_wg.update_critic(batch)
                    critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                    metrics.update(critic_output_metrics)

                # 更新actor
                if self.config.trainer.critic_warmup <= self.global_steps:
                    with marked_timer("update_actor", timing_raw, color="red"):
                        batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                        actor_output = self.actor_wg.update_actor(batch)
                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_output_metrics)

                    # 更新参数版本并同步到Rollouter
                    with marked_timer("sync_params", timing_raw, color="purple"):
                        self._update_param_version_and_sync()

                # 记录rollout生成
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

                # 验证
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.val_freq is not None
                    and self.global_steps % self.config.trainer.val_freq == 0
                ):
                    with marked_timer("validation", timing_raw, color="brown"):
                        val_metrics = self._validate()
                        if val_metrics:
                            pprint(f"Validation metrics at step {self.global_steps}: {val_metrics}")
                            last_val_metrics = val_metrics

            # 计算性能指标
            timing_metrics = compute_timing_metrics(timing_raw)
            throughput_metrics = compute_throughout_metrics(timing_raw, len(batch))
            data_metrics = compute_data_metrics(batch, self.tokenizer)

            # 添加样本新鲜度指标
            freshness_metrics = {
                "avg_sample_age": avg_sample_age,
                "max_sample_age": max_sample_age,
                "processed_samples": self.processed_samples,
                "param_version": self.current_param_version,
            }

            metrics.update(timing_metrics)
            metrics.update(throughput_metrics)
            metrics.update(data_metrics)
            metrics.update(freshness_metrics)

            if last_val_metrics is not None:
                metrics.update(last_val_metrics)
                last_val_metrics = None

            # 记录日志
            logger.log(data=metrics, step=self.global_steps)

            # 更新进度条
            progress_bar.update(1)
            progress_bar.set_postfix(
                {
                    "reward": f"{metrics.get('reward/mean', 0):.3f}",
                    "kl": f"{metrics.get('actor/approx_kl', 0):.3f}",
                    "queue_size": self.message_queue_client.get_queue_size(),
                    "param_version": self.current_param_version,
                }
            )

            # 保存检查点
            if self.config.trainer.save_freq is not None and self.global_steps % self.config.trainer.save_freq == 0:
                self._save_checkpoint()

            if do_profile:
                self.actor_wg.end_profile()
                if self.use_reference_policy:
                    self.ref_policy_wg.end_profile()
                if self.use_critic:
                    self.critic_wg.end_profile()
                if self.use_rm:
                    self.rm_wg.end_profile()

            self.global_steps += 1
            self.processed_samples += len(batch_samples)

        progress_bar.close()
        logger.info(f"Training completed after {self.global_steps} steps")

        # 最终验证
        if self.val_reward_fn is not None:
            val_metrics = self._validate()
            if val_metrics:
                pprint(f"Final validation metrics: {val_metrics}")
                logger.log(data=val_metrics, step=self.global_steps)

        # 最终检查点保存
        self._save_checkpoint()

    def get_statistics(self) -> dict:
        """获取训练统计信息"""
        return {
            "global_steps": self.global_steps,
            "processed_samples": self.processed_samples,
            "stale_samples_processed": self.stale_samples_processed,
            "current_param_version": self.current_param_version,
            "queue_size": self.message_queue_client.get_queue_size() if self.message_queue_client else 0,
        }
