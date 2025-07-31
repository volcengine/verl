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
from pprint import pprint

import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from recipe.fully_async_policy.message_queue import QueueSample, MessageQueueClient
from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
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
        device_name="cuda",
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert not self.hybrid_engine

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

        # 角色配置 - 参考OneStepOffRayTrainer的配置
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # KL控制器
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        # 确定是否使用critic - 参考OneStepOffRayTrainer的逻辑
        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            # AdvantageEstimator.REMAX, # TODO:REMAX advantage estimator is not yet supported in one_step_off_policy
            AdvantageEstimator.RLOO,
            AdvantageEstimator.OPO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
            AdvantageEstimator.GPG,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError(f"Unsupported advantage estimator: {self.config.algorithm.adv_estimator}")

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
        self.param_sync_count = 0

        self._validate_config()

    def _validate_config(self):
        """验证配置"""
        required_configs = ["trainer.total_training_steps", "algorithm.adv_estimator", "data.train_batch_size"]

        for config_path in required_configs:
            if not OmegaConf.select(self.config, config_path):
                raise ValueError(f"Missing required config: {config_path}")

    def set_message_queue_client(self, message_queue_client: MessageQueueClient):
        """设置消息队列客户端"""
        self.message_queue_client = message_queue_client

    def set_rollouter_actor(self, rollouter_actor):
        """设置Rollouter Actor的引用"""
        self.rollouter_actor = rollouter_actor

    def init_workers(self):
        """初始化训练workers - 参考OneStepOffRayTrainer的实现"""
        logger.info("Initializing FullyAsyncTrainer workers...")

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

    def _load_checkpoint(self):
        """加载检查点"""
        # TODO: 实现检查点加载逻辑
        logger.info("Checkpoint loading not implemented yet")

    def _validate(self):
        """执行验证 - 参考OneStepOffRayTrainer的验证逻辑"""
        if self.val_reward_fn is None:
            return None

        # TODO: 实现完整的验证逻辑
        logger.info("Running validation...")
        val_metrics = {"val_reward": 0.0}  # 简化的验证指标
        return val_metrics

    def _save_checkpoint(self):
        """保存检查点"""
        # TODO: 实现检查点保存逻辑
        logger.info("Checkpoint saving not implemented yet")

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path):
        """保存生成结果"""
        # TODO: 实现生成结果保存逻辑
        logger.debug(f"Dumping generations to {dump_path}")

    def _balance_batch(self, batch: DataProto, metrics: dict):
        """平衡batch中的有效token数量 - 参考OneStepOffRayTrainer的实现"""
        # TODO: 实现batch平衡逻辑
        pass

    def _sync_parameters_to_rollouter(self):
        """同步参数到Rollouter - 改进的同步机制"""
        if self.rollouter_actor is None:
            logger.warning("Rollouter actor not set, skipping parameter sync")
            return

        self.current_param_version += 1

        try:
            # 通知MessageQueue更新参数版本
            self.message_queue_client.update_param_version(self.current_param_version)

            # 同步参数到Rollouter
            sync_future = self.rollouter_actor.update_rollout_weights.remote(self.current_param_version)
            ray.get(sync_future)

            self.param_sync_count += 1
            logger.info(f"Parameter sync completed, version: {self.current_param_version}")

        except Exception as e:
            logger.error(f"Failed to sync parameters: {e}")
            self.current_param_version -= 1  # 回滚版本号
            raise

    def _process_batch_samples(self, batch_samples: list[QueueSample]) -> DataProto:
        """处理从队列获取的batch样本 - 改进的批处理逻辑"""
        if not batch_samples:
            raise ValueError("Empty batch samples")

        if len(batch_samples) == 1:
            return batch_samples[0].data

        # 合并多个batch - 使用DataProto的concat方法
        try:
            all_batches = [sample.data for sample in batch_samples]
            merged_batch = DataProto.concat(all_batches)
            logger.debug(f"Successfully merged {len(batch_samples)} batches")
            return merged_batch
        except Exception as e:
            logger.error(f"Failed to merge batch samples: {e}")
            raise

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

    def fit(self):
        """主训练循环 - 基于OneStepOffRayTrainer的成熟实现"""
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger_tracker = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # 加载检查点
        self._load_checkpoint()

        # 初始验证
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            if val_metrics:
                pprint(f"Initial validation metrics: {val_metrics}")
                logger_tracker.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # 进度条
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Async Training")

        self.global_steps += 1
        last_val_metrics = None

        if self.message_queue_client is None:
            raise ValueError("MessageQueue client not set. Call set_message_queue_client() first.")

        logger.info("Starting fully async training loop...")

        while self.global_steps <= self.total_training_steps:
            # 性能分析
            do_profile = (
                self.global_steps in self.config.trainer.profile_steps
                if self.config.trainer.profile_steps is not None
                else False
            )

            if do_profile:
                self.actor_wg.start_profile()
                if self.use_reference_policy and not self.ref_in_actor:
                    self.ref_policy_wg.start_profile()
                if self.use_critic:
                    self.critic_wg.start_profile()
                if self.use_rm:
                    self.rm_wg.start_profile()

            metrics = {}
            timing_raw = {}
            is_last_step = self.global_steps >= self.total_training_steps

            with marked_timer("step", timing_raw):
                # 从队列获取样本
                with marked_timer("get_batch_from_queue", timing_raw, color="blue"):
                    min_batch_count = self.config.async_training.get("min_batch_count", 1)
                    batch_timeout = self.config.async_training.get("batch_timeout", 30.0)

                    batch_samples = self.message_queue_client.get_samples(
                        min_batch=min_batch_count, timeout=batch_timeout
                    )

                    if batch_samples is None:
                        logger.warning("Timeout waiting for batch samples, retrying...")
                        time.sleep(1.0)
                        continue

                # 处理获取的样本
                with marked_timer("process_batch_samples", timing_raw, color="cyan"):
                    batch = self._process_batch_samples(batch_samples)

                    # 计算样本新鲜度指标
                    freshness_metrics = self._compute_sample_freshness_metrics(batch_samples)
                    metrics.update(freshness_metrics)

                    logger.info(
                        f"Processing batch: {len(batch_samples)} samples, "
                        f"avg_age={freshness_metrics['freshness/avg_sample_age']:.1f}, "
                        f"max_age={freshness_metrics['freshness/max_sample_age']}"
                    )

                # 添加响应掩码 - 参考OneStepOffRayTrainer
                batch.batch["response_mask"] = compute_response_mask(batch)

                # 平衡batch
                if self.config.trainer.balance_batch:
                    self._balance_batch(batch, metrics=metrics)

                # 计算全局有效token数量
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                # 计算奖励 - 参考OneStepOffRayTrainer的实现
                with marked_timer("reward", timing_raw, color="yellow"):
                    if self.use_rm:
                        reward_tensor = self.rm_wg.compute_rm_score(batch)
                        batch = batch.union(reward_tensor)

                    if self.config.reward_model.get("launch_reward_fn_async", False):
                        future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                    else:
                        reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                # 计算旧的log probabilities - 参考OneStepOffRayTrainer
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

                # 计算reference log probabilities
                if self.use_reference_policy:
                    with marked_timer("ref", timing_raw, color="olive"):
                        if not self.ref_in_actor:
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                        else:
                            ref_log_prob = self.actor_wg.compute_ref_log_prob(batch)
                        batch = batch.union(ref_log_prob)

                # 计算values
                if self.use_critic:
                    with marked_timer("values", timing_raw, color="cyan"):
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)

                # 处理奖励和优势计算
                with marked_timer("adv", timing_raw, color="brown"):
                    if self.config.reward_model.get("launch_reward_fn_async", False):
                        reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                    batch.batch["token_level_scores"] = reward_tensor

                    if reward_extra_infos_dict:
                        batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                    # 应用KL惩罚
                    if self.config.algorithm.use_kl_in_reward:
                        batch, kl_metrics = apply_kl_penalty(
                            batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                        )
                        metrics.update(kl_metrics)
                    else:
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                    # 计算优势
                    norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)

                    batch = compute_advantage(
                        batch,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                        num_repeat=self.config.actor_rollout_ref.rollout.n,
                        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        config=self.config.algorithm,
                    )

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

                    # 同步参数到Rollouter
                    with marked_timer("sync_params", timing_raw, color="purple"):
                        self._sync_parameters_to_rollouter()

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
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                            print(last_val_metrics)
                    if val_metrics:
                        metrics.update(val_metrics)

                # 保存检查点
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

            # 收集指标 - 参考OneStepOffRayTrainer的指标收集
            metrics.update(
                {
                    "training/global_step": self.global_steps,
                    "training/param_version": self.current_param_version,
                    "training/param_sync_count": self.param_sync_count,
                }
            )

            # 数据和性能指标
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

            n_gpus = self.resource_pool_manager.get_n_gpus()
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

            # 队列状态指标
            queue_size = self.message_queue_client.get_queue_size()
            queue_stats = self.message_queue_client.get_statistics()
            metrics.update(
                {
                    "queue/size": queue_size,
                    "queue/total_produced": queue_stats["total_produced"],
                    "queue/total_consumed": queue_stats["total_consumed"],
                    "queue/dropped_samples": queue_stats["dropped_samples"],
                }
            )

            # 记录日志
            logger_tracker.log(data=metrics, step=self.global_steps)

            # 更新进度条
            progress_bar.update(1)
            progress_bar.set_postfix(
                {
                    "reward": f"{metrics.get('reward/mean', 0):.3f}",
                    "kl": f"{metrics.get('actor/approx_kl', 0):.3f}",
                    "queue_size": queue_size,
                    "param_ver": self.current_param_version,
                    "avg_age": f"{metrics.get('freshness/avg_sample_age', 0):.1f}",
                }
            )

            if do_profile:
                self.actor_wg.stop_profile()
                if self.use_reference_policy and not self.ref_in_actor:
                    self.ref_policy_wg.stop_profile()
                if self.use_critic:
                    self.critic_wg.stop_profile()
                if self.use_rm:
                    self.rm_wg.stop_profile()

            self.global_steps += 1
            self.processed_samples += len(batch_samples)

            if is_last_step:
                break

        progress_bar.close()
        logger.info(f"Training completed after {self.global_steps} steps")

        # 最终验证
        if self.val_reward_fn is not None:
            val_metrics = self._validate()
            if val_metrics:
                pprint(f"Final validation metrics: {val_metrics}")
                logger_tracker.log(data=val_metrics, step=self.global_steps)

        # 最终检查点保存
        self._save_checkpoint()

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
