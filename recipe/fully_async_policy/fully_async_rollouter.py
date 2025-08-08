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
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pprint import pprint

import ray
from omegaconf import OmegaConf

from recipe.fully_async_policy.message_queue import MessageQueueClient, QueueSample
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role, WorkerType
from verl.utils.debug import marked_timer
from verl.utils.tracking import ValidationGenerationsLogger


@ray.remote(num_cpus=10, max_concurrency=10)
class FullyAsyncRollouter(RayPPOTrainer):
    """
    异步样本生成器，负责持续生成训练样本并放入MessageQueue
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
        device_name=None,
        max_queue_size=1000,
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
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        self.ref_in_actor = False
        self.kl_ctrl_in_reward = False
        self.use_critic = False
        self.use_reference_policy = False
        self.use_rm = False

        # 创建数据集
        print("Creating datasets...")
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        self._validate_config()
        pprint(f"Rollouter _create_dataloader...\n{train_dataset}\n{val_dataset}")
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        # rollouter 参数配置
        self.message_queue_client = None

        self.current_param_version = 0

        # 新鲜度控制 - 改进的配置管理
        async_config = config.async_training
        self.staleness_threshold = async_config.get("staleness_threshold", 3)

        # 统计信息
        self.total_generated_samples = 0
        self.dropped_stale_samples = 0
        self.param_sync_requests = 0

        # Worker groups
        self.rollout_wg = None
        self.message_queue_client = None

        # 并发控制
        self.running = False
        self.paused = False
        self.generation_thread = None
        self.monitor_thread = None
        self.thread_executor = ThreadPoolExecutor(max_workers=2)
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)

        # 暂停/恢复统计信息
        self.pause_count = 0
        self.resume_count = 0
        self.total_pause_time = 0.0
        self.last_pause_time = None

        # 参数同步相关
        self.param_synchronizer = None
        self.last_sync_time = 0
        self.sync_in_progress = False
        self.sync_lock = threading.Lock()

        # 参数同步状态 - 基于one_step_off_policy模式
        self._weights_info = None
        self._is_rollout = True  # rollouter是rollout角色
        self._is_actor = False

        self.max_queue_size = max_queue_size

    def set_message_queue_client(self, message_queue_client: MessageQueueClient):
        """设置消息队列客户端"""
        with self.lock:
            self.message_queue_client = message_queue_client

    def set_parameter_synchronizer(self, param_synchronizer):
        """设置参数同步器"""
        with self.lock:
            self.param_synchronizer = param_synchronizer

    def _validate_config(self):
        # 验证异步训练配置
        if not hasattr(self.config, "async_training"):
            raise ValueError("Missing async_training configuration")

    def _create_actor_rollout_classes(self):
        # only create rollout
        for role in [Role.Rollout]:
            resource_pool = self.resource_pool_manager.get_resource_pool(role)
            role_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[role],
                config=self.config.actor_rollout_ref,
                role=str(role),
            )
            self.resource_pool_to_cls[resource_pool][str(role)] = role_cls

    def _init_models(self):
        self.rollout_wg = self.all_wg[str(Role.Rollout)]
        self.rollout_wg.init_model()
        self.actor_rollout_wg = self.rollout_wg

    def _create_continuous_iterator(self):
        """
        Create a continuous data iterator across epoch
        """
        for epoch in range(self.config.trainer.total_epochs):
            iterator = iter(self.train_dataloader)
            for batch_dict in iterator:
                yield epoch, batch_dict

    def fit(self):
        """开始异步生成样本 - 改进的主运行逻辑"""
        print("Starting Rollouter...")
        if self.message_queue_client is None:
            raise ValueError("MessageQueue client not set. Call set_message_queue_client() first.")
        # if self.param_synchronizer is None:
        #     raise ValueError("param_synchronizer client not set. Call set_parameter_synchronizer() first.")

        # 设置运行状态
        with self.lock:
            self.running = True
            self.paused = False

        # 创建并启动生成线程
        self.generation_thread = threading.Thread(target=self._generation_loop, daemon=True)
        self.generation_thread.start()

        # 创建并启动监控线程
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

        # 等待线程完成
        self.generation_thread.join()
        self.monitor_thread.join()

        print("Rollouter fit completed")

    def _generation_loop(self):
        """
        主要的生成循环

        循环入口，需要
        1. running 判断
        4. 中断判断
        3. 新鲜度判断

        生成样本过程中，需要
        1. running 判断
        2. 中断判断
        """

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

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            self.logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        continuous_iterator = self._create_continuous_iterator()
        for epoch, batch_dict in continuous_iterator:
            with self.lock:
                if not self.running:
                    break

                if self._should_pause_generation():
                    self.pause()

                # 如果被暂停，等待恢复
                while self.paused and self.running:
                    print("Generation thread paused, waiting...")
                    self.condition.wait()

                # 再次检查运行状态
                if not self.running:
                    break

            metrics = {}
            timing_raw = {}
            batch, gen_batch = self._prepare_generate_batch(batch_dict)
            is_last_step = self.global_steps >= self.total_training_steps

            # generate a batch
            with marked_timer("gen", timing_raw, color="red"):
                if not self.async_rollout_mode:
                    gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                else:
                    gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                timing_raw.update(gen_batch_output.meta_info["timing"])
                gen_batch_output.meta_info.pop("timing", None)

            if gen_batch_output is not None:
                # 准备rollout metadata
                rollout_metadata = {
                    "timing": timing_raw,
                    "generation_timestamp": time.time(),
                    "rollout_param_version": self.current_param_version,
                }
                batch = self._post_generate_batch(batch, gen_batch_output, metrics)

                for sample in batch:
                    # for sample in samples:
                    queue_sample = QueueSample(
                        data=sample,
                        rollout_metadata=rollout_metadata,
                    )
                    # 放入队列
                    success = self.message_queue_client.put_sample(
                        sample=ray.cloudpickle.dumps(queue_sample),
                        param_version=self.current_param_version,
                    )
                    print(f"put samples {success}")
                    with self.lock:
                        if success:
                            self.total_generated_samples += 1
                        else:
                            self.dropped_stale_samples += 1

                    if self.global_steps % 1 == 0:
                        print(
                            f"Generated {self.total_generated_samples} batches, \n"
                            f"param_version={self.current_param_version}, \n"
                            f"Dropped stale samples: {self.dropped_stale_samples}\n"
                        )

            self.global_steps += 1

            if is_last_step:
                pprint(f"Final validation metrics: {last_val_metrics}")
                break

        with self.lock:
            self.running = False

    def _monitor_loop(self):
        """监控线程 - 监控状态并处理控制信号"""
        # 主线程保持运行，处理控制信号和状态监控
        last_stats_time = time.time()
        stats_interval = 30.0  # 30秒报告一次统计
        check_interval = 5.0  # 5秒检查一次状态
        while True:
            with self.lock:
                if not self.running:
                    break
            time.sleep(check_interval)
            # 定期打印统计信息
            current_time = time.time()
            if current_time - last_stats_time >= stats_interval:
                print(self.get_statistics())
                last_stats_time = current_time
            # 检查是否应该恢复生成
            if not self._should_pause_generation():
                with self.lock:
                    if self.paused:
                        self.paused = False
                        self.condition.notify_all()
                        print("Generation resumed")

    def _should_pause_generation(self) -> bool:
        """
        判断是否应该暂停生成，基于新鲜度控制 - 改进的判断逻辑
        """
        try:
            queue_stats = self.message_queue_client.get_statistics()
            queue_size = queue_stats["queue_size"]
            current_trainer_version = queue_stats["current_param_version"]

            # 计算参数版本差异
            version_diff = self.current_param_version - current_trainer_version

            # 如果版本差异过大，暂停生成
            if version_diff >= self.staleness_threshold:
                print(
                    f"Should pause due to staleness: rollout_version={self.current_param_version}, "
                    f"trainer_version={current_trainer_version}, diff={version_diff}"
                )
                return True

            # 如果队列太满，也暂停生成
            if queue_size >= self.max_queue_size:
                print(f"Should pause due to full queue: size={queue_size}, max={self.max_queue_size}")
                return True

            return False

        except Exception as e:
            print(f"Error checking pause conditions: {e}")
            return True  # 出错时暂停生成

    def pause(self) -> bool:
        """暂停生成 - 供外部调用"""
        with self.lock:
            if not self.running:
                return False

            if self.paused:
                return True

            self.paused = True
            return True

    def resume(self) -> bool:
        """恢复生成 - 供外部调用"""
        with self.lock:
            if not self.running:
                return False

            if not self.paused:
                return True

            self.paused = False
            self.condition.notify_all()
            print("Generation resumed")
            return True

    def shutdown(self):
        """关闭Rollouter - 改进的关闭逻辑"""
        print("Shutting down Rollouter...")

        with self.lock:
            self.running = False
            self.paused = False
            self.condition.notify_all()

        # 等待生成线程结束
        if self.generation_thread and self.generation_thread.is_alive():
            print("Waiting for generation thread to finish...")
            self.generation_thread.join(timeout=10.0)

            if self.generation_thread.is_alive():
                print("Generation thread did not finish within timeout")

        # 等待监控线程结束
        if self.monitor_thread and self.monitor_thread.is_alive():
            print("Waiting for monitor thread to finish...")
            self.monitor_thread.join(timeout=5.0)

            if self.monitor_thread.is_alive():
                print("Monitor thread did not finish within timeout")

        # 关闭线程池
        if self.thread_executor:
            self.thread_executor.shutdown(wait=True)

        # 清理异步rollout管理器
        if hasattr(self, "async_rollout_manager"):
            try:
                # TODO: 添加异步rollout管理器的清理逻辑
                pass
            except Exception as e:
                print(f"Error cleaning up async rollout manager: {e}")

        print("Rollouter shutdown complete")

    def get_statistics(self) -> dict:
        with self.lock:
            queue_stats = self.message_queue_client.get_statistics()
            stats = {
                "total_generated_samples": self.total_generated_samples,
                "dropped_stale_samples": self.dropped_stale_samples,
                "current_param_version": self.current_param_version,
                "param_sync_requests": self.param_sync_requests,
                "last_sync_time": self.last_sync_time,
                "is_running": self.running,
                "sync_in_progress": self.sync_in_progress,
                "queue_size": f"{queue_stats['queue_size']}",
            }
            return stats

    def update_rollout_weights(self, param_version: int) -> bool:
        """
        更新rollout模型参数 - 改进的参数同步实现
        这个方法由外部Trainer调用

        Args:
            param_version: 新的参数版本号

        Returns:
            bool: 是否成功更新参数
        """
        print(f"Updating rollout weights to version {param_version}")

        with self.sync_lock:
            if self.sync_in_progress:
                print(f"Sync already in progress, skipping version {param_version}")
                return False

            self.sync_in_progress = True

        try:
            # 暂停rollout - 带超时机制
            if not self.rollout_controller.pause(timeout=10.0):
                print("Failed to pause rollout within timeout")
                return False

            # 等待当前generation完成（如果有的话）
            time.sleep(0.1)

            # 执行参数同步
            sync_success = self._execute_parameter_sync(param_version)

            if sync_success:
                self.current_param_version = param_version
                self.param_sync_requests += 1
                self.last_sync_time = time.time()
                print(f"Successfully updated rollout weights to version {param_version}")
            else:
                print(f"Failed to sync parameters to version {param_version}")

        except Exception as e:
            print(f"Error during parameter sync: {e}")
            sync_success = False
        finally:
            # 恢复rollout
            self.rollout_controller.resume()
            self.sync_in_progress = False

        return sync_success

    def _execute_parameter_sync(self, param_version: int) -> bool:
        """
        执行实际的参数同步 - 改进的同步逻辑

        Args:
            param_version: 目标参数版本

        Returns:
            bool: 是否同步成功
        """
        try:
            # 暂停推理引擎
            if self.async_rollout_mode and hasattr(self, "async_rollout_manager"):
                # 对于异步模式，暂停服务器
                pass  # 异步服务器的暂停在 pause() 中已经处理
            else:
                # 对于同步模式，使用sleep/wake_up机制
                sleep_futures = self.rollout_wg.sleep()
                ray.get(sleep_futures)

            # 执行参数同步
            if self.param_synchronizer:
                self.param_synchronizer.sync_weights()
                print("Parameter synchronization completed via synchronizer")
            else:
                # 直接使用rollout worker group的同步机制
                if hasattr(self.rollout_wg, "sync_rollout_weights"):
                    sync_futures = self.rollout_wg.sync_rollout_weights()
                    ray.get(sync_futures)
                    print("Parameter synchronization completed via rollout worker group")
                else:
                    print("No parameter synchronization mechanism available")
                    return False

            # 恢复推理引擎
            if self.async_rollout_mode and hasattr(self, "async_rollout_manager"):
                # 对于异步模式，恢复服务器
                pass  # 异步服务器的恢复在 resume() 中已经处理
            else:
                # 对于同步模式，唤醒workers
                wake_futures = self.rollout_wg.wake_up()
                ray.get(wake_futures)

            return True

        except Exception as e:
            print(f"Parameter sync execution failed: {e}")
            return False
