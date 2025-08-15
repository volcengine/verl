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
import asyncio
import time
from pprint import pprint

import ray
from omegaconf import OmegaConf

from recipe.fully_async_policy.message_queue import MessageQueueClient, RolloutSample
from recipe.fully_async_policy.utils import calculate_one_step_size
from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role, WorkerType
from verl.utils.tracking import ValidationGenerationsLogger


@ray.remote(num_cpus=10, max_concurrency=100)
class FullyAsyncRollouter(RayPPOTrainer):
    """
    Asynchronous sample generator, responsible for continuously generating training samples
    and putting them into MessageQueue
    Based on the mature implementation improvements of OneStepOffRayTrainer
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
        assert self.config.data.train_batch_size == 0, "train_batch_size must be zero"
        assert self.config.data.gen_batch_size == 1, "gen_batch_size must be one"

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

        print("[FullyAsyncRollouter] Creating datasets...")
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        self._validate_config()
        print(f"[FullyAsyncRollouter] Rollouter _create_dataloader...\n{train_dataset}\n{val_dataset}")

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        self.total_rollout_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.rollout.total_rollout_steps is not None:
            self.total_rollout_steps = min(self.config.rollout.total_rollout_steps, self.total_rollout_steps)
        print(f"[FullyAsyncRollouter] Total rollout steps: {self.total_rollout_steps}")

        # Rollouter parameter configuration
        self.message_queue_client = None

        self.current_param_version = 0

        # Freshness control - improved configuration management
        async_config = config.async_training
        self.staleness_threshold = async_config.get("staleness_threshold", 3)

        # Statistics
        self.total_generated_samples = 0
        self.train_step_samples = 0
        self.dropped_stale_samples = 0

        # Worker groups
        self.rollout_wg = None
        self.message_queue_client = None

        # Concurrency control
        self.running = False
        self.paused = False
        # Initialize async locks directly - asyncio.Lock() creation is synchronous
        self.lock = asyncio.Lock()
        self.condition = asyncio.Condition(self.lock)

        # Pause/resume statistics
        self.total_pause_time = 0.0
        self.last_pause_time = None

        # Parameter synchronization related
        self.param_synchronizer = None

        self.async_rollout_manager = None

        # 流式处理相关配置
        self.max_concurrent_samples = async_config.get("max_concurrent_samples", 512)  # 最大并发处理样本数

        # 流式处理统计
        self.max_processing_time = 0.0  # 最长处理时间
        self.processed_sample_count = 0  # 已处理的样本计数
        self.active_sample_count = 0  # 当前正在处理的样本数
        self.queue_full_pause_count = 0  # 队列满导致的暂停次数

        # Calculate the samples needed for a train, used to calculate staleness and interrupt rollout
        self.required_samples = calculate_one_step_size(
            self.minimal_bsz, config.actor_rollout_ref.actor.ppo_mini_batch_size
        )
        self.max_required_samples = self.required_samples * (self.staleness_threshold + 1)

        # queue size
        self.max_queue_size = self.max_required_samples * 10  # x 10 avoid deadlock

    async def set_message_queue_client(self, message_queue_client: MessageQueueClient):
        """Set message queue client"""
        async with self.lock:
            self.message_queue_client = message_queue_client

    async def set_parameter_synchronizer(self, param_synchronizer):
        """Set parameter synchronizer"""
        async with self.lock:
            self.param_synchronizer = param_synchronizer

    def get_rollout_wg(self):
        """Get rollout worker group"""
        return self.rollout_wg

    def get_max_queue_size(self):
        return self.max_queue_size

    async def update_param_version(self, version: int):
        """Update current parameter version"""
        async with self.lock:
            old_version = self.current_param_version
            self.current_param_version = version
            # every time param change, reset train_step_samples
            self.train_step_samples = 0
            print(f"[FullyAsyncRollouter] Parameter version updated from {old_version} to {version}")

    def _validate_config(self):
        # Validate asynchronous training configuration
        if not hasattr(self.config, "async_training"):
            raise ValueError("[FullyAsyncRollouter] Missing async_training configuration")

        super()._validate_config()

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
        for epoch in range(self.config.rollout.total_epochs):
            iterator = iter(self.train_dataloader)
            for batch_dict in iterator:
                yield epoch, batch_dict

    def _init_async_rollout_manager(self):
        # create async rollout manager and request scheduler
        assert self.config.actor_rollout_ref.rollout.mode == "async"
        from verl.experimental.agent_loop import AgentLoopManager

        self.async_rollout_mode = True
        self.async_rollout_manager = AgentLoopManager(
            config=self.config,
            worker_group=self.rollout_wg,
        )

    # 添加样本到待处理队列的协程
    async def _feed_samples(self):
        continuous_iterator = self._create_continuous_iterator()
        sample_count = 0
        should_stop = False

        for epoch, batch_dict in continuous_iterator:
            if should_stop:  # 检查停止标志
                break

            # 类似 _prepare_generate_batch 的逻辑：分离数据
            original_batch, gen_data = self._prepare_single_generation_data(batch_dict)

            # 根据 rollout.n 进行重复
            n_repeats = self.config.actor_rollout_ref.rollout.n

            for rollout_n_index in range(n_repeats):
                sample_id = f"sample_{epoch}_{sample_count}_{rollout_n_index}"

                # 创建部分 RolloutSample，不包含 _gen_data（因为它不在数据类定义中）
                partial_rollout_sample = RolloutSample(
                    original_batch_dict=original_batch,
                    agent_loop_output=None,  # 待处理后填充
                    sample_id=sample_id,
                    epoch=epoch,
                    rollout_n_index=rollout_n_index,
                    original_sample_index=sample_count,
                    processing_time=0.0,  # 待处理后填充
                    generation_timestamp=0.0,  # 待处理后填充
                    param_version=0,  # 待处理后填充
                )

                # 动态添加临时字段（处理完后删除）
                partial_rollout_sample._gen_data = gen_data

                await self.pending_samples_queue.put(partial_rollout_sample)

                # 检查是否到达最后一步
                if self.global_steps >= self.total_rollout_steps:
                    print(
                        f"[FullyAsyncRollouter] 达到最大步数，停止添加新样本 "
                        f"{self.global_steps} >= {self.total_rollout_steps}"
                    )
                    should_stop = True  # 设置停止标志
                    break

                self.global_steps += 1

            sample_count += 1

        # 发送结束信号
        await self.pending_samples_queue.put("DONE")
        print(f"[FullyAsyncRollouter] 样本添加完成，总共添加了 {self.global_steps} 个步骤的样本")

    def _prepare_single_generation_data(self, batch_dict):
        """
        类似 ray_trainer._prepare_generate_batch 的逻辑，但针对单个样本
        分离出用于生成的数据和需要保留的原始数据

        Returns:
            tuple: (original_batch_dict, gen_data_for_single_sample)
        """
        from verl import DataProto

        # 创建完整的 DataProto
        full_batch = DataProto.from_single_dict(batch_dict)

        # 定义需要传递给生成服务器的字段
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]

        # 处理可选字段
        optional_fields = [
            "multi_modal_data",
            "raw_prompt",
            "tools_kwargs",
            "interaction_kwargs",
            "index",
            "agent_name",
        ]

        for field in optional_fields:
            if field in full_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append(field)

        # 分离数据：gen_batch 用于生成，original_batch 保留原始信息
        gen_batch = full_batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )

        # 添加全局步数到生成数据
        gen_batch.meta_info["global_steps"] = self.global_steps

        # 保留原始 batch 信息（转换为字典格式以便序列化）
        original_batch_dict = {
            "batch": {k: v.clone() if hasattr(v, "clone") else v for k, v in full_batch.batch.items()},
            "non_tensor_batch": dict(full_batch.non_tensor_batch),
            "meta_info": dict(full_batch.meta_info),
        }

        return original_batch_dict, gen_batch

    async def _submit_worker(self):
        """流式处理工作协程 - 逐个样本立即提交处理，不等待批次"""
        active_tasks = set()

        while True:
            # 获取待处理的部分 RolloutSample
            partial_rollout_sample = await self.pending_samples_queue.get()

            if partial_rollout_sample == "DONE":
                print("收到结束信号，等待剩余任务完成...")
                # 等待所有活动任务完成
                if active_tasks:
                    await asyncio.gather(*active_tasks, return_exceptions=True)
                break

            # 检查并发数是否超限
            while len(active_tasks) >= self.max_concurrent_samples:
                print(f"达到最大并发数 {self.max_concurrent_samples}，等待任务完成...")
                # 等待至少一个任务完成
                done_tasks, active_tasks = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)
                # 清理已完成的任务
                for task in done_tasks:
                    await task

            # 立即提交单个样本处理
            task = asyncio.create_task(
                self._process_single_sample_streaming(partial_rollout_sample),
                name=f"process_{partial_rollout_sample.sample_id}",
            )
            active_tasks.add(task)

            # 标记队列任务完成
            self.pending_samples_queue.task_done()

    async def _process_single_sample_streaming(self, partial_rollout_sample):
        """流式处理单个样本"""
        # 检查是否需要暂停处理，如果需要暂停则等待resume信号
        while await self._should_pause_generation() and self.running:
            print(f"[FullyAsyncRollouter] 暂停处理样本 {partial_rollout_sample.sample_id}，等待resume...")
            async with self.lock:
                await self.condition.wait()
            print(f"[FullyAsyncRollouter] 样本 {partial_rollout_sample.sample_id} 收到resume信号，继续处理")

        # 如果系统已停止，跳过处理
        if not self.running:
            print(f"[FullyAsyncRollouter] 系统已停止，跳过样本 {partial_rollout_sample.sample_id}")
            return

        start_time = time.time()

        # 从 RolloutSample 中提取生成数据（临时字段）
        gen_data = partial_rollout_sample._gen_data

        # 将单个样本数据包装成 DataProto (用于 generate_single_sample_async)
        gen_batch_single = DataProto.from_items([gen_data])

        # 调用异步生成方法
        agent_loop_output, processing_time = await self.async_rollout_manager.generate_single_sample_async(
            gen_batch_single, partial_rollout_sample.sample_id
        )
        end_time = time.time()

        # 直接更新 RolloutSample 对象，填充剩余字段
        partial_rollout_sample.agent_loop_output = agent_loop_output
        partial_rollout_sample.processing_time = processing_time
        partial_rollout_sample.generation_timestamp = time.time()
        partial_rollout_sample.param_version = self.current_param_version

        # 删除临时字段
        delattr(partial_rollout_sample, "_gen_data")

        # 直接放入结果队列
        await self.result_queue.put(partial_rollout_sample)

        async with self.lock:
            self.processed_sample_count += 1
            # 更新最大处理时间统计
            if processing_time > self.max_processing_time:
                self.max_processing_time = processing_time

        print(
            f"[FullyAsyncRollouter] 样本 {partial_rollout_sample.sample_id} 处理完成，"
            f"耗时 {processing_time:.2f}s {end_time - start_time:.2f}s"
        )

    async def _consumer_worker(self):
        """消费者协程，负责从结果队列获取处理结果并放入消息队列"""
        while True:
            async with self.lock:
                if not self.running:
                    # 如果系统停止但还有结果待处理，继续处理
                    if self.result_queue.empty():
                        break

            # 从结果队列获取 RolloutSample
            rollout_sample = await self.result_queue.get()

            # 直接将 RolloutSample 放入消息队列
            success = await self.message_queue_client.put_sample(
                sample=ray.cloudpickle.dumps(rollout_sample),
                param_version=rollout_sample.param_version,
            )

            async with self.lock:
                if success:
                    self.total_generated_samples += 1
                    self.train_step_samples += 1
                else:
                    self.dropped_stale_samples += 1

            print(
                f"[FullyAsyncRollouter] 消费样本 {rollout_sample.sample_id}: "
                f"{'成功' if success else '失败'}放入到消息队列, "
                f"处理时间 {rollout_sample.processing_time:.2f}s"
            )

            # 标记结果队列任务完成
            self.result_queue.task_done()

    async def _streaming_generation_main(self):
        """流式处理的主入口方法，包含初始化和验证逻辑"""
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
            pprint(f"[FullyAsyncRollouter] Initial validation metrics: {val_metrics}")
            self.logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # we start from step 1
        self.global_steps += 1

        # 确保async_rollout_manager已经初始化
        if self.async_rollout_manager is None:
            self._init_async_rollout_manager()

        # 启动流式处理循环
        """流式样本生成主循环 - 优化版本，确保先完成的样本优先进入队列"""
        print(f"[FullyAsyncRollouter] 启动流式处理模式，最大并发样本数: {self.max_concurrent_samples}")

        # 初始化异步队列
        self.pending_samples_queue = asyncio.Queue(maxsize=self.max_concurrent_samples)
        self.result_queue = asyncio.Queue()

        # 启动流式处理协程和消费者协程
        self.feed_task = asyncio.create_task(self._feed_samples())
        self.stream_processor_task = asyncio.create_task(self._submit_worker())
        self.consumer_task = asyncio.create_task(self._consumer_worker())
        # 启动样本添加协程

        try:
            # 等待样本添加完成
            await self.feed_task
            print("[FullyAsyncRollouter] 样本添加完成")

            # 等待流式处理完成
            await self.stream_processor_task
            print("[FullyAsyncRollouter] 流式处理完成")

            # 等待结果队列清空
            await self.result_queue.join()
            print("[FullyAsyncRollouter] 所有结果处理完成")

        except Exception as e:
            print(f"[FullyAsyncRollouter] 流式处理异常: {e}")

        finally:
            # 取消所有任务
            if self.stream_processor_task:
                self.stream_processor_task.cancel()
            if self.consumer_task:
                self.consumer_task.cancel()

            # 等待任务结束
            await asyncio.gather(self.stream_processor_task, self.consumer_task, return_exceptions=True)

        async with self.lock:
            self.running = False

        # 发送终止信号
        await self.message_queue_client.put_sample(
            sample=None,
            param_version=self.current_param_version,
        )

    async def fit(self):
        """
        Start the async rollouter - entry point that sets up and runs async tasks
        Main async fit method that coordinates all coroutines"""

        print("[FullyAsyncRollouter] Starting FullyAsyncRollouter...")

        if self.message_queue_client is None:
            raise ValueError("MessageQueue client not set. Call set_message_queue_client() first.")
        if self.param_synchronizer is None:
            raise ValueError("param_synchronizer client not set. Call set_parameter_synchronizer() first.")

        # 设置运行状态
        async with self.lock:
            self.running = True
            self.paused = False

        # 创建主要的异步任务
        generation_task = asyncio.create_task(self._streaming_generation_main())
        monitor_task = asyncio.create_task(self._async_monitor_loop())

        try:
            # 并发运行生成和监控任务
            await asyncio.gather(generation_task, monitor_task, return_exceptions=True)
        except Exception as e:
            print(f"[FullyAsyncRollouter] 异步任务执行出错: {e}")
        finally:
            # 清理任务
            if not generation_task.done():
                generation_task.cancel()
            if not monitor_task.done():
                monitor_task.cancel()

            # 等待任务完成
            await asyncio.gather(generation_task, monitor_task, return_exceptions=True)

        print("[FullyAsyncRollouter] Rollouter fit completed")

    async def _async_monitor_loop(self):
        """
        Async coroutine for monitoring:
        Function 1: Log information output
        Function 2: Trigger rollout recovery
        """
        last_stats_time = time.time()
        stats_interval = 30.0
        check_interval = 5.0

        while True:
            async with self.lock:
                if not self.running:
                    break

            await asyncio.sleep(check_interval)

            # 定期打印统计信息
            current_time = time.time()
            if current_time - last_stats_time >= stats_interval:
                stats = await self.get_statistics()
                print(f"[FullyAsyncRollouter] {stats}")
                last_stats_time = current_time

            if not await self._should_pause_generation():
                await self.resume()

    async def _should_pause_generation(self) -> bool:
        """Determine whether the build should be paused"""
        queue_stats = self.message_queue_client.get_statistics_sync()
        queue_size = queue_stats["queue_size"]
        current_trainer_version = queue_stats["current_param_version"]

        version_diff = self.current_param_version - current_trainer_version

        if version_diff > self.staleness_threshold:
            print(
                "[FullyAsyncRollouter] "
                f"Should pause due to version_diff > self.staleness_threshold: "
                f"rollout_version={self.current_param_version}, "
                f"trainer_version={current_trainer_version}, diff={version_diff}"
            )
            return True

        if queue_size >= self.max_queue_size:
            print(
                f"[FullyAsyncRollouter] Should pause due to full queue: "
                f"size={queue_size}, max={self.max_queue_size}"
            )
            return True

        if self.train_step_samples >= self.max_required_samples:
            print(
                f"[FullyAsyncRollouter] Should pause due to step_generated_samples >= max_required_samples: "
                f"self.step_generated_samples={self.train_step_samples}, max={self.max_required_samples}"
            )
            return True

        return False

    async def pause(self) -> bool:
        """pause rollout
        TODO integrated Partial Rollout
        """
        print("[FullyAsyncRollouter] pause")
        async with self.lock:
            if not self.running:
                return False

            if self.paused:
                return True

            self.paused = True
            return True

    async def resume(self) -> bool:
        """resume rollout
        TODO integrated Partial Rollout
        """
        print("[FullyAsyncRollouter] resume")
        async with self.lock:
            if not self.running:
                return False

            if not self.paused:
                return True

            self.paused = False
            self.condition.notify_all()
            return True

    async def get_statistics(self) -> dict:
        async with self.lock:
            queue_stats = self.message_queue_client.get_statistics_sync()
            stats = {
                "is_running": self.running,
                "total_generated_samples": self.total_generated_samples,
                "train_step_samples": self.train_step_samples,
                "dropped_stale_samples": self.dropped_stale_samples,
                "current_param_version": self.current_param_version,
                "queue_size": queue_stats["queue_size"],
                "queue_max_size": self.max_queue_size,
                "max_concurrent_samples": self.max_concurrent_samples,
                "max_processing_time": self.max_processing_time,
                "pending_samples_queue_size": self.pending_samples_queue.qsize(),
                "result_queue_size": self.result_queue.qsize(),
            }

            return stats

