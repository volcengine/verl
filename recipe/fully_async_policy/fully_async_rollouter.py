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
from pprint import pformat, pprint

import ray
from omegaconf import OmegaConf

from recipe.fully_async_policy.detach_utils import (
    RolloutSample,
    ValidateMetrics,
    merge_rollout_sample,
    prepare_single_generation_data,
)
from recipe.fully_async_policy.message_queue import MessageQueueClient
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role, WorkerType
from verl.utils.profiler import marked_timer
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
        assert self.config.async_training.staleness_threshold >= 0, "staleness_threshold must larger than 0"
        assert self.config.async_training.trigger_parameter_sync_step >= 1, (
            "trigger_parameter_sync_step must larger than 1"
        )

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
        self.total_train_steps = None

        # ==================== fully async config ====================

        # Rollouter parameter configuration
        self.message_queue_client = None

        # Worker groups: rollout_wg is same to actor_rollout_wg
        self.rollout_wg = None
        self.actor_rollout_wg = None
        self.async_rollout_manager = None

        # Config
        self.staleness_threshold: int = config.async_training.get("staleness_threshold", 1)
        self.required_samples = None
        self.max_required_samples = None
        # 单次最多扔一次更新需要的样本
        self.max_concurrent_samples = None
        # queue size
        self.max_queue_size = None

        # Statistics
        self.current_param_version = 0
        self.total_generated_samples = 0
        self.staleness_samples = 0
        self.dropped_stale_samples = 0
        self.processed_sample_count = 0  # 已处理的样本计数
        self.global_steps = 0

        # Concurrency control
        self.paused = False
        self.running = True
        # 通过 pause 和 resume 控制 monitor_loop 中，是否进行 尝试恢复 操作
        self.monitor_loop_trigger = True

        # Initialize async locks directly
        self.lock = asyncio.Lock()
        self.condition = asyncio.Condition(self.lock)

        # 初始化异步队列
        self.pending_queue = asyncio.Queue(maxsize=128)
        self.active_tasks = set()
        self.result_queue = asyncio.Queue()
        self.cancel_queue = asyncio.Queue()

    async def set_message_queue_client(self, message_queue_client: MessageQueueClient):
        """Set message queue client"""
        async with self.lock:
            self.message_queue_client = message_queue_client

    async def set_required_samples(self, required_samples: int):
        async with self.lock:
            self.required_samples = int(required_samples)
            self.max_required_samples = (
                self.required_samples
                * (self.staleness_threshold + 1)
                * self.config.async_training.trigger_parameter_sync_step
            )
            self.total_train_steps = int(
                self.total_rollout_steps
                / (self.required_samples * self.config.async_training.trigger_parameter_sync_step)
            )

            # 单次最多扔一次更新需要的样本
            self.max_concurrent_samples = self.required_samples
            self.max_queue_size = self.max_required_samples

            print(
                f"[FullyAsyncRollouter] required_samples : {self.required_samples} "
                f"max_required_samples: {self.max_required_samples} "
                f"max_queue_size: {self.max_queue_size} "
                f"total_train_steps: {self.total_train_steps} "
                f"total_rollout_steps: {self.total_rollout_steps} "
                f"max_concurrent_samples: {self.max_concurrent_samples} "
            )

    def get_rollout_wg(self):
        """Get rollout worker group"""
        return self.rollout_wg

    def get_max_queue_size(self):
        return self.max_queue_size

    def get_total_train_steps(self):
        return self.total_train_steps

    async def update_param_version(self, version: int, validate: bool = False, global_steps: int = 0):
        """Update current parameter version"""
        async with self.lock:
            old_version = self.current_param_version
            self.current_param_version = version
            # every time param change, reset staleness_samples
            self.staleness_samples = 0
            print(
                f"[FullyAsyncRollouter][Public][update_param_version] "
                f"Parameter version updated from {old_version} to {version}"
            )
            timing_raw = {}
            if (
                self.val_reward_fn is not None
                and self.config.rollout.test_freq > 0
                and self.current_param_version % self.config.rollout.test_freq == 0
                and self.current_param_version > 0 # don't test here in the initial parameter sync
            ) or (
                validate and self.val_reward_fn is not None
            ):
                with marked_timer("testing", timing_raw, color="green"):
                    val_metrics: dict = self._validate()
                data = ValidateMetrics(timing_raw=timing_raw,
                                       metrics=val_metrics,
                                       global_steps=global_steps,
                                       param_version=version)
                await self.message_queue_client.put_validate(ray.cloudpickle.dumps(data))

    def _validate_config(self):
        # Validate asynchronous training configuration
        if not hasattr(self.config, "async_training"):
            raise ValueError("[FullyAsyncRollouter] Missing async_training configuration")
        assert self.config.actor_rollout_ref.rollout.calculate_log_probs, "must rollout calculate log_probs"
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

        for epoch, batch_dict in continuous_iterator:
            # 类似 _prepare_generate_batch 的逻辑：分离数据
            full_batch = prepare_single_generation_data(
                batch_dict, self.global_steps, self.config.actor_rollout_ref.rollout.n
            )

            sample_id = f"sample_{epoch}_{self.global_steps}"

            rollout_sample = RolloutSample(
                full_batch=full_batch,
                agent_loop_output_list=[None] * self.config.actor_rollout_ref.rollout.n,  # 待处理后填充
                sample_id=sample_id,
                epoch=epoch,
                param_version=0,  # 待处理后填充
                processing_times=[],
                rollout_status={},
            )

            await self.pending_queue.put(rollout_sample)

            # 检查是否到达最后一步
            if self.global_steps >= self.total_rollout_steps:
                print(
                    f"[FullyAsyncRollouter][Feed] "
                    f"达到最大步数，停止添加新样本 "
                    f"{self.global_steps} >= {self.total_rollout_steps}"
                )
                break

            self.global_steps += 1

        # 发送结束信号
        await self.pending_queue.put("DONE")
        print(f"[FullyAsyncRollouter][Feed] 样本添加完成，总共添加了 {self.global_steps} 个步骤的样本")

    async def _processor_worker(self):
        """流式处理工作协程 - 逐个样本立即提交处理，不等待批次"""

        while True:
            simple_from_cancel_queue = False
            if not self.cancel_queue.empty():
                rollout_sample = await self.cancel_queue.get()
                simple_from_cancel_queue = True
            else:
                rollout_sample = await self.pending_queue.get()
                self.staleness_samples += 1

            # 判断是否需要暂停
            # self.paused 由 pause() 和 self._should_pause_generation() 负责修改
            if self.paused or await self._should_pause_generation():
                print("[FullyAsyncRollouter][Processor] 收到暂停信号，等待剩余任务完成...")
                while self.active_tasks:
                    async with self.lock:
                        # 获取锁后，active_tasks 数量会发生变化，需要再次校验
                        if self.active_tasks:
                            done_tasks, self.active_tasks = await asyncio.wait(
                                self.active_tasks, return_when=asyncio.FIRST_COMPLETED
                            )
                        for task in done_tasks:
                            await task
                async with self.lock:
                    self.paused = True

                async with self.lock:
                    while self.paused:
                        await self.condition.wait()

            # 获取待处理的部分 RolloutSample
            if rollout_sample == "DONE":
                print("[FullyAsyncRollouter][Processor] 收到结束信号，等待剩余任务完成...")
                while self.active_tasks:
                    async with self.lock:
                        if self.active_tasks:
                            done_tasks, self.active_tasks = await asyncio.wait(
                                self.active_tasks, return_when=asyncio.FIRST_COMPLETED
                            )
                        for task in done_tasks:
                            await task
                break

            # 检查并发数是否超限
            while len(self.active_tasks) >= self.max_concurrent_samples:
                async with self.lock:
                    if self.active_tasks:
                        done_tasks, self.active_tasks = await asyncio.wait(
                            self.active_tasks, return_when=asyncio.FIRST_COMPLETED
                        )
                    for task in done_tasks:
                        await task

            # 立即提交单个样本处理
            async with self.lock:
                # pause结束后，获取到锁，还需要判断是否是暂停阶段，否则继续等待
                while self.paused:
                    await self.condition.wait()
                task = asyncio.create_task(
                    self._process_single_sample_streaming(rollout_sample),
                    name=rollout_sample.sample_id,
                )
                self.active_tasks.add(task)

            # 标记队列任务完成
            if simple_from_cancel_queue:
                self.cancel_queue.task_done()
            else:
                self.pending_queue.task_done()

    async def _process_single_sample_streaming(self, rollout_sample: RolloutSample):
        """流式处理单个样本"""
        # 调用异步生成方法
        agent_loop_output_list = await self.async_rollout_manager.generate_single_sample_async(
            rollout_sample.full_batch, rollout_sample.agent_loop_output_list
        )
        # 直接更新 RolloutSample 对象，填充剩余字段
        rollout_sample.agent_loop_output_list = agent_loop_output_list
        rollout_sample.param_version = self.current_param_version
        rollout_sample.rollout_status = await self.get_statistics()

        is_cancel = False
        # 收集所有信息
        for agent_loop in agent_loop_output_list:
            if not is_cancel and agent_loop.is_cancel:
                is_cancel = True

        # rollout_data = {
        #     "cost": [f"{agent_loop.metrics.generate_sequences:.2f}s" for agent_loop in agent_loop_output_list],
        #     "len": [len(agent_loop.response_ids) for agent_loop in agent_loop_output_list],
        # }
        # if is_cancel:
        #     rollout_data["cancel"] = [agent_loop.is_cancel for agent_loop in agent_loop_output_list]
        # formatted_data = pformat(rollout_data, width=200, compact=True)
        # print(f"[FullyAsyncRollouter] rollout {rollout_sample.sample_id} {formatted_data}")

        if is_cancel:
            # 放入 cancel 队列中，等待恢复生成
            await self.cancel_queue.put(rollout_sample)
        else:
            # 否则放入结果队列
            await self.result_queue.put(rollout_sample)

        self.processed_sample_count += 1

    async def _consumer_worker(self):
        """消费者协程，负责从结果队列获取处理结果并放入消息队列"""
        while True:
            # 从结果队列获取 RolloutSample
            rollout_sample = await self.result_queue.get()
            rollout_sample = merge_rollout_sample(self.config, self.tokenizer, rollout_sample)

            # 直接将 RolloutSample 放入消息队列
            success = await self.message_queue_client.put_sample(
                sample=ray.cloudpickle.dumps(rollout_sample),
                param_version=rollout_sample.param_version,
            )
            if success:
                self.total_generated_samples += 1
            else:
                self.dropped_stale_samples += 1

            # 标记结果队列任务完成
            self.result_queue.task_done()

    async def _streaming_generation_main(self):
        """流式处理的主入口方法，包含初始化和验证逻辑"""

        # we start from step 1
        self.global_steps += 1

        # 确保async_rollout_manager已经初始化
        if self.async_rollout_manager is None:
            self._init_async_rollout_manager()

        # 启动流式处理循环
        print(f"[FullyAsyncRollouter] 启动流式处理模式，最大并发样本数: {self.max_concurrent_samples}")

        # 启动流式处理协程和消费者协程
        self.feed_task = asyncio.create_task(self._feed_samples())
        self.processor_task = asyncio.create_task(self._processor_worker())
        self.consumer_task = asyncio.create_task(self._consumer_worker())
        # 启动样本添加协程

        try:
            # 等待样本添加完成
            await self.feed_task
            print("[FullyAsyncRollouter] 样本添加完成")

            # 等待流式处理完成
            await self.processor_task
            print("[FullyAsyncRollouter] 流式处理完成")

            # 等待结果队列清空
            await self.result_queue.join()
            print("[FullyAsyncRollouter] 所有结果处理完成")

        except Exception as e:
            print(f"[FullyAsyncRollouter] 流式处理异常: {e}")

        finally:
            # 取消所有任务
            if self.processor_task:
                self.processor_task.cancel()
            if self.consumer_task:
                self.consumer_task.cancel()

            # 等待任务结束
            await asyncio.gather(self.processor_task, self.consumer_task, return_exceptions=True)

        # 发送终止信号
        await self.message_queue_client.put_sample(
            sample=None,
            param_version=self.current_param_version,
        )

        async with self.lock:
            self.running = False

    async def fit(self):
        """
        Start the async rollouter - entry point that sets up and runs async tasks
        Main async fit method that coordinates all coroutines"""

        print("[FullyAsyncRollouter] Starting FullyAsyncRollouter...")

        if self.message_queue_client is None:
            raise ValueError("MessageQueue client not set. Call set_message_queue_client() first.")

        # 设置运行状态
        async with self.lock:
            self.paused = False
            self.running = True

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
        stats_interval = 60.0
        check_interval = 10.0

        while True:
            async with self.lock:
                if not self.running:
                    break
            await asyncio.sleep(check_interval)
            # 定期打印统计信息
            current_time = time.time()
            if current_time - last_stats_time >= stats_interval:
                stats = await self.get_statistics()
                print(f"[FullyAsyncRollouter][MonitorLoop][Statistics] {pformat(stats)}")
                last_stats_time = current_time

            # pause 和 resume 之间，不进行恢复操作
            if self.monitor_loop_trigger:
                if not await self._should_pause_generation():
                    async with self.lock:
                        self.paused = False
                        self.condition.notify_all()

    async def _should_pause_generation(self) -> bool:
        """Determine whether the build should be paused"""
        queue_stats = self.message_queue_client.get_statistics_sync()
        queue_size = queue_stats["queue_size"]
        current_trainer_version = queue_stats["current_param_version"]

        version_diff = self.current_param_version - current_trainer_version

        if version_diff > self.staleness_threshold:
            if not self.paused:
                print(
                    "[FullyAsyncRollouter][ShouldPause] "
                    f"due to version_diff > self.staleness_threshold: "
                    f"rollout_version={self.current_param_version}, "
                    f"trainer_version={current_trainer_version}, diff={version_diff}"
                )
            return True

        if queue_size >= self.max_queue_size:
            if not self.paused:
                print(
                    f"[FullyAsyncRollouter][ShouldPause]  "
                    f"due to full queue: size={queue_size}, max={self.max_queue_size}"
                )
            return True

        if self.staleness_samples > self.max_required_samples:
            if not self.paused:
                print(
                    "[FullyAsyncRollouter][ShouldPause] "
                    f"due to "
                    f"staleness_samples {self.staleness_samples} > max_required_samples {self.max_required_samples} "
                )
            return True

        return False

    async def pause(self):
        """pause rollout
        TODO async_rollout_manager clear kv cache
        """
        print("[FullyAsyncRollouter][Public][Pause]")
        async with self.lock:
            self.paused = True
            # 取消rollout所有任务
            if self.config.async_training.partial_rollout:
                await self.async_rollout_manager.cancel_async()
            if self.active_tasks:
                await asyncio.gather(*self.active_tasks, return_exceptions=True)
                self.active_tasks.clear()
                print("[FullyAsyncRollouter][Public][Pause] All active tasks completed")
            self.async_rollout_manager.sleep()
            self.async_rollout_manager.wake_up()
            self.monitor_loop_trigger = False

    async def resume(self):
        print("[FullyAsyncRollouter][Public][Resume]")
        async with self.lock:
            self.paused = False
            self.monitor_loop_trigger = True
            self.condition.notify_all()

            if self.config.async_training.partial_rollout:
                await self.async_rollout_manager.resume_async()

    async def get_statistics(self) -> dict:
        queue_stats = self.message_queue_client.get_statistics_sync()

        stats = {
            "current_param_version": self.current_param_version,
            "total_generated_samples": self.total_generated_samples,
            "staleness_samples": self.staleness_samples,
            "dropped_stale_samples": self.dropped_stale_samples,
            "max_queue_size": self.max_queue_size,
            "queue_size": queue_stats["queue_size"],
            "max_concurrent_samples": self.max_concurrent_samples,
            "pending_queue_size": self.pending_queue.qsize(),
            "active_tasks_size": len(self.active_tasks),
            "result_queue_size": self.result_queue.qsize(),
            "max_required_samples": self.max_required_samples,
            "required_samples": self.required_samples,
            "staleness_threshold": self.staleness_threshold,
            "cancel_queue_size": self.cancel_queue.qsize(),
        }

        return stats
