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
            max_queue_size=1000,
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

        print(f"[FullyAsyncRollouter] Creating datasets...")
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        self._validate_config()
        print(f"[FullyAsyncRollouter] Rollouter _create_dataloader...\n{train_dataset}\n{val_dataset}")
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        total_rollout_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.rollout.total_rollout_steps is not None:
            total_rollout_steps = self.config.rollout.total_rollout_steps

        self.total_rollout_steps = total_rollout_steps
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

        # Calculate the samples needed for a train, used to calculate staleness and interrupt rollout
        n_responses_per_prompt = self.config.actor_rollout_ref.rollout.n
        batch_size = self.config.data.train_batch_size
        required_samples = n_responses_per_prompt * batch_size
        self.max_required_samples = required_samples * (self.staleness_threshold + 1)

        # Worker groups
        self.rollout_wg = None
        self.message_queue_client = None

        # Concurrency control
        self.running = False
        self.paused = False
        self.generation_thread = None
        self.monitor_thread = None
        self.thread_executor = ThreadPoolExecutor(max_workers=2)
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)

        # Pause/resume statistics
        self.total_pause_time = 0.0
        self.last_pause_time = None

        # Parameter synchronization related
        self.param_synchronizer = None

        # queue size
        self.max_queue_size = max_queue_size

    def set_message_queue_client(self, message_queue_client: MessageQueueClient):
        """Set message queue client"""
        with self.lock:
            self.message_queue_client = message_queue_client

    def set_parameter_synchronizer(self, param_synchronizer):
        """Set parameter synchronizer"""
        with self.lock:
            self.param_synchronizer = param_synchronizer

    def get_rollout_wg(self):
        """Get rollout worker group"""
        return self.rollout_wg

    def update_param_version(self, version: int):
        """Update current parameter version"""
        with self.lock:
            old_version = self.current_param_version
            self.current_param_version = version
            # every time param change, reset train_step_samples
            self.train_step_samples = 0
            print(f"[FullyAsyncRollouter] Parameter version updated from {old_version} to {version}")

    def _validate_config(self):
        # Validate asynchronous training configuration
        if not hasattr(self.config, "async_training"):
            raise ValueError("[FullyAsyncRollouter] Missing async_training configuration")

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

    def fit(self):
        print(f"[FullyAsyncRollouter] Starting FullyAsyncRollouter...")

        if self.message_queue_client is None:
            raise ValueError("MessageQueue client not set. Call set_message_queue_client() first.")
        if self.param_synchronizer is None:
            raise ValueError("param_synchronizer client not set. Call set_parameter_synchronizer() first.")

        # 设置运行状态
        with self.lock:
            self.running = True
            self.paused = False

        self.generation_thread = threading.Thread(target=self._generation_loop, daemon=True)
        self.generation_thread.start()

        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

        self.generation_thread.join()
        self.monitor_thread.join()

        print(f"[FullyAsyncRollouter] Rollouter fit completed")

    def _generation_loop(self):
        """

        Main Generation Loop

        Loop Entry Requirements:
        1. Running status validation
        2. Interruption detection
        3. Freshness validation
        4. train_step_samples validation

        During Sample Generation Process:
        1. Running status validation
        2. Interruption detection
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
            pprint(f"[FullyAsyncRollouter] Initial validation metrics: {val_metrics}")
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

                while self.paused and self.running:
                    print(f"[FullyAsyncRollouter] Generation thread paused, waiting...")
                    self.condition.wait()

                if not self.running:
                    break

            metrics = {}
            timing_raw = {}

            with self.lock:
                batch, gen_batch = self._prepare_generate_batch(batch_dict)

            is_last_step = self.global_steps >= self.total_rollout_steps

            # generate a batch
            with marked_timer("gen", timing_raw, color="red"):
                if not self.async_rollout_mode:
                    gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                else:
                    gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                timing_raw.update(gen_batch_output.meta_info["timing"])
                gen_batch_output.meta_info.pop("timing", None)

            if gen_batch_output is not None:
                # prepare rollout metadata
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
                    success = self.message_queue_client.put_sample(
                        sample=ray.cloudpickle.dumps(queue_sample),
                        param_version=self.current_param_version,
                    )
                    with self.lock:
                        if success:
                            self.total_generated_samples += 1
                            self.train_step_samples += 1
                        else:
                            self.dropped_stale_samples += 1

            self.global_steps += 1

            if is_last_step:
                pprint(f"[FullyAsyncRollouter] Final validation metrics: {last_val_metrics}")
                break

        with self.lock:
            self.running = False

        # 发送终止信号
        self.message_queue_client.put_sample(
            sample=None,
            param_version=self.current_param_version,
        )

    def _monitor_loop(self):
        last_stats_time = time.time()
        stats_interval = 30.0
        check_interval = 5.0
        while True:
            with self.lock:
                if not self.running:
                    break
            time.sleep(check_interval)
            # 定期打印统计信息
            current_time = time.time()
            if current_time - last_stats_time >= stats_interval:
                print(f"[FullyAsyncRollouter] {self.get_statistics()}")
                last_stats_time = current_time
            if not self._should_pause_generation():
                with self.lock:
                    if self.paused:
                        self.paused = False
                        self.condition.notify_all()
                        print(f"[FullyAsyncRollouter] Generation resumed")

    def _should_pause_generation(self) -> bool:
        """Determine whether the build should be paused"""
        try:
            queue_stats = self.message_queue_client.get_statistics()
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
                print(f"[FullyAsyncRollouter] Should pause due to full queue: size={queue_size}, max={self.max_queue_size}")
                return True

            if self.train_step_samples >= self.max_required_samples:
                print(f"[FullyAsyncRollouter] Should pause due to step_generated_samples >= max_required_samples: "
                      f"self.step_generated_samples={self.train_step_samples}, max={self.max_required_samples}")
                return True

            return False

        except Exception as e:
            print(f"[FullyAsyncRollouter] Error checking pause conditions: {e}")
            return True

    def pause(self) -> bool:
        """pause rollout
        TODO integrated Partial Rollout
        """
        print(f"[FullyAsyncRollouter] pause")
        with self.lock:
            if not self.running:
                return False

            if self.paused:
                return True

            self.paused = True
            return True

    def resume(self) -> bool:
        """resume rollout
        TODO integrated Partial Rollout
        """
        print(f"[FullyAsyncRollouter] resume")
        with self.lock:
            if not self.running:
                return False

            if not self.paused:
                return True

            self.paused = False
            self.condition.notify_all()
            return True

    def get_statistics(self) -> dict:
        with self.lock:
            queue_stats = self.message_queue_client.get_statistics()
            stats = {
                "is_running": self.running,
                "total_generated_samples": self.total_generated_samples,
                "train_step_samples": self.train_step_samples,
                "dropped_stale_samples": self.dropped_stale_samples,
                "current_param_version": self.current_param_version,
                "queue_size": queue_stats['queue_size'],
                "queue_max_size": self.max_queue_size,
            }
            return stats
