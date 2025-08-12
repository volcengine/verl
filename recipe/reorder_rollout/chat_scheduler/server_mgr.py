# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
import logging
import os
import threading

import ray
from omegaconf import DictConfig

from recipe.reorder_rollout.chat_scheduler.chat_scheduler import (
    ChatCompletionScheduler,
    ReorderScheduler,
    ReorderSchedulerMixin,
)
from verl import DataProto
from verl.single_controller.ray import RayWorkerGroup
from verl.workers.rollout.async_server import async_server_class

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AsyncLLMServerManager:
    """AsyncLLMServerManager manage a group of vllm instances, i.e AsyncvLLMServer."""

    def __init__(self, config: DictConfig, worker_group: RayWorkerGroup):
        """Initialize AsyncLLMServerManager.

        Args:
            config: DictConfig, actor_rollout_ref config.
            worker_group: RayWorkerGroup, worker group of AsyncActorRolloutRefWorker.
        """
        self.full_config = config
        self.config = config.actor_rollout_ref
        self.worker_group = worker_group

        self.rollout_tp_size = self.config.rollout.tensor_model_parallel_size
        self.rollout_dp_size = self.worker_group.world_size // self.rollout_tp_size

        register_center = ray.get_actor(f"{self.worker_group.name_prefix}_register_center")
        workers_info = ray.get(register_center.get_worker_info.remote())
        assert len(workers_info) == self.worker_group.world_size

        self.async_llm_servers = [None] * self.rollout_dp_size
        self.server_addresses = [None] * self.rollout_dp_size

        server_class = async_server_class(
            rollout_backend=self.config.rollout.name,
        )

        # Start all server instances, restart if address already in use.
        unready_dp_ranks = set(range(self.rollout_dp_size))
        while len(unready_dp_ranks) > 0:
            servers = {
                rollout_dp_rank: server_class.options(
                    # make sure AsyncvLLMServer colocates with its corresponding workers
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=workers_info[rollout_dp_rank * self.rollout_tp_size],
                        soft=False,
                    ),
                    name=f"async_llm_server_{rollout_dp_rank}",
                ).remote(config, self.rollout_dp_size, rollout_dp_rank, self.worker_group.name_prefix)
                for rollout_dp_rank in unready_dp_ranks
            }

            for rollout_dp_rank, server in servers.items():
                try:
                    address = ray.get(server.get_server_address.remote())
                    self.server_addresses[rollout_dp_rank] = address
                    self.async_llm_servers[rollout_dp_rank] = server
                    unready_dp_ranks.remove(rollout_dp_rank)
                except Exception:
                    ray.kill(server)
                    print(f"rollout server {rollout_dp_rank} failed, maybe address already in use, restarting...")

        # All server instances are ready, init AsyncLLM engine.
        ray.get([server.init_engine.remote() for server in self.async_llm_servers])

        # Init user provided chat scheduler in sperate thread.
        self.chat_scheduler: ChatCompletionScheduler = None
        self.chat_scheduler_exception: Exception = None
        self.chat_scheduler_loop = None
        self.chat_scheduler_ready = threading.Event()
        self.chat_scheduler_thread = threading.Thread(
            target=self._init_chat_scheduler, daemon=True, name="chat_scheduler_thread"
        )
        self.chat_scheduler_thread.start()
        self.chat_scheduler_ready.wait()
        self.agent_loop_mode = self.config.get("rollout.agent_loop_mode", True)

    def _init_chat_scheduler(self):
        self.chat_scheduler_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.chat_scheduler_loop)

        try:
            _chat_scheduler_cls = chat_scheduler_class(
                scheduler_str=self.config.rollout.chat_scheduler.name,
            )
            self.chat_scheduler = _chat_scheduler_cls(
                config=self.full_config,
                server_handles=self.async_llm_servers,
            )
        except Exception as e:
            logger.exception(f"chat_scheduler init error: {e}")
            self.chat_scheduler_exception = e
        finally:
            self.chat_scheduler_ready.set()
        self.chat_scheduler_loop.run_forever()

    def wake_up(self):
        """Wake up all vllm instances."""
        logger.debug("wake up async_llm_servers")
        ray.get([server.wake_up.remote() for server in self.async_llm_servers])

    def sleep(self):
        """Sleep all vllm instances."""
        logger.debug("sleep async_llm_servers")
        ray.get([server.sleep.remote() for server in self.async_llm_servers])

    def generate_sequences(self, batch: DataProto, **sampling_params) -> DataProto:
        """Generate multiple sequences in parallel via chat scheduler."""
        assert self.chat_scheduler is not None, "chat scheduler is not initialized."
        self.wake_up()
        future = asyncio.run_coroutine_threadsafe(
            self.chat_scheduler.generate_sequences(batch, **sampling_params), self.chat_scheduler_loop
        )

        result = future.result()
        self.sleep()
        return result

    def reorder_generate_sequences(
        self, data_iter, renew, **sampling_params
    ) -> tuple[bool, DataProto, DataProto, DataProto]:
        assert self.chat_scheduler is not None, "chat scheduler is not initialized."
        self.wake_up()
        assert isinstance(self.chat_scheduler, ReorderSchedulerMixin), "this should mix in ReorderSchedulerMixin"
        future = asyncio.run_coroutine_threadsafe(
            self.chat_scheduler.reorder_generate_sequences(data_iter, renew, **sampling_params),
            self.chat_scheduler_loop,
        )
        result = future.result()
        self.sleep()
        return result


def chat_scheduler_class(scheduler_str: str) -> type[ChatCompletionScheduler]:
    """Get chat scheduler class.
    Args:
        scheduler_str: str, scheduler name, "reorder" is the only one supported.
    Returns:
        Type[ChatCompletionScheduler]: chat scheduler class.
    """
    if scheduler_str == "reorder":
        return ReorderScheduler
    else:
        raise NotImplementedError
