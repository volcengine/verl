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
import logging
import os
from typing import Any, Optional

import hydra
import numpy as np
import ray
import yaml
from omegaconf import DictConfig

from recipe.fully_async_policy.vllm_rollout.vllm_async_server import FullyAsyncvLLMReplica
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopManager,
    AgentLoopOutput,
    AgentLoopWorkerBase,
    AsyncLLMServerManager,
    _agent_loop_registry,
    _DummyConfig,
    get_trajectory_info,
)
from verl.protocol import DataProto
from verl.single_controller.ray import RayWorkerGroup
from verl.utils.rollout_trace import rollout_trace_attr
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class FullyAsyncLLMServerManager(AsyncLLMServerManager):
    async def generate_for_partial(self, request_id, prompt_ids, sampling_params, **kwargs_extra) -> TokenOutput:
        """Generate tokens from prompt ids. with partial rollout function"""
        server = self._choose_server(request_id)
        output = await server.generate_for_partial.remote(
            request_id=request_id,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            **kwargs_extra,
        )
        return output


class FullyAsyncAgentLoopOutput(AgentLoopOutput):
    """Agent loop output."""

    is_cancel: bool = False
    """Indicates whether the request was interrupted"""
    log_probs: list[float] = None
    """Response token log probs including LLM generated token, tool response token."""
    param_version_start: int = 0
    """Indicate start parameter version when this response is generated"""
    param_version_end: int = 0
    """Indicate end parameter version when this response is generated, used for partial rollout"""


@ray.remote
class FullyAsyncAgentLoopWorker(AgentLoopWorkerBase):
    def __init__(
        self, config: DictConfig, server_handles: list[ray.actor.ActorHandle], reward_router_address: str = None
    ):
        self.server_manager = FullyAsyncLLMServerManager(config, server_handles)
        super().__init__(config, server_handles, reward_router_address)

    async def generate_sequences_no_post(
        self, batch: DataProto, partial_output_list: Optional[list[AgentLoopOutput]]
    ) -> list[AgentLoopOutput]:
        """Generate sequences from agent loop.

        Args:
            batch (DataProto): Input batch.
            partial_output_list: Optional[List[AgentLoopOutput]]: already rollout result.

        Returns:
            list[FullyAsyncAgentLoopOutput]: List of agent loop outputs, one per sample in the batch.
        """
        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )

        # override sampling params for validation
        if batch.meta_info.get("validate", False):
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["temperature"] = config.val_kwargs.temperature

        # by default, we assume it's a single turn agent
        if "agent_name" not in batch.non_tensor_batch:
            batch.non_tensor_batch["agent_name"] = np.array(["single_turn_agent"] * len(batch), dtype=object)

        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(len(batch))

        trajectory_info = await get_trajectory_info(
            batch.meta_info.get("global_steps", -1), index, batch.meta_info.get("validate", False)
        )

        if not partial_output_list:
            partial_output_list = [None] * len(batch)

        tasks = []
        for i in range(len(batch)):
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
            kwargs["output"] = partial_output_list[i]
            tasks.append(
                asyncio.create_task(self._partial_run_agent_loop(sampling_params, trajectory_info[i], **kwargs))
            )
        return await asyncio.gather(*tasks)

    async def _partial_run_agent_loop(
        self,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        *,
        agent_name: str,
        **kwargs,
    ) -> AgentLoopOutput:
        with rollout_trace_attr(
            step=trajectory["step"],
            sample_index=trajectory["sample_index"],
            rollout_n=trajectory["rollout_n"],
            validate=trajectory["validate"],
            name="agent_loop",
        ):
            assert agent_name in _agent_loop_registry, (
                f"Agent loop {agent_name} not registered, registered agent loops: {_agent_loop_registry.keys()}"
            )

            agent_loop_config = _agent_loop_registry[agent_name]
            agent_loop = hydra.utils.instantiate(
                config=agent_loop_config,
                trainer_config=_DummyConfig(config=self.config),
                server_manager=self.server_manager,
                tokenizer=self.tokenizer,
                processor=self.processor,
            )
            return await agent_loop.run(sampling_params, **kwargs)


class FullyAsyncAgentLoopManager(AgentLoopManager):
    def __init__(self, config: DictConfig, worker_group: RayWorkerGroup = None, rm_wg: RayWorkerGroup = None):
        self.config = config
        self.worker_group = worker_group
        self.reward_model_manager = None
        self.reward_router_address = None
        self.agent_loop_workers_class = FullyAsyncAgentLoopWorker
        self.rollout_replica_class = FullyAsyncvLLMReplica

        self.rm_wg = rm_wg
        self.rollout_replicas = None
        self.server_handles = None
        self.server_addresses = None
        self.agent_loop_workers = None

    @classmethod
    async def create(cls, config: DictConfig, worker_group: RayWorkerGroup = None, rm_wg: RayWorkerGroup = None):
        instance = cls(config, worker_group, rm_wg)
        await instance._async_init()
        return instance

    async def _async_init(self):
        if self.config.reward_model.enable and self.config.reward_model.enable_resource_pool:
            from verl.experimental.reward import RewardModelManager

            self.reward_model_manager = RewardModelManager(self.config.reward_model, self.rm_wg)
            self.reward_router_address = self.reward_model_manager.get_router_address()

        await self._initialize_llm_servers_async()
        self._init_agent_loop_workers()

    async def _initialize_llm_servers_async(self):
        rollout_world_size = self.config.actor_rollout_ref.rollout.tensor_model_parallel_size
        world_size = (
            self.worker_group.world_size
            if self.worker_group
            else self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
        )
        num_replicas = world_size // rollout_world_size

        rollout_config = self.config.actor_rollout_ref.rollout
        model_config = self.config.actor_rollout_ref.model
        self.rollout_replicas = [
            self.rollout_replica_class(
                replica_rank=replica_rank,
                config=rollout_config,
                model_config=model_config,
                gpus_per_node=self.config.trainer.n_gpus_per_node,
            )
            for replica_rank in range(num_replicas)
        ]

        if self.worker_group:
            await asyncio.gather(*[server.init_hybrid(self.worker_group) for server in self.rollout_replicas])
        else:
            await asyncio.gather(*[server.init_standalone() for server in self.rollout_replicas])

        self.server_handles = [server._server_handle for server in self.rollout_replicas]
        self.server_addresses = [server._server_address for server in self.rollout_replicas]

        print(f"AgentLoopManager: {self.server_addresses}")
        # Update Prometheus configuration with server addresses

        if os.getenv("PROMETHEUS_FILE") is not None and os.getenv("PROMETHEUS_PORT") is not None:
            assert not rollout_config.disable_log_stats, "PROMETHEUS need disable_log_stats==False"
            await self._update_prometheus_config()

    async def _update_prometheus_config(self):
        """Update Prometheus configuration file with server addresses and reload on first node."""

        if not self.server_addresses:
            logger.warning("No server addresses available to update Prometheus config")
            return

        prometheus_config_path = str(os.getenv("PROMETHEUS_FILE", "/workdir/tmp/prometheus.yml"))

        try:
            # Read existing Prometheus config or create default one
            prometheus_config = {
                "global": {"scrape_interval": "10s", "evaluation_interval": "10s"},
                "scrape_configs": [
                    {
                        "job_name": "ray",
                        "file_sd_configs": [{"files": ["/tmp/ray/prom_metrics_service_discovery.json"]}],
                    },
                    {"job_name": "vllm", "static_configs": [{"targets": self.server_addresses}]},
                ],
            }

            # Write the configuration to file on all nodes
            @ray.remote(num_cpus=0)
            def write_config_file(config_data, config_path):
                os.makedirs(os.path.dirname(config_path), exist_ok=True)
                with open(config_path, "w") as f:
                    yaml.dump(config_data, f, default_flow_style=False, indent=2)
                return True

            # Get all available nodes and schedule task on each node
            nodes = ray.nodes()
            alive_nodes = [node for node in nodes if node["Alive"]]
            node_count = len(alive_nodes)

            print(f"Found {node_count} alive nodes")
            for i, node in enumerate(alive_nodes):
                print(f"Node {i + 1}: {node['NodeManagerAddress']} ({node['NodeManagerHostname']})")

            # Schedule task on each specific node
            write_tasks = []
            for node in alive_nodes:
                node_ip = node["NodeManagerAddress"]
                task = write_config_file.options(
                    resources={"node:" + node_ip: 0.001}  # Schedule to specific node
                ).remote(prometheus_config, prometheus_config_path)
                write_tasks.append(task)

            await asyncio.gather(*[asyncio.wrap_future(task.future()) for task in write_tasks])

            logger.info(
                f"Updated Prometheus configuration at {prometheus_config_path} "
                f"with {len(self.server_addresses)} VLLM servers"
            )
            logger.info(f"VLLM targets: {self.server_addresses}")

            # Get first node IP and execute reload
            await self._reload_prometheus_on_first_node()

        except Exception as e:
            logger.error(f"Failed to update Prometheus configuration: {e}")

    async def _reload_prometheus_on_first_node(self):
        """尝试在每个节点上重载 Prometheus 配置，失败的节点会被忽略"""

        def get_ray_nodes_info():
            """获取 Ray 集群中所有节点的信息"""
            nodes = ray.nodes()

            alive_nodes = [node for node in nodes if node["Alive"]]
            print(f"Total alive nodes: {len(alive_nodes)}")

            for i, node in enumerate(alive_nodes):
                print(f"Node {i + 1}:")
                print(f"  Node ID: {node['NodeID']}")
                print(f"  IP Address: {node['NodeManagerAddress']}")
                print(f"  Hostname: {node['NodeManagerHostname']}")
                print(f"  Alive: {node['Alive']}")
                print(f"  Resources: {node['Resources']}")
                print(f"  Labels: {node['Labels']}")
                print("---")

        # 使用示例
        get_ray_nodes_info()

        @ray.remote(num_cpus=0)
        def reload_prometheus_on_node():
            import os
            import socket
            import subprocess

            # 获取当前节点的 IP
            hostname = socket.gethostname()
            ip_address = socket.gethostbyname(hostname)
            port = int(os.getenv("PROMETHEUS_PORT", "44398"))

            # 执行 curl 重载命令
            reload_url = f"http://{ip_address}:{port}/-/reload"
            print(f"Reloading Prometheus on node: {reload_url}")

            try:
                result = subprocess.run(["curl", "-X", "POST", reload_url], capture_output=True, text=True, timeout=10)

                return {
                    "success": result.returncode == 0,
                    "ip": ip_address,
                    "hostname": hostname,
                    "url": reload_url,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            except subprocess.TimeoutExpired:
                return {
                    "success": False,
                    "ip": ip_address,
                    "hostname": hostname,
                    "url": reload_url,
                    "error": "Timeout after 10 seconds",
                }
            except Exception as e:
                return {"success": False, "ip": ip_address, "hostname": hostname, "url": reload_url, "error": str(e)}

        # Get all available nodes and schedule task on each node
        nodes = ray.nodes()
        alive_nodes = [node for node in nodes if node["Alive"]]
        node_count = len(alive_nodes)

        print(f"Scheduling reload tasks on {node_count} nodes")

        # Schedule task on each specific node
        reload_tasks = []
        for node in alive_nodes:
            node_ip = node["NodeManagerAddress"]
            task = reload_prometheus_on_node.options(
                resources={"node:" + node_ip: 0.001}  # Schedule to specific node
            ).remote()
            reload_tasks.append(task)

        # 等待所有任务完成
        results = await asyncio.gather(
            *[asyncio.wrap_future(task.future()) for task in reload_tasks],
            return_exceptions=True,  # 确保即使有异常也不会中断其他任务
        )

        # 统计成功和失败的节点
        successful_reloads = []
        failed_reloads = []

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Task exception: {result}")
                failed_reloads.append({"error": str(result)})
                continue

            if result["success"]:
                successful_reloads.append(result)
                logger.info(
                    f"Successfully reloaded Prometheus on node {result['hostname']} "
                    f"({result['ip']}) via {result['url']}"
                )
                if result["stdout"]:
                    logger.info(f"Prometheus reload response: {result['stdout'].strip()}")
            else:
                failed_reloads.append(result)
                logger.warning(
                    f"Failed to reload Prometheus on node {result['hostname']} ({result['ip']}) via {result['url']}"
                )
                if "error" in result:
                    logger.warning(f"Error: {result['error']}")
                if result.get("stderr"):
                    logger.warning(f"Stderr: {result['stderr']}")

        if successful_reloads:
            logger.info(f"Successfully reloaded Prometheus on nodes: {[r['hostname'] for r in successful_reloads]}")

        if failed_reloads:
            logger.warning(
                f"Failed to reload Prometheus on nodes: {[r.get('hostname', 'unknown') for r in failed_reloads]}"
            )

    async def generate_single_sample_async(
        self,
        sample: DataProto,
        partial_output_list: Optional[list[AgentLoopOutput]],
    ) -> list[AgentLoopOutput]:
        """
        Asynchronously process a single sample

        Args:
            sample: Single sample data
            partial_output_list: Optional[List[AgentLoopOutput]]: already rollout result.

        Returns:
            list[AgentLoopOutput]: Processing results
        """
        worker = self._select_best_worker()
        output_future = worker.generate_sequences_no_post.remote(sample, partial_output_list)
        return await asyncio.wrap_future(output_future.future())

    def _select_best_worker(self):
        """Select the best worker, simple round-robin load balancing"""
        if not hasattr(self, "_worker_index"):
            self._worker_index = 0

        worker = self.agent_loop_workers[self._worker_index]
        self._worker_index = (self._worker_index + 1) % len(self.agent_loop_workers)
        return worker

    async def cancel(self):
        await asyncio.gather(*[replica.cancel() for replica in self.rollout_replicas])

    async def resume(self):
        await asyncio.gather(*[replica.resume() for replica in self.rollout_replicas])

    async def wake_up(self):
        await asyncio.gather(*[replica.wake_up() for replica in self.rollout_replicas])

    async def sleep(self):
        await asyncio.gather(*[replica.sleep() for replica in self.rollout_replicas])

    async def reset_prefix_cache(self):
        await asyncio.gather(*[replica.reset_prefix_cache() for replica in self.rollout_replicas])
