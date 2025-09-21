# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import argparse
import asyncio
import logging
import os
import pickle
import threading
from pprint import pprint
from typing import Any, Callable, Optional

import numpy as np
import ray
import vllm.entrypoints.cli.serve
import zmq
from omegaconf import DictConfig
from ray.actor import ActorHandle
from vllm import SamplingParams
from vllm.entrypoints.openai.api_server import (
    build_app,
    build_async_engine_client,
    init_app_state,
)
from vllm.inputs import TokensPrompt
from vllm.outputs import RequestOutput
from vllm.utils import FlexibleArgumentParser
from vllm.v1.executor.abstract import Executor

from verl.single_controller.ray import RayClassWithInitArgs
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.replica import RolloutMode, RolloutReplica, TokenOutput
from verl.workers.rollout.utils import get_free_port, run_unvicorn
from verl.workers.rollout.vllm_rollout import vLLMAsyncRollout

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class ExternalZeroMQDistributedExecutor(Executor):
    """An executor that engines are launched by external ray actors."""

    uses_ray: bool = False

    def _init_executor(self) -> None:
        addresses = os.environ["VERL_VLLM_ZMQ_ADDRESSES"].split(",")
        self.context = zmq.Context()
        self.sockets = []
        for address in addresses:
            socket = self.context.socket(zmq.REQ)
            socket.connect(address)
            self.sockets.append(socket)

        kwargs = dict(
            vllm_config=self.vllm_config,
            local_rank=None,
            rank=None,
            distributed_init_method="env://",
            is_driver_worker=True,
        )
        self.collective_rpc("init_worker", args=([kwargs],))
        self.collective_rpc("init_device")
        self.collective_rpc("load_model")

    def collective_rpc(
        self,
        method: str | Callable,
        timeout: Optional[float] = None,
        args: tuple = (),
        kwargs: Optional[dict[str, Any]] = None,
    ) -> list[Any]:
        if isinstance(method, str):
            sent_method = method
        else:
            sent_method = pickle.dumps(method)
        del method

        message = pickle.dumps((sent_method, args, kwargs or {}))
        for socket in self.sockets:
            socket.send(message, zmq.DONTWAIT)

        outputs = []
        for socket in self.sockets:
            outputs.append(pickle.loads(socket.recv()))
        return outputs

    def check_health(self):
        return


@ray.remote(num_cpus=1)
class vLLMHttpServer:
    """vLLM http server in single node, this is equivalent to launch server with command line:
    ```
    vllm serve --tensor-parallel-size=8 ...
    ```
    """

    def __init__(
        self,
        config: DictConfig,
        rollout_mode: RolloutMode,
        workers: list[ActorHandle],
        replica_rank: int,
        node_rank: int,
        gpus_per_node: int,
        nnodes: int,
    ):
        """
        Args:
            config (DictConfig): full config.
            rollout_mode (RolloutMode): rollout mode.
            replica_rank (int): replica rank, a replica may contain multiple nodes.
            node_rank (int): node rank.
            gpus_per_node (int): number of gpus per node.
            nnodes (int): number of nodes.
        """
        super().__init__()

        self.config: RolloutConfig = omega_conf_to_dataclass(config.actor_rollout_ref.rollout)
        self.model_config: HFModelConfig = omega_conf_to_dataclass(
            config.actor_rollout_ref.model, dataclass_type=HFModelConfig
        )
        self.config.max_model_len = self.config.prompt_length + self.config.response_length
        self.rollout_mode = rollout_mode
        self.workers = workers

        self.replica_rank = replica_rank
        self.node_rank = node_rank
        self.gpus_per_node = gpus_per_node
        self.nnodes = nnodes

        if self.rollout_mode != RolloutMode.HYBRID and self.config.load_format == "dummy":
            logger.warning(f"rollout mode is {self.rollout_mode}, load_format is dummy, set to auto")
            self.config.load_format = "auto"

        # used for http server
        self._server_address = ray.util.get_node_ip_address().strip("[]")
        self._server_port = None

        # used for data parallel: --data-parallel-address, --data-parallel-rpc-port
        if self.node_rank == 0:
            self._master_address = self._server_address
            self._master_port = get_free_port()
        else:
            self._master_address = None
            self._master_port = None

    def get_master_address(self):
        """Get master address and port for data parallel."""
        return self._master_address, self._master_port

    def get_server_address(self):
        """Get http server address and port."""
        assert self._server_port is not None, "http server is not launched, port is None"
        return self._server_address, self._server_port

    async def launch_server(self, master_address: str = None, master_port: int = None):
        if self.node_rank != 0:
            assert master_address and master_port, "non-master node should provide master address and port"
            self._master_address = master_address
            self._master_port = master_port

        # 1. setup vllm serve cli args
        engine_kwargs = self.config.get("engine_kwargs", {}).get("vllm", {}) or {}
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        if self.config.get("limit_images", None):  # support for multi-image data
            engine_kwargs["limit_mm_per_prompt"] = {"image": self.config.get("limit_images")}
        if self.config.cudagraph_capture_sizes:
            engine_kwargs["cuda_graph_sizes"] = self.config.cudagraph_capture_sizes

        args = {
            "dtype": self.config.dtype,
            "load_format": self.config.load_format,
            "skip_tokenizer_init": False,
            "trust_remote_code": True,
            "max_model_len": self.config.max_model_len,
            "max_num_seqs": self.config.max_num_seqs,
            "enable_chunked_prefill": self.config.enable_chunked_prefill,
            "max_num_batched_tokens": self.config.max_num_batched_tokens,
            "enable_prefix_caching": self.config.enable_prefix_caching,
            "enable_sleep_mode": True,
            "disable_custom_all_reduce": True,
            "enforce_eager": self.config.enforce_eager,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "disable_log_stats": self.config.disable_log_stats,
            "tensor_parallel_size": self.config.tensor_model_parallel_size,
            "seed": self.config.get("seed", 0),
            **engine_kwargs,
        }
        if self.config.data_parallel_size > 1:
            args.update(
                {
                    "enable_expert_parallel": False,
                    "data_parallel_size": self.config.data_parallel_size,
                    "data_parallel_size_local": self.gpus_per_node,
                    "data_parallel_start_rank": self.node_rank * self.gpus_per_node,
                    "data_parallel_address": self._master_address,
                    "data_parallel_rpc_port": self._master_port,
                }
            )

        server_args = ["serve", self.model_config.local_path]
        for k, v in args.items():
            if isinstance(v, bool):
                if v:
                    server_args.append(f"--{k}")
            else:
                server_args.append(f"--{k}")
                server_args.append(str(v))

        if self.replica_rank == 0 and self.node_rank == 0:
            pprint(server_args)

        CMD_MODULES = [vllm.entrypoints.cli.serve]
        parser = FlexibleArgumentParser(description="vLLM CLI")
        subparsers = parser.add_subparsers(required=False, dest="subparser")
        cmds = {}
        for cmd_module in CMD_MODULES:
            new_cmds = cmd_module.cmd_init()
            for cmd in new_cmds:
                cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
                cmds[cmd.name] = cmd
        server_args = parser.parse_args(args=server_args)
        server_args.model = server_args.model_tag
        if server_args.subparser in cmds:
            cmds[server_args.subparser].validate(server_args)

        # 2. setup distributed executor backend
        distributed_executor_backend = ExternalZeroMQDistributedExecutor if len(self.workers) > 0 else None
        server_args.distributed_executor_backend = distributed_executor_backend

        zmq_addresses = ray.get([worker.get_zeromq_address.remote() for worker in self.workers])
        logger.info(
            f"replica_rank={self.replica_rank}, node_rank={self.node_rank}, nnodes={self.nnodes}, "
            f"get worker zmq addresses: {zmq_addresses}"
        )
        os.environ["VERL_VLLM_ZMQ_ADDRESSES"] = ",".join(zmq_addresses)

        # 3. launch server
        if self.node_rank == 0:
            await self.run_server(server_args)
        else:
            from vllm.entrypoints.cli.serve import run_headless

            self.server_thread = threading.Thread(target=run_headless, args=(server_args,))
            self.server_thread.start()

    async def run_server(self, args: argparse.Namespace):
        async with build_async_engine_client(args) as engine_client:
            app = build_app(args)

            vllm_config = await engine_client.get_vllm_config()
            await init_app_state(engine_client, vllm_config, app.state, args)
            if self.replica_rank == 0 and self.node_rank == 0:
                logger.info(f"Initializing a V1 LLM engine with config: {vllm_config}")

            self.engine = engine_client
            self._server_port, self._server_task = await run_unvicorn(app, args)

            # prevent context manager from shutting down the engine
            engine_client.shutdown = lambda: None

    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
    ) -> TokenOutput:
        """Generate sequence with token-in-token-out."""
        # TODO(@wuxibin): switch to `/generate` http endpoint once multi-modal support ready.
        max_tokens = self.config.max_model_len - len(prompt_ids)
        sampling_params["logprobs"] = 0 if sampling_params.pop("logprobs", False) else None
        sampling_params.setdefault("repetition_penalty", self.config.get("repetition_penalty", 1.0))
        sampling_params = SamplingParams(max_tokens=max_tokens, **sampling_params)
        prompt_ids = _qwen2_5_vl_dedup_image_tokens(prompt_ids, self.model_config.processor)
        prompt = TokensPrompt(
            prompt_token_ids=prompt_ids, multi_modal_data={"image": image_data} if image_data else None
        )
        generator = self.engine.generate(prompt=prompt, sampling_params=sampling_params, request_id=request_id)

        # Get final response
        final_res: Optional[RequestOutput] = None
        async for output in generator:
            final_res = output
        assert final_res is not None

        token_ids = final_res.outputs[0].token_ids
        log_probs = None
        if sampling_params.logprobs is not None:
            log_probs = [logprobs[token_ids[i]].logprob for i, logprobs in enumerate(final_res.outputs[0].logprobs)]
        return TokenOutput(token_ids=token_ids, log_probs=log_probs)

    async def wake_up(self):
        if self.rollout_mode == RolloutMode.HYBRID:
            # Call all workers to switch between trainer mode and rollout mode.
            await asyncio.gather(*[worker.wake_up.remote() for worker in self.workers])
        elif self.rollout_mode == RolloutMode.COLOCATED:
            # Directly call engine to wake up without sync weights.
            await self.engine.wake_up(tags=["kv_cache", "weights"])
        elif self.rollout_mode == RolloutMode.STANDALONE:
            logger.info("skip wake_up in standalone mode")

    async def sleep(self):
        if self.rollout_mode == RolloutMode.HYBRID:
            await self.engine.reset_prefix_cache()
            await asyncio.gather(*[worker.sleep.remote() for worker in self.workers])
        elif self.rollout_mode == RolloutMode.COLOCATED:
            await self.engine.reset_prefix_cache()
            await self.engine.sleep(level=1)
        elif self.rollout_mode == RolloutMode.STANDALONE:
            logger.info("skip sleep in standalone mode")


_rollout_worker_actor_cls = ray.remote(vLLMAsyncRollout)


class vLLMReplica(RolloutReplica):
    def get_ray_class_with_init_args(self) -> RayClassWithInitArgs:
        """Get rollout worker actor class for colocated and standalone mode."""
        worker_dict_cls = RayClassWithInitArgs(
            cls=_rollout_worker_actor_cls,
            config=self.rollout_config,
            model_config=self.model_config,
            device_mesh=None,
        )
        return worker_dict_cls

    async def launch_servers(self):
        """Launch http server in each node."""
        assert len(self.workers) == self.world_size, (
            f"worker number {len(self.workers)} not equal to world size {self.world_size}"
        )

        # get node_id of all workers
        worker_node_ids = await asyncio.gather(
            *[
                worker.__ray_call__.remote(lambda self: ray.get_runtime_context().get_node_id())
                for worker in self.workers
            ]
        )

        # For non-data parallel case, there's only one server whether it's single or multi nodes.
        nnodes, gpus_per_node = self.nnodes, self.gpus_per_node
        if self.rollout_config.data_parallel_size == 1:
            nnodes = 1
            gpus_per_node = self.world_size

        # create server actor in each node with node affinity
        for node_rank in range(nnodes):
            workers = self.workers[node_rank * gpus_per_node : (node_rank + 1) * gpus_per_node]
            node_id = worker_node_ids[node_rank * gpus_per_node]
            server = vLLMHttpServer.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_id,
                    soft=False,
                ),
                name=f"vllm_server_{self.replica_rank}_{node_rank}",
            ).remote(
                config=self.config,
                rollout_mode=self.rollout_mode,
                workers=workers,
                replica_rank=self.replica_rank,
                node_rank=node_rank,
                gpus_per_node=gpus_per_node,
                nnodes=nnodes,
            )
            self.servers.append(server)

        # launch http server in each node
        master_address, master_port = await self.servers[0].get_master_address.remote()
        await asyncio.gather(
            *[
                server.launch_server.remote(master_address=master_address, master_port=master_port)
                for server in self.servers
            ]
        )

        # get http server address from first server
        server_address, server_port = await self.servers[0].get_server_address.remote()
        self._server_handle = self.servers[0]
        self._server_address = f"{server_address}:{server_port}"


def _qwen2_5_vl_dedup_image_tokens(prompt_ids: list[int], processor):
    """Deduplicate consecutive image tokens in prompt_ids for Qwen2.5-VL, since vLLM will replicate the
    <|image_pad|> token by image_data.

    For example,
    ```
    <|vision_start|><|image_pad|><|image_pad|>...<|image_pad|><|vision_end|>
    =>
    <|vision_start|><|image_pad|><|vision_end|>
    ```
    """
    if processor is not None and "Qwen2VLImageProcessor" in processor.image_processor.__class__.__name__:
        prompt_ids = np.array(prompt_ids)

        # Create a mask where True indicates elements to keep
        mask = np.ones(len(prompt_ids), dtype=bool)

        # Find where the array equals the value
        is_value = prompt_ids == processor.image_token_id

        # Find consecutive duplicates by checking if previous element is also the value
        mask[1:] &= ~(is_value[1:] & is_value[:-1])

        return prompt_ids[mask].tolist()
    else:
        return prompt_ids
