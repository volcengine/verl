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
import asyncio
import logging
import os
import random
import socket
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cloudpickle
import fastapi
import ray
import uvicorn
from omegaconf import DictConfig
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, ChatCompletionResponse, ErrorResponse
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_models import BaseModelPath, OpenAIServingModels
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.executor.abstract import Executor
from vllm.worker.worker_base import WorkerWrapperBase

from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray.base import RayWorkerGroup
from verl.utils.fs import copy_to_local
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.workers.sharding_manager import FSDPVLLMShardingManager

logger = logging.getLogger(__file__)


def _get_free_port():
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


class ExternalRayDistributedExecutor(Executor):
    """An executor that engines are launched by external ray actors."""
    uses_ray: bool = False

    def _init_executor(self) -> None:
        assert self.vllm_config.instance_id is not None, \
            "instance_id must be set for external ray actors."

        fields = self.vllm_config.instance_id.split(":")
        assert len(fields) == 4, \
            f"instance_id: {self.vllm_config.instance_id} must be in " \
            f"the format of <namespace>:<wg_prefix>:<vllm_dp_size>:<vllm_dp_rank>."
        namespace, wg_prefix, vllm_dp_size, vllm_dp_rank = fields[0], fields[1], int(fields[2]), int(fields[3])

        # Make sure subprocess in same namespace as parent actor.
        # actor name format: {name_prefix}WorkerDict_{pg_idx}:{local_rank}
        ray.init(namespace=namespace)
        actor_names = [
            actor_name for actor_name in ray.util.list_named_actors() if actor_name.startswith(f"{wg_prefix}WorkerDict")
        ]

        vllm_tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        assert len(actor_names) == vllm_dp_size * vllm_tp_size, \
            f"instance_id: {self.vllm_config.instance_id} has {len(actor_names)} actors, " \
            f"but vllm_dp_size: {vllm_dp_size} * vllm_tp_size: {vllm_tp_size} = " \
            f"{vllm_dp_size * vllm_tp_size} is expected."

        def get_pg_index_and_local_rank(actor_name) -> Tuple[int, int]:
            fields = actor_name.split(":")
            assert len(fields) == 2, f"invalid actor name: {actor_name}"
            pg_index, local_rank = int(fields[0].split("_")[-1]), int(fields[1])
            return pg_index, local_rank

        # sort actor names by pg_index and local_rank
        actor_names = sorted(actor_names, key=get_pg_index_and_local_rank)
        actor_names = actor_names[vllm_dp_rank * vllm_tp_size:(vllm_dp_rank + 1) * vllm_tp_size]
        self.workers: List[WorkerWrapperBase] = [ray.get_actor(actor_name) for actor_name in actor_names]
        print(f"instance_id: {self.vllm_config.instance_id} intializes with external actors: {actor_names}")

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
        print(f"instance_id: {self.vllm_config.instance_id} intializes finished.")

    def collective_rpc(self,
                       method: Union[str, Callable],
                       timeout: Optional[float] = None,
                       args: Tuple = (),
                       kwargs: Optional[Dict[str, Any]] = None) -> List[Any]:
        # TODO(wuxibin): support ray compiled graph
        if isinstance(method, str):
            sent_method = method
        else:
            sent_method = cloudpickle.dumps(method)
        del method

        outputs = ray.get(
            [worker.execute_method.remote(sent_method, *args, **(kwargs or {})) for worker in self.workers])
        return outputs

    def check_health(self):
        return


@ray.remote(num_cpus=1)
class AsyncLLMWorker:
    """
    AsyncLLMWorker is a wrapper for AsyncLLM, it uses ExternalRayDistributedExecutor to launch engines
    in hybrid rollout workers, i.e AsyncActorRolloutRefWorker.
 
    It works as follows:
    1. Initialize AsyncLLM with ExternalRayDistributedExecutor.
    2. AsyncLLM spawn EngineCore in subprocess.
    3. EngineCore initialize ExternalRayDistributedExecutor.
    4. ExternalRayDistributedExecutor lookup its corresponding actors by name.
    5. ExternalRayDistributedExecutor init executor: init_worker, init_device, load_model.
    6. AsyncLLM initialize done, start FastAPI server.

    For vLLM AsyncLLM design, see: https://github.com/vllm-project/vllm/pull/9826
    """

    def __init__(self, config: DictConfig, vllm_dp_size: int, vllm_dp_rank: int, wg_prefix: str):
        """
        Args:
            config: DictConfig, actor_rollout_ref config.
            vllm_dp_size: int, vllm data parallel size.
            vllm_dp_rank: int, vllm data parallel rank.
            wg_prefix: str, worker group prefix, used to lookup actors.
        """
        model_path = config.model.path
        model_name = "/".join(model_path.split("/")[-2:])
        local_path = copy_to_local(model_path)
        trust_remote_code = config.model.get('trust_remote_code', False)
        config = config.rollout

        tensor_parallel_size = config.get('tensor_model_parallel_size', 1)
        max_num_batched_tokens = config.get('max_num_batched_tokens', 8192)
        max_model_len = config.max_model_len if config.max_model_len \
                        else config.prompt_length + config.response_length
        max_model_len = int(max_model_len)

        if max_num_batched_tokens < max_model_len and config.enable_chunked_prefill:
            raise ValueError('Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill')

        engine_args = AsyncEngineArgs(
            model=local_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend=ExternalRayDistributedExecutor,
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            disable_mm_preprocessor_cache=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format="auto",
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=vllm_dp_rank,
        )

        # init async llm engine
        vllm_config = engine_args.create_engine_config()
        namespace = ray.get_runtime_context().namespace
        vllm_config.instance_id = f"{namespace}:{wg_prefix}:{vllm_dp_size}:{vllm_dp_rank}"
        self.engine = AsyncLLM.from_vllm_config(vllm_config)

        # build serving chat
        model_config = self.engine.model_config
        BASE_MODEL_PATHS = [BaseModelPath(name=model_name, model_path=model_path)]
        models = OpenAIServingModels(self.engine, model_config, BASE_MODEL_PATHS)
        self.openai_serving_chat = OpenAIServingChat(
            self.engine,
            model_config,
            models,
            "assistant",
            request_logger=None,
            chat_template=None,
            chat_template_content_format="auto",
        )

        # start FastAPI server
        self.address = ray._private.services.get_node_ip_address()
        self.port = None
        self.server_ready = asyncio.Event()
        asyncio.create_task(self._start_fastapi_server())

    async def chat_completion(self, raw_request: Request):
        """OpenAI-compatible HTTP endpoint.

        API reference: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
        """
        request_json = await raw_request.json()
        request = ChatCompletionRequest(**request_json)
        generator = await self.openai_serving_chat.create_chat_completion(request, raw_request)

        if isinstance(generator, ErrorResponse):
            return JSONResponse(content=generator.model_dump(), status_code=generator.code)
        if request.stream:
            return StreamingResponse(content=generator, media_type="text/event-stream")
        else:
            assert isinstance(generator, ChatCompletionResponse)
            return JSONResponse(content=generator.model_dump())

    async def _start_fastapi_server(self):
        app = fastapi.FastAPI()
        app.router.add_api_route("/v1/chat/completions", self.chat_completion, methods=["POST"])

        # TODO: random sleep to reduce port conflict, retry if port is already in use
        asyncio.sleep(random.uniform(0, 3))
        self.port = _get_free_port()
        config = uvicorn.Config(app, host=["::", "0.0.0.0"], port=self.port)
        server = uvicorn.Server(config)
        self.server_ready.set()
        await server.serve()

    async def get_server_address(self) -> Tuple[str, int]:
        await self.server_ready.wait()
        return f"{self.address}:{self.port}"

    async def wake_up(self):
        await self.engine.wake_up()

    async def sleep(self):
        await self.engine.sleep()


class AsyncActorRolloutRefWorker(ActorRolloutRefWorker):

    def _build_rollout(self, trust_remote_code=False):
        rollout, rollout_sharding_manager = super()._build_rollout(trust_remote_code)

        # NOTE: rollout is not actually initialized here, it's deferred
        # to be initialized by AsyncLLMWorker.

        self.vllm_tp_size = self.config.rollout.tensor_model_parallel_size
        self.vllm_dp_rank = int(os.environ["RANK"]) // self.vllm_tp_size
        self.vllm_tp_rank = int(os.environ["RANK"]) % self.vllm_tp_size

        # used for sleep/wake_up
        rollout.sharding_manager = rollout_sharding_manager

        return rollout, rollout_sharding_manager

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        raise NotImplementedError("AsyncActorRolloutRefWorker does not support generate_sequences")

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    def execute_method(self, method: Union[str, bytes], *args, **kwargs):
        """Called by ExternalRayDistributedExecutor collective_rpc."""
        if self.vllm_tp_rank == 0 and method != "execute_model":
            print(f"[DP={self.vllm_dp_rank},TP={self.vllm_tp_rank}] "
                  f"execute_method: {method if isinstance(method, str) else 'Callable'}")
        return self.rollout.execute_method(method, *args, **kwargs)


class AsyncLLMManager:
    """AsyncLLMManager manage a group of vllm instances, i.e AsyncLLMWorker."""

    def __init__(self, config: DictConfig, worker_group: RayWorkerGroup):
        """Initialize AsyncLLMManager.

        Args:
            config: DictConfig, actor_rollout_ref config.
            worker_group: RayWorkerGroup, worker group of AsyncActorRolloutRefWorker. 
        """
        self.config = config
        self.worker_group = worker_group

        self.rollout_tp_size = self.config.rollout.tensor_model_parallel_size
        self.rollout_dp_size = self.worker_group.world_size // self.rollout_tp_size

        register_center = ray.get_actor(f"{self.worker_group.name_prefix}_register_center")
        workers_info = ray.get(register_center.get_worker_info.remote())
        assert len(workers_info) == self.worker_group.world_size

        # make sure AsyncLLMWorker colocates with its corresponding workers
        self.async_llm_workers = [
            AsyncLLMWorker.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=workers_info[rollout_dp_rank * self.rollout_tp_size],
                    soft=False,
                ),
                name=f"async_llm_worker_{rollout_dp_rank}",
            ).remote(config, self.rollout_dp_size, rollout_dp_rank, self.worker_group.name_prefix)
            for rollout_dp_rank in range(self.rollout_dp_size)
        ]
        self.server_addresses = ray.get([worker.get_server_address.remote() for worker in self.async_llm_workers])

    @property
    def server_address(self):
        """Ruturn FastAPI server addresses of all vllm instances."""
        return self.server_addresses

    async def wake_up(self):
        """Wake up all vllm instances."""
        await asyncio.gather(*[worker.wake_up.remote() for worker in self.async_llm_workers])

    async def sleep(self):
        """Sleep all vllm instances."""
        await asyncio.gather(*[worker.sleep.remote() for worker in self.async_llm_workers])
