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
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cloudpickle
import ray
from omegaconf import DictConfig
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, ChatCompletionResponse, ErrorResponse
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_models import BaseModelPath, OpenAIServingModels
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.executor.abstract import Executor
from vllm.worker.worker_base import WorkerWrapperBase

from verl.utils.fs import copy_to_local
from verl.workers.rollout.async_server import AsyncServerBase

logger = logging.getLogger(__file__)


class ExternalRayDistributedExecutor(Executor):
    """An executor that engines are launched by external ray actors."""

    uses_ray: bool = False

    def _init_executor(self) -> None:
        assert self.vllm_config.instance_id is not None, "instance_id must be set for external ray actors."

        fields = self.vllm_config.instance_id.split(":")
        assert len(fields) == 4, f"instance_id: {self.vllm_config.instance_id} must be in the format of <namespace>:<wg_prefix>:<vllm_dp_size>:<vllm_dp_rank>."
        namespace, wg_prefix, vllm_dp_size, vllm_dp_rank = fields[0], fields[1], int(fields[2]), int(fields[3])

        # Make sure subprocess in same namespace as parent actor.
        # actor name format: {name_prefix}WorkerDict_{pg_idx}:{local_rank}
        ray.init(namespace=namespace)
        actor_names = [actor_name for actor_name in ray.util.list_named_actors() if actor_name.startswith(f"{wg_prefix}WorkerDict")]

        vllm_tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        assert len(actor_names) == vllm_dp_size * vllm_tp_size, f"instance_id: {self.vllm_config.instance_id} has {len(actor_names)} actors, but vllm_dp_size: {vllm_dp_size} * vllm_tp_size: {vllm_tp_size} = {vllm_dp_size * vllm_tp_size} is expected."

        def get_pg_index_and_local_rank(actor_name) -> Tuple[int, int]:
            fields = actor_name.split(":")
            assert len(fields) == 2, f"invalid actor name: {actor_name}"
            pg_index, local_rank = int(fields[0].split("_")[-1]), int(fields[1])
            return pg_index, local_rank

        # sort actor names by pg_index and local_rank
        actor_names = sorted(actor_names, key=get_pg_index_and_local_rank)
        actor_names = actor_names[vllm_dp_rank * vllm_tp_size : (vllm_dp_rank + 1) * vllm_tp_size]
        self.workers: List[WorkerWrapperBase] = [ray.get_actor(actor_name) for actor_name in actor_names]
        print(f"instance_id: {self.vllm_config.instance_id} initializes with external actors: {actor_names}")

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
        print(f"instance_id: {self.vllm_config.instance_id} initializes finished.")

    def collective_rpc(
        self,
        method: Union[str, Callable],
        timeout: Optional[float] = None,
        args: Tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        # TODO(wuxibin): support ray compiled graph
        if isinstance(method, str):
            sent_method = method
        else:
            sent_method = cloudpickle.dumps(method)
        del method

        # ~3ms overhead per schedule step due to SchedulerOutput/ModelRunnerOutput serialization/deserialization.
        outputs = ray.get([worker.execute_method.remote(sent_method, *args, **(kwargs or {})) for worker in self.workers])
        return outputs

    def check_health(self):
        return


@ray.remote(num_cpus=1)
class AsyncvLLMServer(AsyncServerBase):
    """
    AsyncvLLMServer is a wrapper for AsyncLLM, it uses ExternalRayDistributedExecutor to launch engines
    in hybrid rollout workers, i.e AsyncActorRolloutRefWorker.

    AsyncvLLMServer works as follows:
    1. Start FastAPI server first.
    2. Initialize AsyncLLM with ExternalRayDistributedExecutor.
    3. AsyncLLM spawn EngineCore in subprocess.
    4. EngineCore initialize ExternalRayDistributedExecutor.
    5. ExternalRayDistributedExecutor lookup its corresponding actors by name.
    6. ExternalRayDistributedExecutor init executor: init_worker, init_device, load_model.

    For vLLM AsyncLLM design, see: https://github.com/vllm-project/vllm/pull/9826
    """

    def __init__(self, config: DictConfig, vllm_dp_size: int, vllm_dp_rank: int, wg_prefix: str):
        """
        Args:
            config: DictConfig.
            vllm_dp_size: int, vllm data parallel size.
            vllm_dp_rank: int, vllm data parallel rank.
            wg_prefix: str, worker group prefix, used to lookup actors.
        """
        super().__init__()

        self.config = config.actor_rollout_ref
        self.vllm_dp_size = vllm_dp_size
        self.vllm_dp_rank = vllm_dp_rank
        self.wg_prefix = wg_prefix
        self.engine: AsyncLLM = None

    async def init_engine(self):
        """Init vLLM AsyncLLM engine."""
        config = self.config
        model_path = config.model.path
        model_name = "/".join(model_path.split("/")[-2:])
        local_path = copy_to_local(model_path)
        trust_remote_code = config.model.get("trust_remote_code", False)
        config = config.rollout

        tensor_parallel_size = config.get("tensor_model_parallel_size", 1)
        max_num_batched_tokens = config.get("max_num_batched_tokens", 8192)
        max_model_len = config.max_model_len if config.max_model_len else config.prompt_length + config.response_length
        max_model_len = int(max_model_len)

        # Override default generation config from hugging face model config,
        # user can still override them by passing kwargs in each request.
        kwargs = dict(
            n=1,
            logprobs=0,
            max_new_tokens=config.response_length,
        )
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)
        print(f"override_generation_config: {kwargs}")

        engine_args = AsyncEngineArgs(
            model=local_path,
            enable_sleep_mode=True,
            override_generation_config=kwargs,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend=ExternalRayDistributedExecutor if os.environ.get("VERL_VLLM_USE_RAY_BACKEND", "1") == "1" else None,
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
            seed=self.vllm_dp_rank,
        )

        # init async llm engine
        vllm_config = engine_args.create_engine_config()
        namespace = ray.get_runtime_context().namespace
        vllm_config.instance_id = f"{namespace}:{self.wg_prefix}:{self.vllm_dp_size}:{self.vllm_dp_rank}"
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
            request_logger=RequestLogger(max_log_len=4096),
            chat_template=None,
            chat_template_content_format="auto",
            enable_auto_tools=True,
            tool_parser=config.multi_turn.format,  # hermes, llama3_json, ...
        )

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

    async def wake_up(self):
        await self.engine.wake_up()

    async def sleep(self):
        # TODO: https://github.com/vllm-project/vllm/issues/17103
        await self.engine.reset_prefix_cache()
        await self.engine.sleep()
