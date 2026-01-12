# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
import inspect
import json
import logging
import os
from pprint import pprint
from typing import Any, Optional

import ray
import vllm_omni.entrypoints.cli.serve
from vllm.usage.usage_lib import UsageContext
from vllm_omni import SamplingParams
from vllm_omni.engine.arg_utils import AsyncEngineArgs
from vllm_omni.entrypoints import AsyncOmniDiffusion
from vllm_omni.entrypoints.openai.api_server import build_app, omni_init_app_state
from vllm_omni.inputs import TokensPrompt
from vllm_omni.lora.request import LoRARequest
from vllm_omni.outputs import RequestOutput

from verl.single_controller.ray import RayClassWithInitArgs
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.replica import ImageOutput, RolloutReplica
from verl.workers.rollout.utils import (
    run_unvicorn,
)
from verl.workers.rollout.vllm_rollout import vLLMOmniAsyncRollout
from verl.workers.rollout.vllm_rollout.utils import (
    VLLM_LORA_INT_ID,
    VLLM_LORA_NAME,
    VLLM_LORA_PATH,
    get_vllm_max_lora_rank,
)
from verl.workers.rollout.vllm_rollout.vllm_async_server import (
    ExternalZeroMQDistributedExecutor,
    FlexibleArgumentParser,
    vLLMHttpServer,
)

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class vLLMOmniHttpServer(vLLMHttpServer):
    """vLLM-Omni http server in single node, this is equivalent to launch server with command line:
    ```
    vllm serve --tensor-parallel-size=8 ...
    ```
    """

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

        # Override default generation config from hugging face model config,
        # user can still override them by passing kwargs in each request.
        override_generation_config = dict(
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            repetition_penalty=1.0,
            max_new_tokens=self.config.response_length,
        )
        logger.info(f"override_generation_config: {override_generation_config}")

        logger.info(f"enable_sleep_mode: {self.config.enable_sleep_mode}")
        if not self.config.enable_sleep_mode:
            from verl.utils.device import set_expandable_segments

            set_expandable_segments(True)

        quantization = self.config.quantization

        if quantization is not None:
            raise NotImplementedError("vLLM-Omni server does not support quantization yet.")

        hf_overrides = {}

        args = {
            "dtype": self.config.dtype,
            "load_format": self.config.load_format,
            "skip_tokenizer_init": False,
            "trust_remote_code": self.model_config.trust_remote_code,
            "max_model_len": self.config.max_model_len,
            "max_num_seqs": self.config.max_num_seqs,
            "enable_chunked_prefill": self.config.enable_chunked_prefill,
            "max_num_batched_tokens": self.config.max_num_batched_tokens,
            "enable_prefix_caching": self.config.enable_prefix_caching,
            "enable_sleep_mode": self.config.enable_sleep_mode,
            "logprobs_mode": self.config.logprobs_mode,
            "disable_custom_all_reduce": True,
            "enforce_eager": self.config.enforce_eager,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "disable_log_stats": self.config.disable_log_stats,
            "tensor_parallel_size": self.config.tensor_model_parallel_size,
            "seed": self.config.get("seed", 0),
            "override_generation_config": json.dumps(override_generation_config),
            "quantization": quantization,
            "hf_overrides": hf_overrides,
            **engine_kwargs,
        }

        if self.config.prometheus.enable:
            if self.config.prometheus.served_model_name:
                # Extract model name from path if it's a full path
                served_model_name = self.config.prometheus.served_model_name
                if "/" in served_model_name:
                    # If it's a full path, extract the last part as model name
                    served_model_name = served_model_name.split("/")[-1]
                args["served_model_name"] = served_model_name

        if self.config.expert_parallel_size > 1:
            assert self.gpus_per_node % self.config.tensor_model_parallel_size == 0, (
                "gpus_per_node should be divisible by tensor_model_parallel_size"
            )
            data_parallel_size_local = self.gpus_per_node // self.config.tensor_model_parallel_size
            assert len(self.workers) == data_parallel_size_local * self.config.tensor_model_parallel_size, (
                f"num workers ({len(self.workers)}) should be equal to dp_size_local "
            )
            f"({data_parallel_size_local}) * tp_size ({self.config.tensor_model_parallel_size})"

            args.update(
                {
                    "enable_expert_parallel": self.config.expert_parallel_size > 1,
                    "data_parallel_size": self.config.data_parallel_size,
                    "data_parallel_size_local": data_parallel_size_local,
                    "data_parallel_start_rank": self.node_rank * data_parallel_size_local,
                    "data_parallel_address": self._master_address,
                    "data_parallel_rpc_port": self._master_port,
                }
            )

        # update lora-related args
        if self.model_config.lora_rank > 0:
            args.update(
                {
                    "enable_lora": True,
                    "max_loras": 1,
                    "max_lora_rank": get_vllm_max_lora_rank(self.model_config.lora_rank),
                }
            )

        if self.config.enable_rollout_routing_replay:
            args.update({"enable_return_routed_experts": True})

        server_args = ["serve", self.model_config.local_path]
        for k, v in args.items():
            if isinstance(v, bool):
                if v:
                    server_args.append(f"--{k}")
            elif v is not None:
                server_args.append(f"--{k}")
                # Use json.dumps for dict to ensure valid JSON format
                server_args.append(json.dumps(v) if isinstance(v, dict) else str(v))

        if self.replica_rank == 0:
            pprint(server_args)

        CMD_MODULES = [vllm_omni.entrypoints.cli.serve]
        parser = FlexibleArgumentParser(description="vLLM-Omni CLI")
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
            await self.run_headless(server_args)

    async def run_server(self, args: argparse.Namespace):
        engine_args = AsyncEngineArgs.from_cli_args(args)
        usage_context = UsageContext.OPENAI_API_SERVER
        vllm_config = engine_args.create_engine_config(usage_context=usage_context)
        vllm_config.parallel_config.data_parallel_master_port = self._dp_master_port

        fn_args = set(dict(inspect.signature(AsyncOmniDiffusion.from_vllm_config).parameters).keys())
        kwargs = {}
        if "enable_log_requests" in fn_args:
            kwargs["enable_log_requests"] = engine_args.enable_log_requests
        if "disable_log_stats" in fn_args:
            kwargs["disable_log_stats"] = engine_args.disable_log_stats

        engine_client = AsyncOmniDiffusion(vllm_config=vllm_config, usage_context=usage_context, **kwargs)

        # Don't keep the dummy data in memory
        await engine_client.reset_mm_cache()

        app = build_app(args)
        await omni_init_app_state(engine_client, vllm_config, app.state, args)
        if self.replica_rank == 0 and self.node_rank == 0:
            logger.info(f"Initializing a V1 LLM engine with config: {vllm_config}")

        self.engine = engine_client
        self._server_port, self._server_task = await run_unvicorn(app, args, self._server_address)

    async def run_headless(self, args: argparse.Namespace):
        # Create the EngineConfig.
        raise NotImplementedError("vLLM-Omni headless mode is not implemented yet.")

    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
    ) -> ImageOutput:
        """Generate sequence with token-in-token-out."""
        # Calculate the maximum possible new tokens based on available context space
        # This serves as a safety upper bound
        max_possible_tokens = self.config.max_model_len - len(prompt_ids)
        if max_possible_tokens < 0:
            raise ValueError(
                f"Prompt length ({len(prompt_ids)}) exceeds the model's maximum context length "
                f"({self.config.max_model_len})."
            )

        # Determine max_tokens from sampling_params or use configured response_length as default
        if "max_tokens" in sampling_params:
            max_tokens = sampling_params.pop("max_tokens")
        elif "max_new_tokens" in sampling_params:
            # support sglang-style 'max_new_tokens' param
            max_tokens = sampling_params.pop("max_new_tokens")
        else:
            # Default to a calculation that considers configured lengths
            max_tokens = self.config.response_length + self.config.prompt_length - len(prompt_ids)

        # Clamp max_tokens to the valid range [0, max_possible_tokens]
        max_tokens = max(0, min(max_tokens, max_possible_tokens))

        assert max_tokens <= max_possible_tokens, (
            f"max_tokens {max_tokens} exceeds available context space {max_possible_tokens}"
        )
        sampling_params["logprobs"] = 0 if sampling_params.pop("logprobs", False) else None
        sampling_params.setdefault("repetition_penalty", self.config.get("repetition_penalty", 1.0))
        sampling_params = SamplingParams(max_tokens=max_tokens, **sampling_params)
        multi_modal_data = {}
        if image_data is not None:
            multi_modal_data["image"] = image_data
        if video_data is not None:
            multi_modal_data["video"] = video_data

        prompt = TokensPrompt(prompt_token_ids=prompt_ids, multi_modal_data=multi_modal_data)

        # Add lora request
        lora_request = None
        if self.model_config.lora_rank > 0:
            # Make sure we also check that the lora is already loaded in the engine
            lora_loaded = VLLM_LORA_INT_ID in await self.engine.list_loras()
            if lora_loaded:
                lora_request = LoRARequest(
                    lora_name=VLLM_LORA_NAME, lora_int_id=VLLM_LORA_INT_ID, lora_path=VLLM_LORA_PATH
                )

        generator = self.engine.generate(
            prompt=prompt, sampling_params=sampling_params, request_id=request_id, lora_request=lora_request
        )

        # Get final response
        final_res: Optional[RequestOutput] = None
        async for output in generator:
            final_res = output
        assert final_res is not None

        image = final_res.outputs[0].image
        log_probs = final_res.outputs[0].logprobs

        # Determine stop reason from finish_reason
        finish_reason = final_res.outputs[0].finish_reason
        if finish_reason == "abort":
            stop_reason = "aborted"
        elif finish_reason in ("stop", "length"):
            stop_reason = "completed"
        else:
            stop_reason = finish_reason  # for more stop reason in the future

        return ImageOutput(image=image, log_probs=log_probs, stop_reason=stop_reason)


_rollout_worker_actor_cls = ray.remote(vLLMOmniAsyncRollout)


class vLLMOmniReplica(RolloutReplica):
    def __init__(
        self,
        replica_rank: int,
        config: RolloutConfig,
        model_config: HFModelConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
    ):
        super().__init__(replica_rank, config, model_config, gpus_per_node, is_reward_model)
        self.server_class = ray.remote(vLLMOmniHttpServer)

    def get_ray_class_with_init_args(self) -> RayClassWithInitArgs:
        """Get rollout worker actor class for colocated and standalone mode."""
        worker_dict_cls = RayClassWithInitArgs(
            cls=_rollout_worker_actor_cls,
            config=self.config,
            model_config=self.model_config,
            device_mesh=None,
        )
        return worker_dict_cls
