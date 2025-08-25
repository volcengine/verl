# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import os
import time
from copy import deepcopy
from json import JSONDecodeError
from typing import Any, Optional
from uuid import uuid4

import numpy as np
import sglang.srt.entrypoints.engine
import torch
import torch.distributed as dist
from sglang.srt.managers.tokenizer_manager import (
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import (
    assert_pkg_version,
    get_ip,
    get_open_port,
    is_cuda,
    set_prometheus_multiproc_dir,
    set_ulimit,
)
from tensordict import TensorDict
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, ProcessorMixin

from verl import DataProto
from verl.interactions.base import BaseInteraction
from verl.interactions.utils.interaction_registry import initialize_interactions_from_config
from verl.third_party.sglang import parallel_state as sglang_ps
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionCallSchema, OpenAIFunctionParsedSchema, OpenAIFunctionToolCall
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.net_utils import is_ipv6
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_sequence_to_length
from verl.workers.config import RolloutConfig
from verl.workers.rollout.async_server import TokenOutput
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.schemas import (
    AsyncRolloutRequest,
    AsyncRolloutRequestStateEnum,
    FinishReasonTypeEnum,
    Message,
)
from verl.workers.rollout.sglang_rollout.utils import broadcast_pyobj

try:
    from sglang.srt.function_call.function_call_parser import FunctionCallParser
except ImportError:
    from sglang.srt.function_call_parser import FunctionCallParser

try:
    from sglang.srt.entrypoints.openai.protocol import Tool
except ImportError:
    from sglang.srt.openai_api.protocol import Tool


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# patch to avoid issue https://github.com/sgl-project/sglang/issues/6723
def _set_envs_and_config(server_args: ServerArgs):
    # Set global environments
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = str(int(server_args.enable_nccl_nvls))
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "4"
    os.environ["CUDA_MODULE_LOADING"] = "AUTO"

    # Set prometheus env vars
    if server_args.enable_metrics:
        set_prometheus_multiproc_dir()

    # Set ulimit
    set_ulimit()

    # Check flashinfer version
    if server_args.attention_backend == "flashinfer":
        assert_pkg_version(
            "flashinfer_python",
            "0.2.5",
            "Please uninstall the old version and reinstall the latest version by following the instructions at https://docs.flashinfer.ai/installation.html.",
        )
    if is_cuda():
        assert_pkg_version(
            "sgl-kernel",
            "0.1.1",
            "Please reinstall the latest version with `pip install sgl-kernel --force-reinstall`",
        )

    # Set mp start method
    mp.set_start_method("spawn", force=True)


sglang.srt.entrypoints.engine._set_envs_and_config = _set_envs_and_config


# because chatCompletion is an async method, it makes the whole ray actor be an async actor
# which can not call loop.run_until_complete. So we need to make the engine to be an async class
class AsyncEngine(sglang.srt.entrypoints.engine.Engine):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # default to use dummy load format, which need to reload weights in first time
        self._need_reload = True

    async def release_memory_occupation(self, tags: Optional[list[str]] = None):
        """Release GPU occupation temporarily."""
        if tags is None:
            obj = ReleaseMemoryOccupationReqInput()
        else:
            obj = ReleaseMemoryOccupationReqInput(tags=tags)
        return await self.tokenizer_manager.release_memory_occupation(obj, None)

    async def resume_memory_occupation(self, tags: Optional[list[str]] = None):
        """Resume GPU occupation."""
        # because __init__ is a sync method, it can not call the async release_memory_occupation
        # have to move release_memory_occupation from __init__ to here
        # For multi-stage awake, we run release weight and kv_cache when we resume weights for the first time.
        if self._need_reload:
            await self.release_memory_occupation()
            self._need_reload = False

        if tags is None:
            obj = ResumeMemoryOccupationReqInput()
        else:
            obj = ResumeMemoryOccupationReqInput(tags=tags)
        return await self.tokenizer_manager.resume_memory_occupation(obj, None)

    async def update_weights_from_tensor(self, update_weights_request: UpdateWeightsFromTensorReqInput):
        return await self.tokenizer_manager.update_weights_from_tensor(update_weights_request, None)

    async def flush_cache(self):
        return await self.tokenizer_manager.flush_cache()

    async def abort_request(self, rid: str = "", abort_all: bool = False):
        """Abort a specific request or all requests.

        Args:
            rid: The request ID to abort. If empty and abort_all is False, no action is taken.
            abort_all: If True, abort all running requests regardless of rid.
        """
        return self.tokenizer_manager.abort_request(rid=rid, abort_all=abort_all)


# NOTE(sgm): add for verl. We can optimize it by making
#  the dataloader yield List[int] without padding.
def _pre_process_inputs(
    pad_token_id,
    prompt_token_ids: torch.Tensor,
) -> torch.Tensor:
    # remove the left padding in the prompt token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    return prompt_token_ids[non_pad_index:]


def _extract_logprob_from_output(output):
    """
    extract log_prob from single sglang inference output
    """

    def _map_each_response(resp):
        input_token_logprobs = resp["meta_info"]["input_token_logprobs"]
        log_probs, output_token_ids = zip(
            *[(log_prob, token_ids) for log_prob, token_ids, _ in input_token_logprobs[1:]], strict=False
        )
        return torch.tensor(output_token_ids), torch.tensor(log_probs)

    output_token_ids, log_probs = _map_each_response(output)
    return output_token_ids, log_probs


# NOTE(linjunrong): adhoc
def _post_process_outputs(processing_class, output):
    try:
        # This is when processing_class is a processor
        tokenizer = processing_class.tokenizer
    except AttributeError:
        try:
            # This is when processing_class is a tokenizer
            tokenizer = processing_class
        except AttributeError as e:
            raise ValueError(f"Cannot get tokenizer from processing_class {processing_class}") from e

    def _map_each_response(resp):
        output_token_logprobs = resp["meta_info"]["output_token_logprobs"]
        log_probs, output_token_ids = zip(
            *[(log_prob, token_ids) for log_prob, token_ids, _ in output_token_logprobs], strict=True
        )
        return torch.tensor(output_token_ids), torch.tensor(log_probs)

    out_map = map(lambda x: _map_each_response(x), output)
    batched_output_token_ids = []
    batched_logprobs = []
    for output_token_ids, log_probs in out_map:
        batched_output_token_ids.append(output_token_ids)
        batched_logprobs.append(log_probs)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    batched_output_token_ids = pad_sequence(batched_output_token_ids, batch_first=True, padding_value=pad_token_id)
    if len(batched_logprobs) > 0:
        batched_logprobs = pad_sequence(batched_logprobs, batch_first=True, padding_value=pad_token_id)
    return batched_output_token_ids, batched_logprobs


def get_tool_call_parser_type(
    processing_class: PreTrainedTokenizer | PreTrainedTokenizerFast | ProcessorMixin,
) -> str:
    items = FunctionCallParser.ToolCallParserEnum.items()
    for parser_type, parser_cls in items:
        parser = parser_cls()
        try:
            # This is when processing_class is a tokenizer
            tokenizer_vocab = processing_class.get_vocab()
        except AttributeError:
            try:
                # This is when processing_class is a processor
                tokenizer_vocab = processing_class.tokenizer.get_vocab()
            except AttributeError as e:
                raise ValueError(f"Cannot get vocab from processing_class {processing_class}") from e

        if parser.bot_token.strip() in tokenizer_vocab and (
            parser.eot_token == "" or parser.eot_token.strip() in tokenizer_vocab
        ):
            return parser_type
    else:
        raise ValueError(f"No tool call parser found for processing_class {processing_class}")


class SGLangRollout(BaseRollout):
    def __init__(
        self,
        actor_module: str,
        config: RolloutConfig,
        processing_class: PreTrainedTokenizer | PreTrainedTokenizerFast | ProcessorMixin,
        model_hf_config,
        port=None,
        trust_remote_code: bool = False,
        device_mesh: DeviceMesh | None = None,
        **kwargs,
    ):
        """Synchronized SGLang rollout engine.

        Args:
            actor_module: Huggingface model name or path to the model. The
                model should be supported by SGLang.
            config: A DictConfig object containing SGLang-specific operational
                parameters and rollout settings.
                Refer to https://docs.sglang.ai/backend/server_arguments.html
            processing_class: The tokenizer or processor instance compatible with the actor_module.
            model_hf_config: The Hugging Face model's configuration (e.g.,
                `transformers.PretrainedConfig`). It provides architectural
                details and hyperparameters like `max_position_embeddings`,
                used by SGLang for correct model initialization. This is
                the model's inherent design, not SGLang's runtime behavior.
            port: Optional port for multi-node initialization when nnodes > 1.
            trust_remote_code: Whether or not to allow for custom models
                defined on the Hub in their own modeling files.
            device_mesh: Optional `DeviceMesh` object for distributed setup.
            **kwargs: Additional keyword arguments, primarily `train_tp` for
                Megatron Backend integration to initialize hybrid engine
                process groups.
        """
        super().__init__()
        self.config = config
        self._device_mesh_cpu = device_mesh
        os.environ.setdefault("SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK", "true")

        (
            self._tool_schemas,
            self._tool_map,
            self._tool_call_parser_type,
            self._sgl_tools,
            self._function_call_parser,
        ) = self._initialize_tools(config, processing_class)
        self.interaction_map: dict[str, BaseInteraction] = self._initialize_interactions(config)
        # If turn on `free_cache_engine`, SGLang engine's KV cache
        # will be freed after each `generate_sequences` call.
        logger.info(
            f"tool_schemas: {self._tool_schemas}, tool_map: {self._tool_map}, tool_call_parser_type: "
            f"{self._tool_call_parser_type}, sgl_tools: {self._sgl_tools}, function_call_parser: "
            f"{self._function_call_parser}"
        )

        self._init_distributed_env(device_mesh_cpu=device_mesh, **kwargs)

        self._verify_config(model_hf_config=model_hf_config)
        # initialize the inference engine
        self._init_inference_engine(trust_remote_code, actor_module, port)

        self._init_sampling_params(**kwargs)

        self.processing_class = processing_class

        try:
            # This is when processing_class is a tokenizer
            self.pad_token_id = self.processing_class.pad_token_id
        except AttributeError:
            try:
                # This is when processing_class is a processor
                self.pad_token_id = self.processing_class.tokenizer.pad_token_id
            except AttributeError as e:
                raise ValueError(f"Cannot get pad_token_id from processing_class {self.processing_class}") from e

    def _init_distributed_env(self, device_mesh_cpu, **kwargs):
        self._device_mesh_cpu = device_mesh_cpu
        os.environ.setdefault("SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK", "true")
        self.tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert self.tensor_parallel_size <= dist.get_world_size(), (
            "tensor parallel size should be less than or equal to the world size"
        )
        self.train_tp = kwargs.get("train_tp", None)
        if self.train_tp is not None:
            # deployed with megatron
            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            train_tp = kwargs.get("train_tp", None)
            num_tp_per_train_tp = train_tp // self.tensor_parallel_size
            sglang_ps.initialize_parallel_state(
                tensor_model_parallel_size=self.tensor_parallel_size,
                num_tp_per_train_tp=num_tp_per_train_tp,
            )

        tp_size = self.tensor_parallel_size
        world_size = int(os.getenv("WORLD_SIZE", "-1"))

        # init device mesh
        if self._device_mesh_cpu is None:
            device_mesh_kwargs = dict(
                mesh_shape=(world_size // tp_size, tp_size, 1),
                mesh_dim_names=["dp", "tp", "pp"],
            )

            self._device_mesh_cpu = init_device_mesh("cpu", **device_mesh_kwargs)

        self._rank = self._device_mesh_cpu.get_rank()
        self._tp_rank = self._device_mesh_cpu["tp"].get_local_rank()
        self._tp_size = self._device_mesh_cpu["tp"].size()
        if self._rank == 0:
            logger.info(f"_init_distributed_env: :tp_world: {self._tp_size}, global_world: {world_size}")
        # get tp_rank of this process in this tp group
        visible_devices = [None] * self._device_mesh_cpu.size(1)

        torch.distributed.all_gather_object(
            visible_devices, os.environ["CUDA_VISIBLE_DEVICES"], self._device_mesh_cpu.get_group("tp")
        )
        self.visible_devices_set = set(",".join(visible_devices).split(","))
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(sorted(list(self.visible_devices_set)))

    def _verify_config(self, model_hf_config):
        if not self.config.get("max_model_len", None):
            self.config.max_model_len = self.config.prompt_length + self.config.response_length
        assert (
            self.config.max_model_len >= self.config.prompt_length + self.config.response_length
        ), f"""max_model_len should be greater than total sequence length (prompt_length + response_length): 
            {self.config.max_model_len} >= {self.config.prompt_length} + {self.config.response_length}"""
        max_position_embeddings = None
        if hasattr(model_hf_config, "max_position_embeddings"):
            max_position_embeddings = model_hf_config.max_position_embeddings
        elif hasattr(model_hf_config, "llm_config") and hasattr(model_hf_config.llm_config, "max_position_embeddings"):
            max_position_embeddings = model_hf_config.llm_config.max_position_embeddings
        elif hasattr(model_hf_config, "text_config") and hasattr(
            model_hf_config.text_config, "max_position_embeddings"
        ):
            max_position_embeddings = model_hf_config.text_config.max_position_embeddings
        if max_position_embeddings is None:
            raise ValueError("max_position_embeddings not found in model_hf_config")
        rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
        if not rope_scaling_config:
            assert max_position_embeddings >= self.config.prompt_length + self.config.response_length, (
                "model context length should be greater than total sequence length"
            )
        else:
            # handle type where there's a length extend factor
            # see https://qwen.readthedocs.io/en/latest/deployment/vllm.html#extended-context-support
            # for using yarn as an example
            rope_scaling_factor = rope_scaling_config.get("factor", 1.0)

            assert (
                model_hf_config.max_position_embeddings * rope_scaling_factor
                >= self.config.prompt_length + self.config.response_length
            ), (
                f"model context length should be greater than total sequence length, "
                f"got rope_scaling_factor={rope_scaling_factor} and "
                f"max_position_embeddings={model_hf_config.max_position_embeddings}"
            )

        # currently max_assistant_turns stand for max number of tool calls
        if self.config.multi_turn.max_assistant_turns is None:
            self.config.multi_turn.max_assistant_turns = self.config.max_model_len // 3
        if self.config.multi_turn.max_user_turns is None:
            self.config.multi_turn.max_user_turns = self.config.max_model_len // 3

    def _init_inference_engine(self, trust_remote_code, actor_module, port):
        # initialize the inference engine
        nnodes = -(-self._tp_size // len(self.visible_devices_set))
        if nnodes > 1:
            ip = get_ip()
            port = get_open_port() if port is None else port
            [ip, port] = broadcast_pyobj(
                [ip, port],
                rank=self._rank,
                dist_group=self._device_mesh_cpu.get_group("tp"),
                src=self._device_mesh_cpu["tp"].mesh[0].item(),
                force_cpu_device=False,
            )
            dist_init_addr = f"[{ip}]:{port}" if is_ipv6(ip) else f"{ip}:{port}"
        else:
            dist_init_addr = None

        load_format = "dummy" if self.config.load_format.startswith("dummy") else self.config.load_format
        tp_size_per_node = self._tp_size // nnodes
        node_rank = self._tp_rank // tp_size_per_node
        first_rank_in_node = self._tp_rank % tp_size_per_node == 0
        engine_kwargs = self.config.get("engine_kwargs", {}).get("sglang", {}) or {}
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}

        # attention backend will be changed to fa3 if not specified
        attention_backend = engine_kwargs.pop("attention_backend", None)
        max_running_requests = self.config.get("max_num_seqs", None)

        if first_rank_in_node:
            rank = dist.get_rank()
            os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"
            self._engine = AsyncEngine(
                model_path=actor_module,
                dtype=self.config.dtype,
                mem_fraction_static=self.config.gpu_memory_utilization,
                enable_memory_saver=True,
                base_gpu_id=0,
                gpu_id_step=1,
                tp_size=self._tp_size,
                node_rank=node_rank,
                load_format=load_format,
                dist_init_addr=dist_init_addr,
                nnodes=nnodes,
                max_running_requests=max_running_requests,
                trust_remote_code=trust_remote_code,
                # NOTE(linjunrong): add rank to prevent SGLang generate same port inside PortArgs.init_new
                # when random.seed is being set during training
                port=30000 + rank,
                # NOTE(Chenyang): turn on log_level to see the decoding speed of SGLang Engine
                # log_level="INFO"
                # NOTE(Chenyang): turn the following lines to see the input and output of each request
                # log_requests=True,
                # log_requests_level=2,
                # NOTE(Chenyang): turn on max_running_requests to set the max concurrent running requests
                # max_running_requests=1,
                mm_attention_backend="fa3",
                attention_backend=attention_backend if attention_backend is not None else "fa3",
                # In async mode for AgentLoop, SGLang support token in token out to avoid the tokenizer
                # inconsistency issue.
                skip_tokenizer_init=self.config.mode == "async",
                **engine_kwargs,
            )
        else:
            self._engine = None

        self.sharding_manager = None
        self.is_sleep = True

    def _init_sampling_params(self, **kwargs):
        kwargs = dict(
            n=1,
            max_new_tokens=self.config.response_length,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            repetition_penalty=1.0,
        )
        # supporting adding any sampling params from the config file
        for k in self.config.keys():
            if hasattr(SamplingParams(), str(k)) or "stop" in str(k):
                kwargs[k] = self.config.get(k)
        kwargs["n"] = 1  # already repeat in ray_trainer
        self.sampling_params = kwargs

    def _initialize_tools(self, config, processing_class):
        """Initialize tools from configuration.

        Args:
            config: Configuration object containing tool-related settings,
                    specifically `config.multi_turn.tool_config_path`.
            tokenizer: The tokenizer instance used for parsing tool calls from
                       the model's generated text.

        Returns:
            tuple: A tuple containing:
                - tool_schemas (list[dict]): OpenAI-formatted JSON schemas
                  defining each tool's capabilities.
                - tool_map (dict[str, BaseTool]): A dictionary mapping tool
                  names to their executable `BaseTool` objects.
                - tool_call_parser_type (str): The identifier for the specific
                  parser type (e.g., 'json_mode', 'tool_code') used to extract
                  tool calls.
                - sgl_tools (list[sglang.srt.openai_api.protocol.Tool]): Tool
                  definitions optimized for SGLang's internal engine.
                - function_call_parser (sglang.srt.function_call_parser.FunctionCallParser):
                  The active parser instance responsible for extracting
                  structured tool calls from model outputs.
        """
        if config.multi_turn.tool_config_path is None:
            return [], {}, None, [], None

        tools_config_file = config.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tools_config_file)

        logger.info(f"Initialize tools from configuration.: tool_list: {tool_list}")
        tool_schemas = [tool.get_openai_tool_schema().model_dump() for tool in tool_list]
        tool_map = {tool.name: tool for tool in tool_list}
        tool_call_parser_type = get_tool_call_parser_type(processing_class)
        sgl_tools = [Tool.model_validate(tool_schema) for tool_schema in tool_schemas]
        function_call_parser = FunctionCallParser(
            sgl_tools,
            tool_call_parser_type,
        )

        return (
            tool_schemas,
            tool_map,
            tool_call_parser_type,
            sgl_tools,
            function_call_parser,
        )

    def _initialize_interactions(self, config):
        """Initialize interactions from configuration.

        Returns:
            dict[str, BaseInteraction]: A dictionary mapping interaction names to interaction instances.
        """
        if config.multi_turn.interaction_config_path is None:
            return {}

        interaction_config_file = config.multi_turn.interaction_config_path
        interaction_map = initialize_interactions_from_config(interaction_config_file)

        logger.info(f"Initialize interactions from configuration: interaction_map: {list(interaction_map.keys())}")
        return interaction_map

    @GPUMemoryLogger(role="sglang rollout", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generate sequences for a batch of prompts.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        if self.config.multi_turn.enable:
            return self._req_level_generate_sequences(prompts, **kwargs)
        return self._batch_level_generate_sequences(prompts, **kwargs)

    @GPUMemoryLogger(role="sglang rollout", logger=logger)
    @torch.no_grad()
    def _batch_level_generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generates single-turn sequences for a batch of prompts.
        For single-turn generation, all prompts are processed in one request.
        `_batch_level_generate_sequences` involves:
        1.  Extracting and pre-processing prompt token IDs from the input
            `prompts`. This includes handling padding and preparing raw
            token ID lists.
        2.  Preparing inputs for the SGLang engine, including multi-modal
            data if present.
        3.  Invoking the SGLang engine (`self._engine.async_generate`,
            an async coroutine) with the batch of processed inputs and
            specified sampling parameters on the master TP rank.
        4.  Broadcasting the results from the master TP rank to all
            other TP ranks.
        5.  Post-processing the engine's output to format the generated
            token IDs and (if applicable) log probabilities.
        6.  Constructing the final sequences by concatenating original
            prompts with the generated responses.
        7.  Updating attention masks and position IDs to reflect the full
            concatenated sequences.
        8.  If `self.config.free_cache_engine` is true, the SGLang engine's
            KV cache is flushed after generation on the master TP rank.
        Args:
            prompts: A `DataProto` object containing the batch of
              input prompts, including tensor data (like `input_ids`,
              `attention_mask`) and meta-information (like `eos_token_id`,
              `do_sample`).
            **kwargs: Additional keyword arguments that can override the
              default sampling parameters (e.g., `temperature`, `top_p`,
              `max_new_tokens`). These are temporarily applied using
              `update_sampling_params`.
        Returns:
            DataProto: A `DataProto` object containing the batch of
              generated sequences. This includes tensors for `prompts`
              (original input IDs), `responses` (generated token IDs),
              `input_ids` (concatenated prompt and response),
              `attention_mask`, and `position_ids` for the full
              sequences.
        Note that in GRPO, if the prompts are validated, we repeat the prompts for rollout.n times in ray_trainer.
        Thus we do not need to repeat the prompts here and set the sampling parameter n to 1.
        """
        # input ids: (bs, prompt_length), left-padded
        idx = prompts.batch["input_ids"]
        # attention_mask: (bs, seq_length), left-padded
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to generate attention mask for the
        # response based on EOS token position
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        # Extract non-tensor data
        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]).tolist() for i in range(batch_size)],
                dtype=object,
            )

        if "multi_modal_data" in non_tensor_batch:
            sglang_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(
                non_tensor_batch.pop("raw_prompt_ids"),
                non_tensor_batch.pop("multi_modal_data"),
                strict=True,
            ):
                sglang_inputs.append(
                    {
                        "prompt_token_ids": raw_prompt_ids,
                        "multi_modal_data": multi_modal_data,
                        "image_data": (
                            multi_modal_data.get("image", None) if isinstance(multi_modal_data, dict) else None
                        ),
                    }
                )
        else:
            sglang_inputs = [
                {"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            ]

        for input_data in sglang_inputs:
            # Ensure token IDs are lists or numpy arrays
            if not isinstance(input_data["prompt_token_ids"], list | np.ndarray):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}"
                )

            input_data["prompt_token_ids"] = list(input_data["prompt_token_ids"])

        # Extract token IDs and image data for SGLang Engine
        idx_list = [input_data["prompt_token_ids"] for input_data in sglang_inputs]
        image_list = [input_data.get("image_data", None) for input_data in sglang_inputs]

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)

        # Create request-level sampling parameters
        request_sampling_params = self.sampling_params.copy()
        if not do_sample:
            request_sampling_params.update(
                {
                    "n": 1,
                    "presence_penalty": 0.0,
                    "frequency_penalty": 0.0,
                    "repetition_penalty": 1.0,
                    "temperature": 0,
                    "top_p": 1,
                    "top_k": -1,
                    "ignore_eos": False,
                    "min_new_tokens": 0,
                    "max_new_tokens": self.config.response_length,
                    "skip_special_tokens": True,
                    "spaces_between_special_tokens": True,
                }
            )
        elif is_validate:
            request_sampling_params.update(
                {
                    "top_k": self.config.val_kwargs.top_k,
                    "top_p": self.config.val_kwargs.top_p,
                    "temperature": self.config.val_kwargs.temperature,
                    "n": 1,  # if validate, already repeat in ray_trainer
                }
            )

        # Update with any additional kwargs
        request_sampling_params.update(kwargs)

        if self._tp_rank == 0:
            loop = asyncio.get_event_loop()
            output = loop.run_until_complete(
                self._engine.async_generate(
                    prompt=None,  # because we have already convert it to prompt token id
                    sampling_params=request_sampling_params,
                    return_logprob=True,
                    input_ids=idx_list,
                    image_data=image_list,
                )
            )
        else:
            output = None

        # Most naive implementation, can extract tensor and send via gloo if too slow
        dist.barrier()
        [output] = broadcast_pyobj(
            data=[output],
            rank=self._rank,
            dist_group=self._device_mesh_cpu["tp"].get_group(),
            src=self._device_mesh_cpu["tp"].mesh[0].item(),
            force_cpu_device=False,
        )
        out = _post_process_outputs(self.processing_class, output)

        response = out[0].to(idx.device)
        rollout_log_probs = None
        if self.config.calculate_log_probs:
            rollout_log_probs = out[1].to(idx.device)

        if response.shape[1] < self.config.response_length:
            response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
            if self.config.calculate_log_probs:
                rollout_log_probs = pad_sequence_to_length(
                    rollout_log_probs, self.config.response_length, self.pad_token_id
                )

        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        if self.config.calculate_log_probs:
            # we will recompute old log prob with actor
            batch["rollout_log_probs"] = rollout_log_probs

        # free cache engine
        if self._engine is not None and self._tp_rank == 0:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self._engine.flush_cache())

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    async def _async_rollout_a_request(
        self,
        req: AsyncRolloutRequest,
        do_sample: bool = True,
        is_validate: bool = False,
        **kwargs,
    ) -> AsyncRolloutRequest:
        assert self._tp_rank == 0, "only the master process can call this function"
        _req = deepcopy(req)
        finish_reason_type = None
        output = None

        current_turns = 0
        user_turns = 0
        user_turn_rewards = []

        # Create request-level sampling parameters
        request_sampling_params = self.sampling_params.copy()
        if not do_sample:
            request_sampling_params.update(
                {
                    "n": 1,
                    "presence_penalty": 0.0,
                    "frequency_penalty": 0.0,
                    "repetition_penalty": 1.0,
                    "temperature": 0,
                    "top_p": 1,
                    "top_k": -1,
                    "ignore_eos": False,
                    "min_new_tokens": 0,
                    "max_new_tokens": self.config.response_length,
                    "skip_special_tokens": True,
                    "spaces_between_special_tokens": True,
                }
            )
        elif is_validate:
            request_sampling_params.update(
                {
                    "top_k": self.config.val_kwargs.top_k,
                    "top_p": self.config.val_kwargs.top_p,
                    "temperature": self.config.val_kwargs.temperature,
                    "n": 1,  # if validate, already repeat in ray_trainer
                }
            )

        # Update with any additional kwargs
        request_sampling_params.update(kwargs)

        while current_turns < self.config.multi_turn.max_assistant_turns:
            if _req.state == AsyncRolloutRequestStateEnum.PENDING:
                await self._handle_pending_state(_req)
                _req.state = AsyncRolloutRequestStateEnum.RUNNING
            elif _req.state == AsyncRolloutRequestStateEnum.TOOL_CALLING:
                if _req.messages[-1].tool_calls is not None:
                    parsed_tool_calls = _req.messages[-1].tool_calls
                    tool_call_results = await asyncio.gather(
                        *[
                            self._tool_map[tool_call.function.name].execute(
                                _req.request_id,
                                tool_call.function.arguments,
                                **_req.tools_kwargs.get(tool_call.function.name, {}).get("execute_kwargs", {}),
                            )
                            for tool_call in parsed_tool_calls
                        ]
                    )
                    _req.add_tool_response_messages(self.processing_class, [resp for resp, _, _ in tool_call_results])
                    for tool_call, (resp, reward, metrics) in zip(parsed_tool_calls, tool_call_results, strict=True):
                        _req.update_metrics(metrics, tool_call.function.name)
                    if len(_req.input_ids) >= self.config.max_model_len:
                        finish_reason_type = FinishReasonTypeEnum.STOP
                        break
                    _req.state = AsyncRolloutRequestStateEnum.RUNNING
                else:
                    raise ValueError(f"Unexpected tool calling last message state: {_req.messages[-1]}")
            elif _req.state == AsyncRolloutRequestStateEnum.RUNNING:
                # Only continue the conversation if the prompt length is not greater than max_model_len - 1,
                # since SGLang raises an error when max_new_tokens + 1 is greater to max_model_len (the extra
                # token accounts for the EOS token).
                prompt_length = len(_req.get_generation_prompt_ids(self.processing_class))

                if prompt_length + 1 >= self.config.max_model_len:
                    finish_reason_type = FinishReasonTypeEnum.LENGTH
                    break

                # Video support is not implemented yet
                image_data = (
                    _req.multi_modal_data["image"]
                    if _req.multi_modal_data and "image" in _req.multi_modal_data
                    else None
                )
                video_data = (
                    _req.multi_modal_data["video"]
                    if _req.multi_modal_data and "video" in _req.multi_modal_data
                    else None
                )
                if video_data:
                    logger.warning(
                        "video support is not implemented yet, current length of video data is %d", len(video_data)
                    )

                output = await self._handle_engine_call(_req, request_sampling_params, image_data=image_data)
                content = output["text"]
                finish_reason_type = FinishReasonTypeEnum.from_str(output["meta_info"]["finish_reason"]["type"])
                current_turns += 1
                if finish_reason_type == FinishReasonTypeEnum.LENGTH:
                    _req.add_assistant_message(self.processing_class, content)
                    break
                else:
                    if self._function_call_parser and self._function_call_parser.has_tool_call(content):
                        finish_reason_type = FinishReasonTypeEnum.TOOL_CALL
                        _req.state = AsyncRolloutRequestStateEnum.TOOL_CALLING
                        try:
                            normed_content, tool_calls = self._function_call_parser.parse_non_stream(content)
                        except JSONDecodeError:
                            normed_content = content
                            tool_calls = []
                        except AttributeError:
                            normed_content = content
                            tool_calls = []
                        parsed_tool_calls = []
                        for tool_call in tool_calls:
                            function, has_decode_error = OpenAIFunctionCallSchema.from_openai_function_parsed_schema(
                                OpenAIFunctionParsedSchema(
                                    name=tool_call.name,
                                    arguments=tool_call.parameters,
                                )
                            )
                            # Drop the tool call if its arguments has decode error
                            if has_decode_error:
                                continue
                            parsed_tool_calls.append(
                                OpenAIFunctionToolCall(
                                    id=str(tool_call.tool_index),
                                    function=function,
                                )
                            )
                        if len(parsed_tool_calls) > 0:
                            _req.add_assistant_message(
                                self.processing_class, normed_content, tool_calls=parsed_tool_calls
                            )
                        else:
                            _req.add_assistant_message(self.processing_class, content)
                            finish_reason_type = FinishReasonTypeEnum.STOP
                            _req.state = AsyncRolloutRequestStateEnum.COMPLETED
                            break
                    else:
                        _req.add_assistant_message(
                            self.processing_class,
                            content,
                        )
                        if (
                            _req.interaction_kwargs
                            and self.interaction_map
                            and user_turns < self.config.multi_turn.max_user_turns
                            and current_turns < self.config.multi_turn.max_assistant_turns
                        ):
                            _req.state = AsyncRolloutRequestStateEnum.INTERACTING
                        else:
                            break
            elif _req.state == AsyncRolloutRequestStateEnum.INTERACTING:
                user_turns += 1
                messages = [{"role": x.role, "content": x.content} for x in _req.messages]

                # Get interaction by name from interaction_kwargs
                interaction_name = _req.interaction_kwargs.get(
                    "name", "gsm8k"
                )  # Default to gsm8k for backward compatibility
                if interaction_name not in self.interaction_map:
                    raise ValueError(
                        f"Interaction '{interaction_name}' not found in interaction_map. Available interactions: "
                        f"{list(self.interaction_map.keys())}"
                    )

                interaction = self.interaction_map[interaction_name]
                should_terminate_sequence, content, reward, metrics = await interaction.generate_response(
                    _req.request_id, messages, **_req.interaction_kwargs
                )
                user_turn_rewards.append(reward)
                if should_terminate_sequence:
                    finish_reason_type = FinishReasonTypeEnum.STOP
                    _req.state = AsyncRolloutRequestStateEnum.COMPLETED
                    break
                else:
                    _req.add_user_message(self.processing_class, content)
                    if len(_req.input_ids) >= self.config.max_model_len:
                        finish_reason_type = FinishReasonTypeEnum.STOP
                        break
                    else:
                        _req.state = AsyncRolloutRequestStateEnum.RUNNING

        if current_turns >= self.config.multi_turn.max_assistant_turns:
            finish_reason_type = FinishReasonTypeEnum.STOP

        # Calculate the reward for each tool
        async def calc_reward_and_release_fn(name: str, tool: BaseTool):
            reward = await tool.calc_reward(_req.request_id, **_req.tools_kwargs[name].get("calc_reward_kwargs", {}))
            await tool.release(_req.request_id, **_req.tools_kwargs[name].get("release_kwargs", {}))
            return name, reward

        tool_reward_tasks = []
        for name in _req.tools_kwargs.keys():
            tool = self._tool_map[name]
            tool_reward_tasks.append(calc_reward_and_release_fn(name, tool))
        tool_reward_scores = await asyncio.gather(*tool_reward_tasks)
        tool_reward_scores = dict(tool_reward_scores)
        all_rewards = {**tool_reward_scores, **{"user_turn_rewards": user_turn_rewards}}
        _req.finalize(self.processing_class, all_rewards, finish_reason_type)
        if self.config.calculate_log_probs:
            debug_sampling_params = {**self.sampling_params}
            debug_sampling_params["max_new_tokens"] = 0
            output = await self._engine.async_generate(
                prompt=None,
                input_ids=_req.input_ids,
                sampling_params=debug_sampling_params,
                return_logprob=True,
                logprob_start_len=0,
            )
            # len(input_token_logprobs) = len(input_tokens)-1，because logprob of 1st token is None
            _req.output_token_ids, _req.rollout_log_probs = _extract_logprob_from_output(output)
        return _req

    async def _handle_engine_call(
        self, _req: AsyncRolloutRequest, sampling_params: dict, image_data: Optional[list[Any]] = None
    ) -> dict:
        generation_prompt_ids = _req.get_generation_prompt_ids(self.processing_class)
        return await self._handle_engine_generate(generation_prompt_ids, sampling_params, image_data)

    async def _handle_engine_generate(
        self, generation_prompt_ids: list[int], sampling_params: dict, image_data: Optional[list[Any]] = None
    ) -> dict:
        max_new_tokens = min(self.config.response_length, self.config.max_model_len - len(generation_prompt_ids) - 1)

        kwargs = sampling_params.copy()
        kwargs["max_new_tokens"] = max_new_tokens
        kwargs["n"] = 1  # group size is supported in preprocess
        return_logprob = kwargs.pop("logprobs", False)

        output = await self._engine.async_generate(
            input_ids=generation_prompt_ids,
            sampling_params=kwargs,
            return_logprob=return_logprob,
            image_data=image_data,
        )
        return output

    async def _handle_pending_state(self, _req: AsyncRolloutRequest) -> AsyncRolloutRequest:
        if _req.tool_schemas is not None:
            tool_creation_coroutines = []
            for tool_schema in _req.tool_schemas:
                tool = self._tool_map[tool_schema.function.name]
                create_kwargs = _req.tools_kwargs[tool.name].get("create_kwargs", {})
                tool_creation_coroutines.append(tool.create(_req.request_id, **create_kwargs))
            tool_creation_results = await asyncio.gather(*tool_creation_coroutines)
            _req.add_tool_response_messages(
                self.processing_class, [tool_result for _, tool_result in tool_creation_results]
            )
        if _req.interaction_kwargs and self.interaction_map:
            interaction_kwargs = _req.interaction_kwargs
            # Get interaction by name from interaction_kwargs
            interaction_name = interaction_kwargs.get("name", "gsm8k")  # Default to gsm8k for backward compatibility
            if interaction_name not in self.interaction_map:
                raise ValueError(
                    f"Interaction '{interaction_name}' not found in interaction_map. Available interactions: "
                    f"{list(self.interaction_map.keys())}"
                )

            interaction = self.interaction_map[interaction_name]
            await interaction.start_interaction(_req.request_id, **interaction_kwargs)

    @GPUMemoryLogger(role="sglang rollout", logger=logger)
    @torch.no_grad()
    def _req_level_generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generates multi-turn sequences for a batch of prompts.
        For multi-turn generation, each prompt is processed separately via
        `_req_level_generate_sequences` for better tool calling control.
        Note that in multi-turn generation, we repeat the prompts for rollout.n times in ray_trainer.
        Thus we do not need to repeat the prompts here and set the sampling parameter n to 1.
        """
        # Async rollout with tools support
        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        tgt_device = prompts.batch["input_ids"].device

        if self._tp_rank == 0:
            req_list = self._preprocess_prompt_to_async_rollout_requests(
                prompts,
            )

            # distinguish training and validation
            if is_validate:
                # Validation mode: process all requests without abort
                loop = asyncio.get_event_loop()
                output_req_list = loop.run_until_complete(
                    asyncio.gather(
                        *[self._async_rollout_a_request(req, do_sample, is_validate, **kwargs) for req in req_list],
                    )
                )
            else:
                # add progress monitoring and abort function
                total_requests = len(req_list)
                target_completion = int(total_requests * (1 - self.config.get("over_sample_rate", 0.0)))
                # abort when target_completion of requests are completed

                completed_count = 0
                aborted_requests = []
                all_tasks = []

                async def rollout_a_request_with_cancellation_handler(req):
                    try:
                        result = await self._async_rollout_a_request(req, do_sample, is_validate, **kwargs)
                        return result
                    except asyncio.CancelledError:
                        # request is cancelled, return padding
                        logger.info(f"Request {req.request_id} was cancelled, creating padding")
                        aborted_requests.append(req.request_id)
                        return self._create_padding_request(req)

                async def run_with_cancellation():
                    nonlocal all_tasks
                    nonlocal completed_count
                    all_tasks = [
                        asyncio.create_task(rollout_a_request_with_cancellation_handler(req)) for req in req_list
                    ]

                    # Wait for target_completion tasks to complete
                    try:
                        for completed_task in asyncio.as_completed(all_tasks):
                            await completed_task
                            completed_count += 1
                            if completed_count >= target_completion:
                                break
                    finally:
                        # Cancel remaining tasks
                        for t in all_tasks:
                            if not t.done():
                                t.cancel()

                        # Wait for all tasks to finish (including cancelled ones)
                        final_results = await asyncio.gather(*all_tasks, return_exceptions=True)
                        # Abort all requests in SGLang engine
                        await self._engine.abort_request(abort_all=True)
                    return final_results

                loop = asyncio.get_event_loop()
                output_req_list = loop.run_until_complete(run_with_cancellation())

            sorted_output_req_list = sorted(output_req_list, key=lambda x: (x.batch_data_id, x.rollout_offset))
        else:
            sorted_output_req_list = None

        dist.barrier()
        [sorted_output_req_list] = broadcast_pyobj(
            data=[sorted_output_req_list],
            rank=self._rank,
            dist_group=self._device_mesh_cpu["tp"].get_group(),
            src=self._device_mesh_cpu["tp"].mesh[0].item(),
            force_cpu_device=False,
        )
        # Construct the batch data
        prompt_ids, response_ids = [], []
        prompt_attention_mask, response_attention_mask = [], []
        prompt_position_ids, response_position_ids = [], []
        response_loss_mask = []
        messages = []
        reward_scores = []
        multi_modal_inputs = []
        request_ids = []
        if self.config.calculate_log_probs:
            output_logprobs = []
            rollout_output_token_ids = []

        for req in sorted_output_req_list:
            assert req.state == AsyncRolloutRequestStateEnum.COMPLETED, f"Request {req.request_id} is not completed"
            assert (
                req.input_ids.shape[-1]
                == req.attention_mask.shape[-1]
                == req.position_ids.shape[-1]
                == req.loss_mask.shape[-1]
            ), f"""Request {req.request_id} has different length of 
                {req.input_ids.shape[-1]=}, {req.attention_mask.shape[-1]=}, 
                {req.position_ids.shape[-1]=}, {req.loss_mask.shape[-1]=}"""
            error_message_lines = [
                f"""Request {req.request_id} has input_ids length {req.input_ids.shape[-1]}
                    greater than max_model_len {self.config.max_model_len}""",
                f"Decoded input_ids: {self.processing_class.decode(req.input_ids.squeeze(0))}",
                f"Decoded prompt_ids: {self.processing_class.decode(req.prompt_ids.squeeze(0))}",
                f"Decoded response_ids: {self.processing_class.decode(req.response_ids.squeeze(0))}",
                f"Messages: {req.messages}",
                f"Max model length: {req.max_model_len}",
            ]
            error_message = "\n".join(error_message_lines)
            assert req.input_ids.shape[-1] <= self.config.max_model_len, error_message

            prompt_ids.append(req.prompt_ids.to(tgt_device).squeeze(0))
            response_ids.append(req.response_ids.to(tgt_device).squeeze(0))
            if req.response_ids.shape[-1] > self.config.response_length:
                logger.warning(
                    f"""{req.request_id=} has response_ids length {req.response_ids.shape[-1]} 
                    greater than max_response_len {self.config.response_length},\n{req=}"""
                )
            prompt_attention_mask.append(req.prompt_attention_mask.to(tgt_device).squeeze(0))
            response_attention_mask.append(req.response_attention_mask.to(tgt_device).squeeze(0))
            prompt_position_ids.append(req.prompt_position_ids.to(tgt_device).squeeze(0))
            response_position_ids.append(req.response_position_ids.to(tgt_device).squeeze(0))
            response_loss_mask.append(req.response_loss_mask.to(tgt_device).squeeze(0))
            messages.append({"messages": req.messages})
            reward_scores.append(req.reward_scores)
            multi_modal_inputs.append(req.multi_modal_inputs)
            request_ids.append(req.request_id)
            if self.config.calculate_log_probs:
                # extract output log_probs
                output_logprobs.append(req.rollout_log_probs[-len(req.response_ids) :])
                rollout_output_token_ids.append(req.output_token_ids[-len(req.response_ids) :])

        prompt_ids = pad_sequence(
            prompt_ids,
            batch_first=True,
            padding_value=self.pad_token_id,
            padding_side="left",
        )
        if prompt_ids.shape[-1] < self.config.prompt_length:
            prompt_ids = pad_sequence_to_length(prompt_ids, self.config.prompt_length, self.pad_token_id, left_pad=True)
        response_ids = pad_sequence(response_ids, batch_first=True, padding_value=self.pad_token_id)
        if response_ids.shape[-1] < self.config.response_length:
            response_ids = pad_sequence_to_length(response_ids, self.config.response_length, self.pad_token_id)
        prompt_attention_mask = pad_sequence(
            prompt_attention_mask,
            batch_first=True,
            padding_value=0,
            padding_side="left",
        )
        if prompt_attention_mask.shape[-1] < self.config.prompt_length:
            prompt_attention_mask = pad_sequence_to_length(
                prompt_attention_mask, self.config.prompt_length, 0, left_pad=True
            )
        response_attention_mask = pad_sequence(response_attention_mask, batch_first=True, padding_value=0)
        if response_attention_mask.shape[-1] < self.config.response_length:
            response_attention_mask = pad_sequence_to_length(response_attention_mask, self.config.response_length, 0)

        # padding prompt_position_ids
        if prompt_position_ids[0].dim() == 2:
            # if prompt_position_ids is a 2D tensor
            # e.g. from qwen2vl, prompt_position_ids.shape = (3, seq_len)
            transposed_prompt_position_ids = [p.transpose(0, 1) for p in prompt_position_ids]
            prompt_position_ids = pad_sequence(
                transposed_prompt_position_ids, batch_first=True, padding_value=0, padding_side="left"
            )
            prompt_position_ids = prompt_position_ids.transpose(1, 2)
        else:
            prompt_position_ids = pad_sequence(
                prompt_position_ids, batch_first=True, padding_value=0, padding_side="left"
            )
        if prompt_position_ids.shape[-1] < self.config.prompt_length:
            prompt_position_ids = pad_sequence_to_length(
                prompt_position_ids, self.config.prompt_length, 0, left_pad=True
            )

        # padding response_position_ids
        if response_position_ids[0].dim() == 2:
            # if response_position_ids is a 2D tensor
            # e.g. from qwen2vl, response_position_ids.shape = (3, seq_len)
            transposed_response_position_ids = [p.transpose(0, 1) for p in response_position_ids]
            response_position_ids = pad_sequence(
                transposed_response_position_ids, batch_first=True, padding_value=0, padding_side="left"
            )
            response_position_ids = response_position_ids.transpose(1, 2)
        else:
            response_position_ids = pad_sequence(response_position_ids, batch_first=True, padding_value=0)
        if response_position_ids.shape[-1] < self.config.response_length:
            response_position_ids = pad_sequence_to_length(response_position_ids, self.config.response_length, 0)

        response_loss_mask = pad_sequence(response_loss_mask, batch_first=True, padding_value=0)
        if response_loss_mask.shape[1] < self.config.response_length:
            response_loss_mask = pad_sequence_to_length(response_loss_mask, self.config.response_length, 0)
        if self.config.calculate_log_probs:
            output_logprobs = pad_sequence(output_logprobs, padding_value=0.0, batch_first=True)
            output_logprobs = pad_sequence_to_length(
                output_logprobs, pad_token_id=0.0, max_seq_len=response_ids.shape[-1]
            ).to(tgt_device)
            rollout_output_token_ids = pad_sequence(
                rollout_output_token_ids, padding_value=self.pad_token_id, batch_first=True
            )
            rollout_output_token_ids = pad_sequence_to_length(
                rollout_output_token_ids, pad_token_id=self.pad_token_id, max_seq_len=response_ids.shape[-1]
            ).to(tgt_device)

        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)
        position_ids = torch.cat((prompt_position_ids, response_position_ids), dim=-1)

        # Construct the batch data
        batch = TensorDict(
            {
                "prompts": prompt_ids,
                "responses": response_ids,
                "response_mask": response_loss_mask,
                "input_ids": input_ids,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=len(sorted_output_req_list),
        )
        if self.config.calculate_log_probs:
            batch["rollout_log_probs"] = output_logprobs
            batch["rollout_output_token_ids"] = rollout_output_token_ids

        # free cache engine
        if self._engine is not None and self._tp_rank == 0:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self._engine.flush_cache())

        non_tensor_batch = {
            "messages": np.array(messages),
            "reward_scores": np.array(reward_scores),
            "request_id": np.array(request_ids),
        }

        is_multimodal = isinstance(self.processing_class, ProcessorMixin) and (
            hasattr(self.processing_class, "image_processor") or hasattr(self.model_hf_config, "vision_config")
        )

        if is_multimodal:
            non_tensor_batch["multi_modal_inputs"] = np.array(multi_modal_inputs, dtype=object)

        return DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
        )

    def _create_padding_request(self, original_req: AsyncRolloutRequest) -> AsyncRolloutRequest:
        # create a padding request to replace the aborted request
        # the padding request has the following characteristics:
        # 1. state is COMPLETED, but contains empty response
        # 2. response_loss_mask is all 0, ensuring it is ignored in loss calculation
        # 3. keep the original request structure, but the content is empty
        # create padding response_ids (all pad_token_id)
        padding_response_length = self.config.response_length
        device = original_req.input_ids.device if original_req.input_ids is not None else "cpu"
        padding_response_ids = torch.full(
            (1, padding_response_length),
            self.pad_token_id,
            dtype=torch.long,
            device=device,
        )

        # create padding attention_mask (all 0)
        padding_response_attention_mask = torch.zeros(
            (1, padding_response_length),
            dtype=torch.long,
            device=device,
        )

        # create padding position_ids
        if original_req.position_ids is not None:
            first_dim = 1
            # if position_ids is a 2D tensor (e.g. qwen2vl)
            if original_req.position_ids.dim() == 2:
                first_dim = original_req.position_ids.shape[0]
            padding_response_position_ids = torch.zeros(
                (first_dim, padding_response_length),
                dtype=torch.long,
                device=device,
            )
        else:
            padding_response_position_ids = None

        # create padding prompt_attention_mask (all 0)
        padding_prompt_attention_mask = torch.zeros(
            (1, original_req.prompt_attention_mask.shape[-1]),
            dtype=torch.long,
            device=device,
        )

        # create padding loss_mask (all 0, ensuring it is ignored)
        padding_response_loss_mask = torch.zeros(
            (1, padding_response_length),
            dtype=torch.long,
            device=device,
        )

        padding_req = original_req.model_copy(deep=True)
        padding_req.state = AsyncRolloutRequestStateEnum.COMPLETED
        padding_req.response_ids = padding_response_ids
        padding_req.prompt_attention_mask = padding_prompt_attention_mask
        padding_req.response_attention_mask = padding_response_attention_mask
        padding_req.response_position_ids = padding_response_position_ids
        padding_req.response_loss_mask = padding_response_loss_mask
        padding_req.reward_scores = {}
        padding_req.metrics = {}
        padding_req.output_token_ids = None
        padding_req.rollout_log_probs = None
        return padding_req

    def _preprocess_prompt_to_async_rollout_requests(self, prompts: DataProto, n: int = 1) -> list[AsyncRolloutRequest]:
        assert "raw_prompt" in prompts.non_tensor_batch, (
            "need data.return_raw_chat=True, due to no official way do parse_messages"
        )
        logger.info(
            "n is deprecated for SGLang rollout since ray ppo trainer will repeat the prompts for rollout.n times"
        )
        req_list = []
        multi_modal_data_list = prompts.non_tensor_batch.get(
            "multi_modal_data", [None] * len(prompts.non_tensor_batch["raw_prompt"])
        )

        for data_idx, (raw_prompt, multi_modal_data) in enumerate(
            zip(prompts.non_tensor_batch["raw_prompt"], multi_modal_data_list, strict=True)
        ):
            if self._tool_schemas:
                _tools_kwargs = prompts.non_tensor_batch["tools_kwargs"][data_idx]
                _tool_schemas = [self._tool_map[k].get_openai_tool_schema() for k in _tools_kwargs.keys()]
                _input_ids = None
                _attention_mask = None
            else:
                _input_ids = _pre_process_inputs(self.pad_token_id, prompts.batch["input_ids"][data_idx])
                _attention_mask = _pre_process_inputs(0, prompts.batch["attention_mask"][data_idx])
                _tools_kwargs = {}
                _tool_schemas = None

            if self.interaction_map:
                _interaction_kwargs = prompts.non_tensor_batch["interaction_kwargs"][data_idx]
            else:
                _interaction_kwargs = {}

            if not isinstance(raw_prompt, list | np.ndarray):
                raise TypeError(f"raw_prompt must be a list or numpy array, got {type(raw_prompt)}")

            req = AsyncRolloutRequest(
                batch_data_id=data_idx,
                rollout_offset=0,
                request_id=str(uuid4()),
                state=AsyncRolloutRequestStateEnum.PENDING,
                messages=list(raw_prompt),
                multi_modal_data=multi_modal_data,
                tool_schemas=_tool_schemas,
                tools_kwargs=_tools_kwargs,
                interaction_kwargs=_interaction_kwargs,
                input_ids=_input_ids,
                response_ids=None,
                attention_mask=_attention_mask,
                response_attention_mask=None,
                response_position_ids=None,
                response_loss_mask=None,
                reward_scores={},
                max_prompt_len=self.config.prompt_length,
                max_response_len=self.config.response_length,
                max_model_len=min(self.config.max_model_len, self.config.prompt_length + self.config.response_length),
                use_inference_chat_template=self.config.multi_turn.use_inference_chat_template,
                tokenization_sanity_check_mode=self.config.multi_turn.tokenization_sanity_check_mode,
                processing_class=self.processing_class,
            )
            error_message = f"""Request {req.request_id} has mismatched lengths: 
            input_ids={req.input_ids.shape[-1]}, 
            attention_mask={req.attention_mask.shape[-1]}, 
            position_ids={req.position_ids.shape[-1]}, 
            loss_mask={req.loss_mask.shape[-1]}"""
            assert (
                req.input_ids.shape[-1]
                == req.attention_mask.shape[-1]
                == req.position_ids.shape[-1]
                == req.loss_mask.shape[-1]
            ), error_message
            req_list.append(req)

        return req_list

    # ==================== server mode public methods ====================

    async def chat_completion(self, json_request):
        """OpenAI chat completion API."""
        assert self._tp_rank == 0, "only called in tp rank 0"
        _input_ids = None
        _attention_mask = None
        _position_ids = None
        _tool_schemas = []
        _tools_kwargs = {}

        req = AsyncRolloutRequest(
            request_id=str(uuid4()),
            state=AsyncRolloutRequestStateEnum.PENDING,
            messages=[Message.model_validate(msg) for msg in json_request["messages"]],
            tool_schemas=_tool_schemas,
            tools_kwargs=_tools_kwargs,
            input_ids=_input_ids,
            prompt_ids=_input_ids,
            response_ids=None,
            attention_mask=_attention_mask,
            prompt_attention_mask=_attention_mask,
            response_attention_mask=None,
            position_ids=_position_ids,
            prompt_position_ids=_position_ids,
            response_position_ids=None,
            loss_mask=None,
            prompt_loss_mask=None,
            response_loss_mask=None,
            reward_scores={},
            max_prompt_len=self.config.prompt_length,
            max_response_len=self.config.response_length,
            max_model_len=min(self.config.max_model_len, self.config.prompt_length + self.config.response_length),
            use_inference_chat_template=self.config.multi_turn.use_inference_chat_template,
            tokenization_sanity_check_mode=self.config.multi_turn.tokenization_sanity_check_mode,
            processing_class=self.processing_class,
        )

        # json_request already contains sampling_params
        # Filter only valid SamplingParams arguments
        valid_sampling_params = {}
        temp_sampling_params = SamplingParams()  # Create temporary instance to check valid attributes
        for k, v in json_request.items():
            if k not in ["messages", "model", "tools"] and hasattr(temp_sampling_params, k):
                valid_sampling_params[k] = v
        output = await self._handle_engine_call(req, valid_sampling_params)
        # it can be Dict or AsyncIterator[Dict]
        if isinstance(output, dict):
            outputs = [output]
        else:
            outputs = output

        # build openai chat completion format
        choices = []
        id = None
        for i, content in enumerate(outputs):
            choices.append(
                {
                    "index": i,
                    "message": {
                        "role": "assistant",
                        "content": content["text"],
                    },
                    "finish_reason": content["meta_info"]["finish_reason"]["type"],
                }
            )
            id = content["meta_info"]["id"]

        return {
            "id": "chatcmpl-" + id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": json_request.get("model", "sglang_model"),
            "choices": choices,
        }

        # this function is left for uniform train-inference resharding

    async def generate(
        self,
        prompt_ids: torch.Tensor,
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
    ) -> TokenOutput:
        """Generate sequence with token-in-token-out."""
        request_sampling_params = self.sampling_params.copy()
        request_sampling_params.update(sampling_params)
        output = await self._handle_engine_generate(prompt_ids, request_sampling_params, image_data=image_data)
        if sampling_params.get("logprobs", False):
            output_token_logprobs = output["meta_info"]["output_token_logprobs"]
            log_probs, token_ids = zip(
                *[(log_prob, token_ids) for log_prob, token_ids, _ in output_token_logprobs], strict=True
            )
        else:
            token_ids = output["output_ids"]
            log_probs = None
        return TokenOutput(token_ids=token_ids, log_probs=log_probs)

    async def wake_up(self):
        """Load model weights and build kv cache."""
        if not self.is_sleep:
            return
        await self.sharding_manager.wake_up()  # pylint: disable=C2801
        self.is_sleep = False

    async def sleep(self):
        """Offload model weights and discard kv cache."""
        if self.is_sleep:
            return
        await self.sharding_manager.sleep()
        self.is_sleep = True
