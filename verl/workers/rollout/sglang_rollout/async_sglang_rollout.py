# Copyright 2023-2024 SGLang Team
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
# ==============================================================================
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
import os
import asyncio
import logging
from json import JSONDecodeError
from uuid import uuid4
from contextlib import contextmanager
from copy import deepcopy
from typing import TYPE_CHECKING, List, Optional
from omegaconf import DictConfig
from tensordict import TensorDict

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from torch.distributed.device_mesh import init_device_mesh
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.workers.rollout.base import BaseRollout
from verl.workers.tool.base_tool import BaseTool
# TODO(yuzhen): get_eos_mask was changed to get_response_mask in commit 8cae42dc29736d0802ded43c5ecf67a809d56bd8, check if it's still correct
from verl.utils.torch_functional import get_response_mask, pad_sequence_to_length, pad_2d_list_to_length
from verl.utils.model import compute_position_id_with_mask
from verl.third_party.sglang import parallel_state as sglang_ps
from verl.workers.rollout.data_model import Message, AsyncRolloutRequest, AsyncRolloutRequestStateEnum, FinishReasonTypeEnum
from verl.workers.tool.data_model import OpenAIFunctionParsedSchema, OpenAIFunctionToolCall
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.openai_api.protocol import Tool
from sglang.srt.function_call_parser import FunctionCallParser
from sglang.srt.server import Engine
from sglang.srt.patch_torch import monkey_patch_torch_reductions
from sglang.srt.model_executor.model_runner import LocalSerializedTensor
from sglang.srt.utils import MultiprocessingSerializer, broadcast_pyobj


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'WARN'))


if TYPE_CHECKING:
    from torch import nn


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


# NOTE(linjunrong): adhoc
def _post_process_outputs(tokenizer, output):

    def _map_each_response(l):
        # output_token_ids = torch.tensor(l['token_ids'])
        log_probs = []
        output_token_ids = []
        for log_prob, token_ids, _ in l["meta_info"]["output_token_logprobs"]:
            log_probs.append(log_prob)
            output_token_ids.append(token_ids)
        log_probs = torch.tensor(log_probs)
        output_token_ids = torch.tensor(output_token_ids)
        return output_token_ids, log_probs

    out_map = map(lambda x: _map_each_response(x), output)
    batched_output_token_ids = []
    batched_logprobs = []
    for output_token_ids, log_probs in out_map:
        batched_output_token_ids.append(output_token_ids)
        batched_logprobs.append(log_probs)
    pad_token_id = (tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    batched_output_token_ids = pad_sequence(batched_output_token_ids, batch_first=True, padding_value=pad_token_id)
    if len(batched_logprobs) > 0:
        batched_logprobs = pad_sequence(batched_logprobs, batch_first=True, padding_value=pad_token_id)
    return batched_output_token_ids, batched_logprobs


def get_tool_call_parser_type(tokenizer: PreTrainedTokenizer) -> str:
    for parser_type, parser_cls in FunctionCallParser.ToolCallParserEnum.items():
        parser = parser_cls()
        if parser.bot_token in tokenizer.get_vocab() and (parser.eot_token == '' or parser.eot_token in tokenizer.get_vocab()):
            return parser_type
    else:
        raise ValueError(f"No tool call parser found for tokenizer {tokenizer}")


class AsyncSGLangRollout(BaseRollout):

    def __init__(
        self,
        actor_module: nn.Module | str,
        config: DictConfig,
        tokenizer,
        model_hf_config,
        **kwargs,
    ):
        """A SGLang rollout. It requires the module is supported by the SGLang.

        Args:
            actor_module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in SGLang
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        # currently max_turns stand for max number of tool calls
        if self.config.multi_turn.max_turns is None:
            self.max_turns = self.config.max_model_len // 3 
        else:
            self.max_turns = self.config.multi_turn.max_turns

        tool_list = None
        if config.multi_turn.tool_config_path is not None:
            from omegaconf import OmegaConf
            def initialize_tools(tool_config) -> List:
                import sys
                import importlib.util
                from typing import List
                from verl.workers.tool.data_model import OpenAIFunctionToolSchema

                tool_list = []
                
                for tool_config in tool_config.tools:
                    cls_name = tool_config.class_name
                    module_name, class_name = cls_name.rsplit(".", 1)
                    
                    if module_name not in sys.modules:
                        spec = importlib.util.find_spec(module_name)
                        module = importlib.util.module_from_spec(spec)
                        sys.modules[module_name] = module
                        spec.loader.exec_module(module)
                    else:
                        module = sys.modules[module_name]
                        
                    tool_cls = getattr(module, class_name)
                    
                    tool_schema_dict = OmegaConf.to_container(tool_config.tool_schema, resolve=True)
                    tool_schema = OpenAIFunctionToolSchema.parse_obj(tool_schema_dict)
                    
                    tool = tool_cls(
                        config=OmegaConf.to_container(tool_config.config, resolve=True),
                        tool_schema=tool_schema
                    )
                    tool_list.append(tool)
                
                return tool_list
 
            tool_config_path = config.multi_turn.tool_config_path
            tool_config = OmegaConf.load(tool_config_path)
            tool_list = initialize_tools(tool_config)           

        if tool_list is not None:
            self._tool_schemas = [tool.get_openai_tool_schema().model_dump() for tool in tool_list]
            self._tool_map = {tool.name: tool for tool in tool_list}
            self._tool_call_parser_type = get_tool_call_parser_type(tokenizer)
            self._sgl_tools = [Tool.model_validate(tool_schema) for tool_schema in self._tool_schemas]
            # print(f"{self._sgl_tools=}\n{type(self._sgl_tools[0])=}")
            self._function_call_parser = FunctionCallParser(
                    self._sgl_tools, 
                    self._tool_call_parser_type,
                )
        else:
            self._tool_schemas = []
            self._tool_map = {}
            self._tool_call_parser_type = None
            self._sgl_tools = []
            self._function_call_parser = None
        assert not (not config.enforce_eager and
                    config.free_cache_engine), "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert (tensor_parallel_size <= dist.get_world_size()
               ), "tensor parallel size should be less than or equal to the world size"

        if kwargs.get("train_tp", None) is not None:
            # deployed with megatron
            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            train_tp = kwargs.get("train_tp", None)
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            sglang_ps.initialize_parallel_state(
                tensor_model_parallel_size=tensor_parallel_size,
                num_tp_per_train_tp=num_tp_per_train_tp,
            )

        if not self.config.get("max_model_len", None):
            self.config.max_model_len = self.config.prompt_length + self.config.response_length
        assert self.config.max_model_len >= self.config.prompt_length + self.config.response_length, \
            f"max_model_len should be greater than total sequence length (prompt_length + response_length): {self.config.max_model_len} >= {self.config.prompt_length} + {self.config.response_length}"
        assert (model_hf_config.max_position_embeddings >= self.config.max_model_len), \
            "model context length should be greater than total sequence length"

        tp_size = tensor_parallel_size
        world_size = int(os.getenv("WORLD_SIZE", "-1"))

        # init device mesh
        device_mesh_kwargs = dict(
            mesh_shape=(world_size // tp_size, tp_size, 1),
            mesh_dim_names=["dp", "tp", "pp"],
        )
        device_mesh_cpu = init_device_mesh("cpu", **device_mesh_kwargs)
        # device_mesh_device = init_device_mesh("cuda", **device_mesh_kwargs)

        # get tp_rank of this process in this tp group
        visible_devices = [None] * device_mesh_cpu.size(1)
        dist.all_gather_object(visible_devices, os.environ["CUDA_VISIBLE_DEVICES"],
                                            device_mesh_cpu.get_group("tp"))
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(visible_devices)

        # initialize the inference engine
        monkey_patch_torch_reductions()
        nnodes = -(-tp_size // len(visible_devices))
        self._device_mesh_cpu = device_mesh_cpu
        self._tp_rank = device_mesh_cpu["tp"].get_local_rank()
        self._tp_size = device_mesh_cpu["tp"].size()
        tp_size_per_node = self._tp_size // nnodes
        node_rank = self._tp_rank // tp_size_per_node
        first_rank_in_node = self._tp_rank % tp_size_per_node == 0

        if first_rank_in_node:
            os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"
            self._engine = Engine(
                model_path=actor_module,
                dtype=config.dtype,
                mem_fraction_static=config.gpu_memory_utilization,
                enable_memory_saver=True,
                base_gpu_id=0,
                gpu_id_step=1,
                tp_size=self._tp_size,
                node_rank=node_rank,
                nnodes=nnodes,
            )
        else:
            self._engine = None
        
        # offload
        if self._tp_rank == 0:
            self._engine.release_memory_occupation()

        kwargs = dict(n=1,
                      max_new_tokens=config.response_length,
                      presence_penalty=0.0,
                      frequency_penalty=0.0,
                      repetition_penalty=1.0)
        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)
        print(f"kwargs: {kwargs}")
        self.sampling_params = kwargs

        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if key in self.sampling_params:
                    old_value = self.sampling_params[key]
                    old_sampling_params_args[key] = old_value
                    self.sampling_params[key] = value
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            self.sampling_params[key] = value

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # if self.config.free_cache_engine:
        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)
        idx_list = []
        # parse idx from torch.Tensor to List[List[str]]
        for i in range(batch_size):
            idx_list.append(_pre_process_inputs(self.pad_token_id, idx[i]))

        do_sample = prompts.meta_info.get("do_sample", True)
        if not do_sample:
            # kwargs = {
            #     'top_p': 1.0,
            #     'top_k': -1,
            #     'min_p': 0.0,
            #     'temperature': 0,
            #     'n': 1  # if greedy, only 1 response
            # }
            kwargs = dict(
                n=1,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                repetition_penalty=1.0,
                temperature=0,
                top_p=1,
                top_k=-1,
                ignore_eos=False,
                min_new_tokens=0,
                max_new_tokens=self.config.response_length,
                skip_special_tokens=True,
                spaces_between_special_tokens=True,
            )
        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            print(f"{self.sampling_params=}")
            if self._tp_rank == 0:
                loop = asyncio.get_event_loop()
                output = loop.run_until_complete(self._engine.async_generate(
                    prompt=None,  # because we have already convert it to prompt token id
                    sampling_params=self.sampling_params,
                    return_logprob=True,
                    input_ids=idx_list,
                ))
            else:
                output = None
            # Most naive implementation, can extract tensor and send via gloo if too slow
            [output] = broadcast_pyobj(
                data=[output],
                rank=self._tp_rank,
                dist_group=self._device_mesh_cpu["tp"].get_group(),
                src=self._device_mesh_cpu["tp"].mesh[0].item(),
            )
        out = _post_process_outputs(self.tokenizer, output)

        response = out[0].to(idx.device)
        log_probs = out[1].to(idx.device)

        if response.shape[1] < self.config.response_length:
            response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
            log_probs = pad_sequence_to_length(log_probs, self.config.response_length, self.pad_token_id)
        if self.config.n > 1 and do_sample:
            idx = idx.repeat_interleave(self.config.n, dim=0)
            attention_mask = attention_mask.repeat_interleave(self.config.n, dim=0)
            position_ids = position_ids.repeat_interleave(self.config.n, dim=0)
            batch_size = batch_size * self.config.n
        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )

        # free cache engine
        if self.config.free_cache_engine and self._engine is not None:
            self._engine.tokenizer_manager.flush_cache()

        return DataProto(batch=batch)
    
    async def _async_rollout_a_request(self, req: AsyncRolloutRequest, do_sample: bool = True, is_validate: bool = False, **kwargs) -> AsyncRolloutRequest:
        assert self._tp_rank == 0, "only the master process can call this function"
        _req = deepcopy(req)
        finish_reason_type = None
        output = None
        
        current_turns = 0
        while current_turns < self.max_turns:
            if _req.state == AsyncRolloutRequestStateEnum.PENDING:
                if _req.tools is not None:
                    tool_creation_coroutines = []
                    for tool_schema in _req.tools:
                        tool = self._tool_map[tool_schema.function.name]
                        create_kwargs = _req.tools_kwargs[tool.name].get("create_kwargs", {})
                        tool_creation_coroutines.append(tool.create(_req.request_id, **create_kwargs))
                    await asyncio.gather(*tool_creation_coroutines)
                _req.state = AsyncRolloutRequestStateEnum.RUNNING
            elif _req.state == AsyncRolloutRequestStateEnum.TOOL_CALLING:
                if _req.messages[-1].tool_calls is not None:
                    parsed_tool_calls = _req.messages[-1].tool_calls
                    tool_call_results = await asyncio.gather(*[
                        self._tool_map[tool_call.function.name].execute(
                            _req.request_id, 
                            tool_call.function.arguments,
                            **_req.tools_kwargs[tool_call.function.name].get("execute_kwargs", {})
                        ) for tool_call in parsed_tool_calls
                    ])
                    for tool_call, (resp, reward, metrics) in zip(parsed_tool_calls, tool_call_results):
                        _req.add_tool_response_message(self.tokenizer, resp)
                        if len(_req.input_ids) >= self.config.max_model_len:
                            break
                    if len(_req.input_ids) >= self.config.max_model_len:
                        finish_reason_type = FinishReasonTypeEnum.STOP
                        break
                    _req.state = AsyncRolloutRequestStateEnum.RUNNING
                else:
                    raise ValueError(f"Unexpected tool calling last message state: {_req.messages[-1]}")
            elif _req.state == AsyncRolloutRequestStateEnum.RUNNING:
                generation_prompt = _req.get_generation_prompt(self.tokenizer)
                if not do_sample:
                    kwargs = dict(
                        n=1,
                        presence_penalty=0.0,
                        frequency_penalty=0.0,
                        repetition_penalty=1.0,
                        temperature=0,
                        top_p=1,
                        top_k=-1,
                        ignore_eos=False,
                        min_new_tokens=0,
                        max_new_tokens=self.config.response_length,
                        skip_special_tokens=True,
                        spaces_between_special_tokens=True,
                    )
                elif is_validate:
                    # TODO: try **
                    kwargs = {
                        'top_k': self.config.val_kwargs.top_k,
                        'top_p': self.config.val_kwargs.top_p,
                        'temperature': self.config.val_kwargs.temperature,
                        'n': 1,  # if validate, already repeat in ray_trainer
                    }
                if 'n' not in kwargs or kwargs['n'] > 1:  # group size is supported in preprocess
                    kwargs["n"] = 1
                # users can customize different sampling_params at different run
                with self.update_sampling_params(**kwargs):
                    output = await self._engine.async_generate(
                        prompt=generation_prompt,
                        sampling_params=self.sampling_params,
                        return_logprob=False,
                    )
                    
                content = output["text"]
                finish_reason_type = FinishReasonTypeEnum.from_str(output["meta_info"]["finish_reason"]["type"])
                current_turns += 1
                if finish_reason_type == FinishReasonTypeEnum.LENGTH:
                    _req.add_assistant_message(self.tokenizer, content, alreadyover_long=True)
                    break
                else:
                    if self._function_call_parser and self._function_call_parser.has_tool_call(content):
                        finish_reason_type = FinishReasonTypeEnum.TOOL_CALL
                        _req.state = AsyncRolloutRequestStateEnum.TOOL_CALLING
                        try:
                            normed_content, tool_calls = self._function_call_parser.parse_non_stream(content)
                        except JSONDecodeError as e:
                            logger.warning(f"Failed to parse tool calls from content: {content}, JSONDecodeError: {e}")
                            normed_content = content
                            tool_calls = []
                            # raise e
                        except AttributeError as e:
                            logger.warning(f"Failed to parse tool calls from content: {content}, AttributeError: {e}")
                            normed_content = content
                            tool_calls = []
                            # raise e
                        # print(f"parsed {tool_calls=}")
                        parsed_tool_calls = [
                            OpenAIFunctionToolCall(
                                id=str(tool_call.tool_index), 
                                function=OpenAIFunctionParsedSchema(name=tool_call.name, arguments=tool_call.parameters)
                            ) for tool_call in tool_calls
                        ]
                        if len(parsed_tool_calls) > 0:
                            _req.add_assistant_message(self.tokenizer, normed_content, tool_calls=parsed_tool_calls)
                        else:
                            _req.add_assistant_message(self.tokenizer, content)
                            finish_reason_type = FinishReasonTypeEnum.STOP
                            _req.state = AsyncRolloutRequestStateEnum.COMPLETED
                            break
                    else:
                        _req.add_assistant_message(self.tokenizer, content)
                        break

        if current_turns >= self.max_turns:
            finish_reason_type = FinishReasonTypeEnum.STOP
        # Calculate the reward for each tool
        async def calc_reward_and_release_fn(name: str, tool: BaseTool):
            reward = await tool.calc_reward(_req.request_id, **_req.tools_kwargs[name].get("calc_reward_kwargs", {}))
            await tool.release(_req.request_id, **_req.tools_kwargs[name].get("release_kwargs", {}))
            return name, reward

        tool_reward_tasks = [
            calc_reward_and_release_fn(name, tool)
            for name, tool in self._tool_map.items()
        ]
        tool_reward_scores = await asyncio.gather(*tool_reward_tasks)
        tool_reward_scores = dict(tool_reward_scores)
        _req.finalize(self.tokenizer, tool_reward_scores, finish_reason_type)

        return _req
    
    @torch.no_grad()
    def generate_sequences_with_tools(self,  prompts: DataProto, **kwargs) -> DataProto:
        # Async rollout with tools support
        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if self._tp_rank == 0:
            req_list = self._preprocess_prompt_to_async_rollout_requests(
                    prompts, 
                    n=1 if is_validate else self.config.n,
                )
            loop = asyncio.get_event_loop()
            output_req_list =  loop.run_until_complete(
                asyncio.gather(
                    *[self._async_rollout_a_request(
                        req, 
                        do_sample, 
                        is_validate, 
                        **kwargs) 
                    for req in req_list],
                )
            )
            sorted_output_req_list = sorted(output_req_list, key=lambda x: (x.batch_data_id, x.rollout_offset))
        else:
            sorted_output_req_list = None

        [sorted_output_req_list] = broadcast_pyobj(
            data=[sorted_output_req_list],
            rank=self._tp_rank,
            dist_group=self._device_mesh_cpu["tp"].get_group(),
            src=self._device_mesh_cpu["tp"].mesh[0].item(),
        )
        # Construct the batch data
        prompt_ids, response_ids = [], []
        prompt_attention_mask, response_attention_mask = [], []
        prompt_position_ids, response_position_ids = [], []
        prompt_loss_mask, response_loss_mask = [], []
        messages = []
        # reward_scores = {tool_name: [] for tool_name in self._tool_map.keys()}
        reward_scores = []
        for req in sorted_output_req_list:
            assert req.state == AsyncRolloutRequestStateEnum.COMPLETED, f"Request {req.request_id} is not completed"
            assert len(req.input_ids) == len(req.attention_mask) == len(req.position_ids) == len(req.loss_mask), \
                f"Request {req.request_id} has different length of {len(req.input_ids)=}, {len(req.attention_mask)=}, {len(req.position_ids)=}, {len(req.loss_mask)=}"
            assert len(req.input_ids) <= self.config.max_model_len, \
                f"Request {req.request_id} has input_ids length {len(req.input_ids)} greater than max_model_len {self.config.max_model_len},\n{self.tokenizer.decode(req.input_ids)=},\n{self.tokenizer.decode(req.prompt_ids)=},\n{self.tokenizer.decode(req.response_ids)=},\n{req.messages=},\n{req.max_model_len=}"
            prompt_ids.append(torch.tensor(req.prompt_ids, dtype=torch.int))
            response_ids.append(torch.tensor(req.response_ids, dtype=torch.int))
            if len(req.response_ids) > self.config.response_length:
                print(f"{req.request_id=} has response_ids length {len(req.response_ids)} greater than max_response_len {self.config.response_length},\n{req=}")
            prompt_attention_mask.append(torch.tensor(req.prompt_attention_mask, dtype=torch.int))
            response_attention_mask.append(torch.tensor(req.response_attention_mask, dtype=torch.int))
            prompt_position_ids.append(torch.tensor(req.prompt_position_ids, dtype=torch.int))
            response_position_ids.append(torch.tensor(req.response_position_ids, dtype=torch.int))
            prompt_loss_mask.append(torch.tensor(req.prompt_loss_mask, dtype=torch.int))
            response_loss_mask.append(torch.tensor(req.response_loss_mask, dtype=torch.int))
            messages.append({"messages": req.messages})
            reward_scores.append(req.reward_scores)
            # for tool_name, score in req.reward_scores.items():
            #     reward_scores[tool_name].append(score)

        prompt_ids = pad_sequence(prompt_ids, batch_first=True, padding_value=self.pad_token_id)
        if prompt_ids.shape[1] < self.config.prompt_length:
            prompt_ids = pad_sequence_to_length(prompt_ids, self.config.prompt_length, self.pad_token_id, left_pad=True)
        response_ids = pad_sequence(response_ids, batch_first=True, padding_value=self.pad_token_id)
        if response_ids.shape[1] < self.config.response_length:
            response_ids = pad_sequence_to_length(response_ids, self.config.response_length, self.pad_token_id)
        prompt_attention_mask = pad_sequence(prompt_attention_mask, batch_first=True, padding_value=0)
        if prompt_attention_mask.shape[1] < self.config.prompt_length:
            prompt_attention_mask = pad_sequence_to_length(prompt_attention_mask, self.config.prompt_length, 0, left_pad=True)
        response_attention_mask = pad_sequence(response_attention_mask, batch_first=True, padding_value=0)
        if response_attention_mask.shape[1] < self.config.response_length:
            response_attention_mask = pad_sequence_to_length(response_attention_mask, self.config.response_length, 0)
        prompt_position_ids = pad_sequence(prompt_position_ids, batch_first=True, padding_value=0)
        if prompt_position_ids.shape[1] < self.config.prompt_length:
            prompt_position_ids = pad_sequence_to_length(prompt_position_ids, self.config.prompt_length, 0, left_pad=True)
        response_position_ids = pad_sequence(response_position_ids, batch_first=True, padding_value=0)
        if response_position_ids.shape[1] < self.config.response_length:
            response_position_ids = pad_sequence_to_length(response_position_ids, self.config.response_length, 0)
        prompt_loss_mask = pad_sequence(prompt_loss_mask, batch_first=True, padding_value=0)
        if prompt_loss_mask.shape[1] < self.config.prompt_length:
            prompt_loss_mask = pad_sequence_to_length(prompt_loss_mask, self.config.prompt_length, 0, left_pad=True)
        response_loss_mask = pad_sequence(response_loss_mask, batch_first=True, padding_value=0)
        if response_loss_mask.shape[1] < self.config.response_length:
            response_loss_mask = pad_sequence_to_length(response_loss_mask, self.config.response_length, 0)
        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)
        position_ids = torch.cat((prompt_position_ids, response_position_ids), dim=-1)
        loss_mask = torch.cat((prompt_loss_mask, response_loss_mask), dim=-1)
       
        if self._tp_rank == 0:
            print(f"examine first request:\n{sorted_output_req_list[0].messages=}\n{self.tokenizer.decode(sorted_output_req_list[0].input_ids)=}")
        # Construct the batch data
        batch = TensorDict(
            {
                "prompts": prompt_ids,
                "responses": response_ids,
                "input_ids": input_ids,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
                # "reward_scores": reward_scores_tensor,
            },
            batch_size=len(sorted_output_req_list)
        )

        return DataProto(batch=batch, non_tensor_batch={"messages": np.array(messages), "reward_scores": np.array(reward_scores)})
    
    def _preprocess_prompt_to_async_rollout_requests(self, prompts: DataProto, n: int) -> List[AsyncRolloutRequest]:
        assert 'raw_prompt' in prompts.non_tensor_batch, "need data.return_raw_chat=True, due to no official way do parse_messages"
        req_list = []
        for data_idx, raw_prompt in enumerate(prompts.non_tensor_batch['raw_prompt']):
            for rollout_offset in range(n):
                if self._tool_schemas:
                    prompt_with_chat_template = self.tokenizer.apply_chat_template(
                        conversation=raw_prompt,
                        tools=self._tool_schemas,
                        add_generation_prompt=True,
                        tokenize=False,
                        return_tensors="pt",
                    )
                    input_data = self.tokenizer(prompt_with_chat_template, return_tensors="pt", add_special_tokens=False)
                    _input_ids = input_data['input_ids'][0].tolist()
                    _attention_mask = input_data['attention_mask'][0].tolist()
                    _position_ids = compute_position_id_with_mask(input_data['attention_mask'][0]).tolist()
                    _tools_kwargs = prompts.non_tensor_batch['tools_kwargs'][data_idx]
                    _tool_schemas = []
                    for k in _tools_kwargs.keys():
                        _tool_schemas.append(self._tool_map[k].get_openai_tool_schema())
                    if len(_input_ids) > self.config.prompt_length:
                        logger.warning(f"Prompt {data_idx} has length {len(_input_ids)} greater than max_prompt_len {self.config.prompt_length}")
                        _input_ids = _input_ids[:self.config.prompt_length]
                        _attention_mask = _attention_mask[:self.config.prompt_length]
                        _position_ids = _position_ids[:self.config.prompt_length]
                else:
                    _input_ids = _pre_process_inputs(self.pad_token_id, prompts.batch['input_ids'][data_idx])
                    _attention_mask = _pre_process_inputs(0, prompts.batch['attention_mask'][data_idx])
                    _position_ids = compute_position_id_with_mask(torch.tensor(_attention_mask)).tolist()
                    _tool_schemas = []
                    _tools_kwargs = {}
                
                req = AsyncRolloutRequest(
                    batch_data_id=data_idx,
                    rollout_offset=rollout_offset,
                    request_id=str(uuid4()),
                    state=AsyncRolloutRequestStateEnum.PENDING,
                    messages=[Message.model_validate(msg) for msg in raw_prompt],
                    tools=_tool_schemas,
                    tools_kwargs=_tools_kwargs,
                    input_ids=_input_ids,
                    prompt_ids=_input_ids,
                    response_ids=[],
                    attention_mask=_attention_mask,
                    prompt_attention_mask=_attention_mask,
                    response_attention_mask=[],
                    position_ids=_position_ids,
                    prompt_position_ids=_position_ids,
                    response_position_ids=[],
                    loss_mask=[0] * len(_input_ids),
                    prompt_loss_mask=[0] * len(_input_ids),
                    response_loss_mask=[],
                    reward_scores={},
                    max_response_len=self.config.response_length,
                    max_model_len=min(self.config.max_model_len, self.config.prompt_length + self.config.response_length)
                )
                assert len(req.input_ids) == len(req.attention_mask) == len(req.position_ids) == len(req.loss_mask), \
                    f"Request {req.request_id} has different length of {len(req.input_ids)=}, {len(req.attention_mask)=}, {len(req.position_ids)=}, {len(req.loss_mask)=},\n{self.pad_token_id=},\n{req.input_ids=},\n{req.attention_mask=},\n{req.position_ids=},\n{req.loss_mask=}"
                req_list.append(req)

        return req_list
