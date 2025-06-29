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
import enum
import functools
import heapq
import importlib
import itertools
import json
import logging
import os
from abc import ABC, abstractmethod
from concurrent.futures.thread import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple
from uuid import uuid4

import numpy as np
import torch
from cachetools import LRUCache
from omegaconf import DictConfig
from openai.types.chat.chat_completion import ChatCompletion
from tensordict import TensorDict
from transformers import PreTrainedTokenizer
from typing_extensions import runtime_checkable

from verl.protocol import DataProto
from verl.tools.base_tool import initialize_tools_from_config
from verl.utils.fs import copy_to_local
from verl.utils.tokenizer import hf_tokenizer
from verl.workers.rollout.chat_scheduler.apis import AsyncCallbackMixin, CallsReq, CoroExternalCallsPlugin, ReduceResp, RolloutReq, RolloutResp
from verl.workers.rollout.chat_scheduler.utils import ActorMeta, DeathLetter, QueueGroup, WorkStealingActor, concat_data_proto

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class CompletionCallback(ABC):
    def __init__(self, config: DictConfig, scheduler: "ChatCompletionScheduler"):
        self.config = config
        self.scheduler = scheduler

        # Initialize tools from config file
        self.max_turns = config.actor_rollout_ref.rollout.multi_turn.max_turns
        tool_config_path = config.actor_rollout_ref.rollout.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        self.tools = {tool.name: tool for tool in tool_list}
        self._tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        print(f"Initialized tools: {self.tools}", flush=True)

        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        self.tokenizer: PreTrainedTokenizer = hf_tokenizer(local_path, trust_remote_code=True)

    @property
    def tool_schemas(self):
        """OpenAI JSON tool schemas."""
        return self._tool_schemas

    @property
    def extra_body(self) -> Dict[str, Any]:
        """Extra body pass to OpenAI API."""
        return None

    @abstractmethod
    async def __call__(self, messages: List[Dict[str, str]], completions: ChatCompletion, info: Dict[str, Any]):
        """Call back function to process completions.

        Args:
            messages: List of messages including raw prompt and assistant, tool response generated so far.
            completions: Chat completions from OpenAI compatible server.
            info: Any other auxiliary information pass across multi-turn.
        """
        raise NotImplementedError

    @abstractmethod
    def postprocess(self, batch: DataProto, batch_conversations: List[List[Dict[str, str]]], n: int) -> DataProto:
        """Post process batch data.

        Args:
            batch: Batch input messages from RLHFDataset.
            batch_conversations: List of messages including raw prompt, assistant response, tool response.
                Note that `len(batch_conversations) == len(batch) * n`, e.g n=2,
                batch_conversations=[messages_0_0, messages_0_1, messages_1_0, messages_1_1, ...]
            n: How many chat completion choices to generate for each input message.

        Returns:
            Batch data, should include ["prompts", "responses", "response_mask", "input_ids", "attention_mask", "position_ids"].
        """
        raise NotImplementedError


class ToolCompletionCallback(CompletionCallback):
    def __init__(self, config: DictConfig, scheduler: "ChatCompletionScheduler"):
        super().__init__(config, scheduler)

        # TODO: add reward manager to calculate reward score once a sample finish

    async def __call__(self, messages: List[Dict[str, str]], completions: ChatCompletion, info: Dict[str, Any]):
        message = completions.choices[0].message.model_dump(exclude_unset=True, exclude_none=True)
        if "content" not in message:
            message["content"] = ""
        messages.append(message)
        finish_reason = completions.choices[0].finish_reason

        # STEP 0: check if we reach max turns
        if self.max_turns and len(messages) >= self.max_turns:
            print(f"[id={completions.id},turn={len(messages)},finish_reason={finish_reason}] Reach max turns, done!")
            return

        # STEP 1: check if the model called tools
        if finish_reason != "tool_calls":
            # print(f"[id={completions.id},turn={len(messages)},finish_reason={finish_reason}] No tool called, done!")
            return

        # STEP 2: call tools
        tool_calls = completions.choices[0].message.tool_calls
        print(f"[id={completions.id},turn={len(messages)},finish_reason={finish_reason}] Call {len(tool_calls)} tools")
        tasks = []
        for tool_call in tool_calls:
            tasks.append(self._call_tool(tool_call))
        tool_responses = await asyncio.gather(*tasks)
        if any(isinstance(item, Exception) for item in tool_responses):
            logger.debug(f"[id={completions.id},turn={len(messages)},finish_reason={finish_reason}] Error when calling tools, done!")
            return
        messages.extend(tool_responses)

        # STEP 3: resubmit completion request with tool responses
        self.scheduler.submit_chat_completions(messages=messages, request_id=completions.id, info=info)

    async def _call_tool(self, tool_call) -> Dict[str, str]:
        """Call tool and return tool response."""
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)
        tool = self.tools[tool_name]

        instance_id = await tool.create()
        try:
            tool_response, tool_reward_score, tool_metrics = await tool.execute(instance_id, tool_args)
        except Exception as e:
            logger.exception(f"Error when executing tool: {e}")
            return e
        finally:
            await tool.release(instance_id)

        return {
            "role": "tool",
            "content": tool_response,
            "tool_call_id": tool_call.id,
        }

    def postprocess(self, batch: DataProto, batch_conversations: List[List[Dict[str, str]]], n: int) -> DataProto:
        # NOTE: consistent with batch version of generate_sequences in vllm_rollout_spmd.py
        # prompts: left pad
        # responses: right pad
        # input_ids: prompt + response
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]

        # prompts: [prompt] from input dataset
        prompts = [self.tokenizer.apply_chat_template(prompt, tools=self.tool_schemas, add_generation_prompt=True, tokenize=False) for prompt in batch.non_tensor_batch["raw_prompt"]]
        assert len(batch_conversations) == len(prompts) * n

        # sequences: [prompt + response]
        sequences = [self.tokenizer.apply_chat_template(conversation, tools=self.tool_schemas, add_generation_prompt=False, tokenize=False) for conversation in batch_conversations]

        # responses: [response]
        responses = [sequence[len(prompts[i // n]) :] for i, sequence in enumerate(sequences)]

        prompts = self.tokenizer(prompts, return_tensors="pt", padding="longest", padding_side="left")
        responses = self.tokenizer(responses, return_tensors="pt", padding="longest", padding_side="right")
        if n > 1:
            prompts["input_ids"] = prompts["input_ids"].repeat_interleave(n, dim=0)
            prompts["attention_mask"] = prompts["attention_mask"].repeat_interleave(n, dim=0)

        # response_mask: response mask with tools calling masked out
        response_mask = self._mask_out_tools_calling_tokens(batch.non_tensor_batch["raw_prompt"].repeat(n, axis=0), batch_conversations, responses["input_ids"], responses["attention_mask"])

        input_ids = torch.cat([prompts["input_ids"], responses["input_ids"]], dim=1)
        attention_mask = torch.cat([prompts["attention_mask"], responses["attention_mask"]], dim=1)
        position_ids = (attention_mask.cumsum(dim=1) - 1) * attention_mask

        batch = TensorDict(
            {
                "prompts": prompts["input_ids"],  # [bsz, prompt_length]
                "responses": responses["input_ids"],  # [bsz, response_length]
                "response_mask": response_mask,  # [bsz, response_length]
                "input_ids": input_ids,  # [bsz, prompt_length + response_length]
                "attention_mask": attention_mask,  # [bsz, prompt_length + response_length]
                "position_ids": position_ids,  # [bsz, prompt_length + response_length]
            },
            batch_size=len(input_ids),
        )

        num_turns = np.array([len(conversation) for conversation in batch_conversations], dtype=np.int32)
        return DataProto(batch=batch, non_tensor_batch={"__num_turns__": num_turns})

    def _mask_out_tools_calling_tokens(
        self,
        raw_prompts: List[List[Dict[str, str]]],
        batch_conversations: List[List[Dict[str, str]]],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mask out tools calling tokens in the responses.

        Args:
            raw_prompts: [prompt] from input dataset
            batch_conversations: [prompt + response]
            input_ids: responses tokens
            attention_mask: responses attention mask

        Returns:
            mask: (batch_size, response_length)
        """
        batch_size = input_ids.size(0)
        assert len(raw_prompts) == batch_size, f"{len(raw_prompts)} != {batch_size}"
        assert len(batch_conversations) == batch_size, f"{len(batch_conversations)} != {batch_size}"

        # Deduplicate adjacent tool calls, since they're merged into one turn.
        # [user, assistant, tool, tool, assistant] -> [user, assistant, tool, assistant]
        # TODO: it's chat_template specific, find a more generic way to do this.
        def deduplicate_adjacent_tool_calls(roles):
            result = []
            for role, group in itertools.groupby(roles):
                if role == "tool":
                    result.append(role)
                else:
                    result.extend(group)
            return result

        loss_mask = attention_mask.clone()
        for i in range(batch_size):
            responses = batch_conversations[i][len(raw_prompts[i]) :]
            assert len(responses) > 0, f"responses is empty: {responses}"

            roles = deduplicate_adjacent_tool_calls([response["role"] for response in responses])
            # Each turn should be: [BOS]...[EOS]
            eos_indices = input_ids[i].eq(self.tokenizer.eos_token_id).nonzero().squeeze(1)[: len(roles)]
            for j in range(len(roles)):
                if roles[j] == "tool":
                    bos = eos_indices[j - 1] + 1 if j > 0 else 0
                    eos = eos_indices[j]
                    loss_mask[i, bos : eos + 1] = 0

        return loss_mask


class AsyncToolCompletionCallback(ToolCompletionCallback, CoroExternalCallsPlugin):
    def __init__(self, config: DictConfig, scheduler: "ChatCompletionScheduler"):
        print("init_async tools")
        ToolCompletionCallback.__init__(self, config, scheduler)
        CoroExternalCallsPlugin.__init__(self, num_workers=5)

    def hit(self, req: CallsReq):
        completions = req.completions
        messages = req.messages
        finish_reason = completions.choices[0].finish_reason

        # STEP 0: check if we reach max turns
        if self.max_turns and len(messages) >= self.max_turns:
            logger.debug(f"[id={completions.id},turn={len(messages)},finish_reason={finish_reason}] Reach max turns, done!")
            return True

        # STEP 1: check if the model called tools
        if finish_reason != "tool_calls":
            logger.debug(f"[id={completions.id},turn={len(messages)},finish_reason={finish_reason}] No tool called, done!")
            return False

    async def __call__(self, req: CallsReq):
        completions = req.rollout_resp.completions
        messages = req.rollout_resp.messages
        finish_reason = completions.choices[0].finish_reason

        # call tools
        tool_calls = completions.choices[0].message.tool_calls
        logger.debug(f"[id={completions.id},turn={len(messages)},finish_reason={finish_reason}] Call {len(tool_calls)} tools")
        tasks = []
        for tool_call in tool_calls:
            tasks.append(self._call_tool(tool_call))
        tool_responses = await asyncio.gather(*tasks)
        if any(isinstance(item, Exception) for item in tool_responses):
            logger.debug(f"[id={completions.id},turn={len(messages)},finish_reason={finish_reason}] Error when calling tools, done!")
            return
        messages.extend(tool_responses)

        # STEP 3: send it back to local_queue
        new_rollout_req = RolloutReq(
            info=req.rollout_resp.info,
            messages=messages,
            model_name=req.rollout_resp.model_name,
            chat_complete_request=req.rollout_resp.chat_complete_request,
        )
        req.actor_meta.queue_group.push(req.actor_meta.actor_id, new_rollout_req)

    def postprocess(self, batch, batch_conversations, n):
        data_proto = super().postprocess(batch, batch_conversations, n)
        return data_proto

    def new_postprocess(self, batch_conversations: List[ReduceResp], n) -> DataProto:
        # do collate function
        # NOTE: consistent with batch version of generate_sequences in vllm_rollout_spmd.py
        # prompts: left pad
        # responses: right pad
        # input_ids: prompt + response
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]

        raw_prompts = np.array([resp.raw_prompt for resp in batch_conversations], dtype=object)

        raw_messages = [message for resp in batch_conversations for message in resp.messages]
        # prompts: [prompt] from input dataset
        prompts = [self.tokenizer.apply_chat_template(resp.raw_prompt, tools=self.tool_schemas, add_generation_prompt=True, tokenize=False) for resp in batch_conversations]

        # sequences: [prompt + response]

        sequences = [self.tokenizer.apply_chat_template(message, tools=self.tool_schemas, add_generation_prompt=False, tokenize=False) for resp in batch_conversations for message in resp.messages]

        assert len(sequences) == len(prompts) * n
        # responses: [response]
        responses = [sequence[len(prompts[i // n]) :] for i, sequence in enumerate(sequences)]

        prompts = self.tokenizer(prompts, return_tensors="pt", padding="longest", padding_side="left")
        responses = self.tokenizer(responses, return_tensors="pt", padding="longest", padding_side="right")
        if n > 1:
            prompts["input_ids"] = prompts["input_ids"].repeat_interleave(n, dim=0)
            prompts["attention_mask"] = prompts["attention_mask"].repeat_interleave(n, dim=0)

        # response_mask: response mask with tools calling masked out
        response_mask = self._mask_out_tools_calling_tokens(raw_prompts.repeat(n, axis=0), raw_messages, responses["input_ids"], responses["attention_mask"])

        input_ids = torch.cat([prompts["input_ids"], responses["input_ids"]], dim=1)
        attention_mask = torch.cat([prompts["attention_mask"], responses["attention_mask"]], dim=1)
        position_ids = (attention_mask.cumsum(dim=1) - 1) * attention_mask

        batch = TensorDict(
            {
                "prompts": prompts["input_ids"],  # [bsz, prompt_length]
                "responses": responses["input_ids"],  # [bsz, response_length]
                "response_mask": response_mask,  # [bsz, response_length]
                "input_ids": input_ids,  # [bsz, prompt_length + response_length]
                "attention_mask": attention_mask,  # [bsz, prompt_length + response_length]
                "position_ids": position_ids,  # [bsz, prompt_length + response_length]
            },
            batch_size=len(input_ids),
        )

        num_turns = np.array([len(conversation) for conversation in raw_messages], dtype=np.int32)
        return DataProto(batch=batch, non_tensor_batch={"__num_turns__": num_turns})


@runtime_checkable
class StreamSchedulerMixin(Protocol):
    async def stream_generate_sequences(self, data_iter: Iterable, batch_size: int) -> Tuple[bool, DataProto, DataProto, DataProto]: ...


class ChatCompletionScheduler:
    def __init__(
        self,
        config: DictConfig,
        server_addresses: List[str],
        max_cache_size: int = 10000,
    ):
        """
        Args:
            config: DictConfig.
            server_addresses: List[str], OpenAI compatible server addresses.
            max_cache_size: int, max cache size of request_id to address mapping.
        """
        self.config = config.actor_rollout_ref.rollout
        model_path = config.actor_rollout_ref.model.path
        self.model_name = "/".join(model_path.split("/")[-2:])

        # Least requests load balancing
        self.weighted_addresses = [[0, address] for address in server_addresses]
        heapq.heapify(self.weighted_addresses)

        # LRU cache to map request_id to address
        self.request_id_to_address = LRUCache(maxsize=max_cache_size)

        self.background_tasks = set()
        if self.config.multi_turn.completion_callback is None:
            self.completion_callback = ToolCompletionCallback(config, self)
            logger.warning("completion_callback is None, use ToolCompletionCallback")
        else:
            module_path, class_name = self.config.multi_turn.completion_callback.rsplit(".", 1)
            module = importlib.import_module(module_path)
            self.completion_callback = getattr(module, class_name)(config, self)

    def submit_chat_completions(self, *, messages: List[Dict[str, str]], request_id: str, info: Dict[str, Any], address: Optional[str] = None):
        """Submit chat completion request without wait, completion_callback will be called when the request is done.

        Args:
            messages: List of messages.
            request_id: Request id.
            info: Any other auxiliary information pass across multi-turn.
        """
        info["__depth__"] += 1
        task = asyncio.create_task(self._submit_chat_completions_and_callback(messages, request_id, info, address))

        # “fire-and-forget” background tasks
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)

    def _routing(self, request_id: str, address: str):
        if address is not None:
            return address
        if request_id:
            request_id = request_id.removeprefix("chatcmpl-")
            assert request_id in self.request_id_to_address
            address = self.request_id_to_address.pop(request_id)
        else:
            address = self.weighted_addresses[0][1]
            self.weighted_addresses[0][0] += 1
            heapq.heapreplace(self.weighted_addresses, self.weighted_addresses[0])
        return address

    async def _submit_chat_completions_and_callback(
        self,
        messages: List[Dict[str, str]],
        request_id: str,
        info: Dict[str, Any],
        address: str = None,
    ):
        """Submit chat completion request, wait request finish and do callback."""
        address = self._routing(request_id, address)
        # use new request_id to avoid duplicate request_id problem
        request_id = uuid4().hex
        self.request_id_to_address[request_id] = address

        completions, exception = None, None
        try:
            # NOTE: OpenAI client uses httpx, seems to have performance issue in high concurrency requests.
            completions = await self._chat_completions_aiohttp(
                address,
                messages=messages,
                tools=self.completion_callback.tool_schemas,
                extra_body=self.completion_callback.extra_body,
                extra_headers={"x-request-id": request_id},
                **info["__sampling_params__"],
            )
        except Exception as e:
            # Let user handle the exception
            exception = e

        info["__depth__"] -= 1

        if exception is not None:
            logger.exception(f"chat completion failed with exception: {exception}")
        else:
            try:
                await self.completion_callback(messages, completions, info)
            except Exception as e:
                logger.exception(f"completion callback failed with exception: {e}")

        # No more ongoing completion requests
        if info["__depth__"] == 0:
            info["__done__"].set()

    async def _chat_completions_openai(self, address: str, **chat_complete_request) -> ChatCompletion:
        from verl.workers.rollout.chat_scheduler.requests import chat_completions_openai

        return await chat_completions_openai(address, **chat_complete_request)

    async def _chat_completions_aiohttp(self, address: str, **chat_complete_request) -> ChatCompletion:
        from verl.workers.rollout.chat_scheduler.requests import chat_completions_aiohttp

        return await chat_completions_aiohttp(address, **chat_complete_request)

    async def generate_sequences(self, batch: DataProto) -> Tuple[bool, DataProto, DataProto, DataProto]:
        # stop_iter , gen_batch_output, gen_batch, batch
        # this follow the same pattern for sync mode, therefore we take one batch from data_iter
        kwargs = dict(
            model=self.model_name,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )

        # override sampling params for validation
        if batch.meta_info.get("validate", False):
            kwargs["top_p"] = self.config.val_kwargs.top_p
            kwargs["temperature"] = self.config.val_kwargs.temperature

        print(f"[ChatCompletionScheduler] generate_sequences sampling params: {kwargs}")

        # NOTE: For multi-turn rollout, repeat raw_prompt n times and process each prompt independently,
        # validation dataset has already been repeated in `PPOTrainer._validate`.
        n = 1 if batch.meta_info.get("validate", False) else self.config.n
        tasks, batch_conversations = [], [None] * len(batch) * n
        for batch_index, conversation in enumerate(batch.non_tensor_batch["raw_prompt"].repeat(n, axis=0)):
            # raw_prompt: [{"role": "user", "content": ""}, ["role": "assistant", "content"], ...]
            batch_conversations[batch_index] = conversation.tolist()

            tasks.append(
                asyncio.create_task(
                    self._submit_chat_completions_semaphore(
                        messages=batch_conversations[batch_index],
                        request_id=None,
                        sampling_params=kwargs,
                    )
                )
            )

        await asyncio.gather(*tasks)
        print("[ChatCompletionScheduler] generate_sequences done")

        return self.completion_callback.postprocess(batch, batch_conversations, n=n)

    async def _submit_chat_completions_semaphore(self, messages: List[Dict[str, str]], request_id: str, sampling_params: Dict[str, Any], address: str = None):
        done = asyncio.Event()

        info = {
            "__done__": done,
            "__depth__": 0,  # indicate how many ongoing completion requests
            "__sampling_params__": sampling_params,
        }

        self.submit_chat_completions(messages=messages, request_id=request_id, info=info, address=address)

        # Wait until all completion requests are done
        await done.wait()


class MicroBatchScheduler(ChatCompletionScheduler):
    def __init__(self, config, server_addresses, max_cache_size=10000, rollout_rate=1, max_inflight_req=8, rollout_req_handler=None, reduce_handler=None, enable_work_stealing=True):
        super().__init__(config, server_addresses, max_cache_size)
        self.mirco_batch_config = config.actor_rollout_ref.rollout.chat_scheduler
        print(self.config)
        self._validate_callback()
        self.micro_batch_per_dp = self.mirco_batch_config.micro_batch.max_inflight_req if self.mirco_batch_config.micro_batch.max_inflight_req else max_inflight_req
        self.server_addresses = server_addresses
        self.enable_work_stealing = self.mirco_batch_config.micro_batch.enable_work_stealing if self.mirco_batch_config.micro_batch.enable_work_stealing else enable_work_stealing
        self.number_of_servers = len(server_addresses)
        self.rollout_rate = 1
        self.rollout_req_handler = rollout_req_handler if rollout_req_handler else self.default_handle_rollout_req
        self.reduce_handler = reduce_handler if reduce_handler else self.default_handle_reduce_req
        self.initialized = False
        self.reduce_format = "ReduceResp"

    def set_rollout_rate(self, rate):
        assert rate <= 1 and rate > 0, "rollout rate must be in (0, 1]"
        self.rollout_rate = rate

    def _get_rollout_batch_size(self, data_batch_size):
        return int(data_batch_size * self.rollout_rate)

    def _validate_callback(self):
        if self.completion_callback is None:
            raise ValueError("completion_callback is None")
        if not isinstance(self.completion_callback, AsyncCallbackMixin):
            raise ValueError("completion_callback mixin AsyncCallbackMixin")
        logger.error(f"completion_callback: {self.completion_callback}")

    def _lazy_init_global_resource(self):
        if self.initialized:
            return
        else:
            self.initialized = True
        # TODO use ZMQ to implement pub-sub for debug purpose
        self.loop = asyncio.get_event_loop()
        self.death_letter = asyncio.Queue()
        self.global_data_queue = asyncio.Queue()
        self.local_data_queue_group = QueueGroup(self.number_of_servers, [asyncio.Queue() for _ in range(self.number_of_servers)])
        self.reduce_data_queue = asyncio.Queue()
        # TODO better implement a supervisor-tree pattern, include dead-letter-queue to monitor whether any actor exit unexpectly
        self.engine_call_actors: List[WorkStealingActor] = self._init_engine_call_actors(server_address=self.server_addresses, max_inflight_req=self.micro_batch_per_dp)
        self.completion_callback.init_plugin_callers()
        self._init_death_letter_consumer()
        logger.info(f"start MicroBatchChatCompletionScheduler, with max_inflight_req: {self.micro_batch_per_dp}, enable_work_stealing: {self.enable_work_stealing}, server_address: {self.server_addresses}")

    def _init_death_letter_consumer(self):
        async def consume_death_letter():
            while True:
                letter = await self.death_letter.get()
                print(f"[MicroBatchChatCompletionScheduler] consume death letter: {letter}")

        asyncio.create_task(consume_death_letter())

    def _init_engine_call_actors(self, server_address, max_inflight_req):
        # we use a group of coroutine to consume send_queue and produce reduce_queue
        # since the asyncio.Queue is not thread safe.
        # max_inflight_req consumer coroutine to get element from local_queue and submit to vllm
        actors = []
        counter = 0
        for idx, addr in enumerate(server_address):
            print(f"[MicroBatchChatCompletionScheduler] init engine call actor {addr}, max_inflight_req: {max_inflight_req}")
            for _ in range(max_inflight_req):
                work_fn = functools.partial(
                    self.rollout_req_handler,
                    addr,
                    self.reduce_data_queue,
                    self.completion_callback,
                )
                actor = WorkStealingActor(worker_id=idx, local_id=counter, local_queues=self.local_data_queue_group, global_queue=self.global_data_queue, work_fn=work_fn, enable_work_stealing=self.enable_work_stealing, death_letter=self.death_letter)
                actors.append(actor)
                counter += 1
        print(f"[MicroBatchChatCompletionScheduler] init engine call actors done, total: {len(actors)}")
        return actors

    def wake_up_engine_actor(
        self,
    ):
        for actor in self.engine_call_actors:
            actor.wakeup()
        self.completion_callback.wake_up()

    async def shut_down_actors(self):
        print("shut down engine actors with length: ", len(self.engine_call_actors))
        for actor in self.engine_call_actors:
            print("ready to shutdown actor: ", actor.actor_meta)
            await actor.shutdown()
        print("[MicroBatchChatCompletionScheduler] shut down engine actor")
        await self.completion_callback.shutdown()
        print("[MicroBatchChatCompletionScheduler] shut down completion callback")

    async def default_handle_rollout_req(self, addr, reduce_queue: asyncio.Queue, external_call: AsyncCallbackMixin, actor_meta: ActorMeta, rollout_req: RolloutReq):
        from verl.workers.rollout.chat_scheduler.requests import chat_completions_aiohttp

        logger.debug(f"[MicroBatchChatCompletionScheduler] _consumer process get sample, addr: {addr}, actor_meta: {actor_meta}")
        request_id = uuid4().hex
        completions, exception, message = None, None, {}
        messages = rollout_req.messages
        try:
            # NOTE: OpenAI client uses httpx, seems to have performance issue in high concurrency requests.
            logger.debug(f"[MicroBatchChatCompletionScheduler] _consumer process get sample, submit to engine {addr}")
            completions = await chat_completions_aiohttp(
                address=addr,
                messages=messages,
                tools=rollout_req.tools_schema,
                extra_body=rollout_req.extra_body,
                extra_headers={"x-request-id": request_id},
                **rollout_req.sampling_params,
            )
            message = completions.choices[0].message.model_dump(exclude_unset=True, exclude_none=True)
        except asyncio.CancelledError as cancel_err:
            logger.warning(f"chat completion failed with exception: {cancel_err}")
            raise cancel_err
        except Exception as e:
            logger.warning(f"chat completion failed with exception: {e}")
            exception = e
        logger.debug(f"[MicroBatchChatCompletionScheduler] _consumer process get sample done,meesage: {message}", actor_meta)
        if "content" not in message:
            message["content"] = ""
        messages.append(message)
        resp = RolloutResp(request=rollout_req, completions=completions, exception=exception, req_id=request_id, messages=messages)
        try:
            if external_call.hit(resp):
                logger.debug(f"[id={completions.id},turn={len(messages)},finish_reason={completions.choices[0].finish_reason}] Call tools")
                external_call.put(CallsReq(rollout_resp=resp, actor_meta=actor_meta))
            else:
                logger.debug(f"[MicroBatchChatCompletionScheduler] _consumer process put sample to reduce_queue,idx: {actor_meta.actor_id}")
                reduce_queue.put_nowait(resp)
        except Exception as e:
            logger.warning(f"[MicroBatchChatCompletionScheduler] _consumer process put sample to reduce_queue failed,idx: {actor_meta.actor_id}, exception: {e}")
            resp.exception = e
            reduce_queue.put_nowait(resp)
        print("[MicroBatchChatCompletionScheduler] _consumer process done")

    # maybe we can make this sink_queue as a pubsub proxy using zmq
    async def default_handle_reduce_req(self, batch_size, n_sample, sink_queue: asyncio.Queue = None, format="ReduceResp"):
        batch_conversations = [None] * len(batch_size)
        # joiner_buffer worked as key for raw_prompt,value for result.
        # make sure n-sample arrived correctlly then ship to batch_conversations as result
        joiner_buffer: Dict[str, List[List[Dict[str, str]]]] = {}
        counter = 0
        while counter < batch_size:
            print(f"[MicroBatchChatCompletionScheduler] _gather_result counter: {counter}")
            sample: RolloutResp = await self.reduce_data_queue.get()
            if sink_queue is not None:
                sink_queue.put(sample)
            if sample.exception is not None:
                raise sample.exception
            sample_id = sample.request.sample_id
            if sample_id in joiner_buffer.keys():
                joiner_buffer[sample_id].append(sample.messages)
            else:
                joiner_buffer[sample_id] = [sample.messages]
            if len(joiner_buffer[sample_id]) == n_sample:
                if format == "ReduceResp":
                    logger.debug(f"finished for samples: {sample_id}")
                    batch_conversations[sample.request.sample_id] = ReduceResp(raw_prompt=sample.request.raw_prompt, messages=joiner_buffer[sample_id])
                else:
                    batch_conversations[sample.request.sample_id] = joiner_buffer[sample_id]
                counter += 1
        print("[MicroBatchChatCompletionScheduler] _gather_result done for one batch")
        return batch_conversations

    async def generate_sequences(self, batch: DataProto, **sampling_params) -> DataProto:
        self._lazy_init_global_resource()
        self.wake_up_engine_actor()
        kwargs = dict(
            model=self.model_name,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )

        # override sampling params for validation
        if batch.meta_info.get("validate", False):
            kwargs["top_p"] = self.config.val_kwargs.top_p
            kwargs["temperature"] = self.config.val_kwargs.temperature

        # NOTE: For multi-turn rollout, repeat raw_prompt n times and process each prompt independently,
        # validation dataset has already been repeated in `PPOTrainer._validate`.
        n = 1 if batch.meta_info.get("validate", False) else self.config.n
        print(f"[ChatCompletionScheduler] generate_sequences sampling params: {kwargs}")
        for batch_index, conversation in enumerate(batch.non_tensor_batch["raw_prompt"].repeat(n, axis=0)):
            # raw_prompt: [{"role": "user", "content": ""}, ["role": "assistant", "content"], ...]
            self.global_data_queue.put_nowait(
                RolloutReq(
                    raw_prompt=conversation,
                    messages=conversation.tolist(),
                    model_name=self.model_name,
                    sampling_params=kwargs,
                    tools_schema=self.completion_callback.tool_schemas,
                    extra_body=self.completion_callback.extra_body,
                    verl_session_id=uuid4().hex,
                    sample_id=batch_index // n,
                )
            )
        print("[MicroBatchChatCompletionScheduler] generate_sequences start, with len(batch): ", len(batch))
        batch_conversations = await self.reduce_handler(self._get_rollout_batch_size(len(batch)), n_sample=n, format=self.reduce_format)
        print(f"partial rollout done, cancel all left request, real size: {len(batch_conversations)}")
        print("[MicroBatchChatCompletionScheduler] generate_sequences done")
        if self.reduce_format == "ReduceResp":
            return self.completion_callback.new_postprocess(batch_conversations, n=n)
        else:
            return self.completion_callback.postprocess(batch, batch_conversations, n=n)


@dataclass
class _State(enum.Enum):
    PENDING = 0
    ACTIVE = 1
    FINISHED = 2


@dataclass
class _Sample:
    rollout_req: RolloutReq
    batch: DataProto
    gen_batch: DataProto
    generation: int


class StreamScheduler(MicroBatchScheduler, StreamSchedulerMixin):
    # StreamScheduler is designed for partial rollout,which aiming to sovle the long tail generations,
    # which means for one batch generation, some samples might be too long to become the struggler.
    # to solve this, we have following solotions

    # 1. using dynamic batching strategy: we might drop those long tail request, and put them into next batch.
    # this strategy offer user options like
    # # a. percentages: engine will stop generation when x% samples have been done.
    # # b. tokens: engine will stop generation when then training samples have generate more then X Tokens.

    # 2. prefetching data from next batch: this strategy will keep fetch data from data iterator and send it to
    # serving engine until the stop terms are met. this strategy works since we basiclly select the shortest generation
    # samples to fill this batch and postpone those long tail samples (even though those will be sent to engine, but eventually
    # they will be 'dropped' since other prefetched samples will be in the output buffer faster then them).
    # couple things we might need to be careful with:

    # # a. host memory issue, there might be a chances that the data fetcher will fetch expect_batch_size*n size of prompts into
    # memory, since it tries to maximize the engine utilization by feeding data as much as possiable. we need to control the maximum
    # number of inflight req (data in global queue, tool-calling state, serving state) to avoid loading too much req in mem.

    # # b. abort pattern: when we hit those stop terms, we should cancel those inflight req. the question is how should we deal
    # with the result of partial generations. drop/save kv cache/ save result/ staleness factor, might be considered.
    # we also need to worry about the requeue pattern, since for grpo, we need to requeue n-samples all-together

    # # c.re-generate batch/gen_batch for training.

    # # d. intergation for pytorch dataloader and asyncio.
    def __init__(self, config, server_addresses, rollout_rate=1, max_inflight_req=8, rollout_req_handler=None, reduce_handler=None, enable_work_stealing=True, data_fetcher=None):
        self.data_memory_utils = 0.1
        self.global_blocker = asyncio.Event()
        self.data_iter = None
        self.data_fetcher = data_fetcher if data_fetcher else self._default_data_fetcher
        self.rollout_req_handler = rollout_req_handler if rollout_req_handler else self.stream_handle_rollout_req
        self.reduce_handler = reduce_handler if reduce_handler else self.stream_handle_reduce_req
        self.data_fetcher_actor = None
        self.data_fetcher_exit = asyncio.Event()
        self.pending_sample: Dict[str, _Sample] = {}
        self.active_sample: Dict[str, _Sample] = {}
        self.done_sample_counter = 0
        self.data_iter_length = 0
        self.buffer_size = 30
        super().__init__(config, server_addresses, 1000, rollout_rate, max_inflight_req, self.rollout_req_handler, self.reduce_handler, enable_work_stealing)

    async def _memory_monitor(self, blocker: asyncio.Event, utilization):
        await asyncio.sleep(2)
        if 1 < utilization:
            logger.debug(f"memory utilization {utilization} exceed threshold {self.data_memory_utils}, stop data fetcher")
            blocker.clear()
        else:
            # wake up data fetcher
            logger.debug(f"memory utilization {utilization} under threshold {self.data_memory_utils}, wake up data fetcher")
            blocker.set()

    def start_memory_monitor(self):
        self.mem_monitor_actor = asyncio.create_task(self._memory_monitor(self.global_blocker, self.data_memory_utils))

        def callback(task):
            if task.exception() is not None:
                letter = DeathLetter(
                    actor_meta=self.actor_meta,
                    async_task=task,
                )
                if self.death_letter is not None:
                    self.death_letter.put_nowait(letter)
                else:
                    logger.warning("global data fetcher exit")

        self.data_fetcher_actor.add_done_callback(callback)

    async def _default_data_fetcher(self, data_iter):
        class _Iter:
            def __init__(self, data_iter):
                self._thread_executor = ThreadPoolExecutor(1, thread_name_prefix="async_dataloader_thread")
                self.data_iter = data_iter

            async def __aiter__(self):
                def _next():
                    while True:
                        try:
                            batch_dict = next(self.data_iter)
                            batch: DataProto = DataProto.from_single_dict(batch_dict)
                            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                            if "multi_modal_data" in batch.non_tensor_batch:
                                non_tensor_batch_keys_to_pop.append("multi_modal_data")
                            if "raw_prompt" in batch.non_tensor_batch:
                                non_tensor_batch_keys_to_pop.append("raw_prompt")
                            if "tools_kwargs" in batch.non_tensor_batch:
                                non_tensor_batch_keys_to_pop.append("tools_kwargs")
                            gen_batch = batch.pop(
                                batch_keys=batch_keys_to_pop,
                                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                            )
                            return gen_batch, batch
                        except StopIteration as e:
                            raise StopAsyncIteration from e

                loop = asyncio.get_event_loop()
                while True:
                    try:
                        yield await loop.run_in_executor(self._thread_executor, _next)
                    except StopAsyncIteration:
                        break

        # this only works for trainer, if for validation.
        kwargs = dict(
            model=self.model_name,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )
        n = self.config.n
        print(f"[ChatCompletionScheduler] generate_sequences sampling params: {kwargs}")
        try:
            async for gen_next_batch in _Iter(data_iter):
                gen_batch, batch = gen_next_batch
                # assert len(batch) == 1
                for conversation in gen_batch.non_tensor_batch["raw_prompt"]:
                    sample_id = self._generate_unique_id()
                    rollout_req = RolloutReq(
                        raw_prompt=conversation, messages=conversation.tolist(), model_name=self.model_name, sampling_params=kwargs, tools_schema=self.completion_callback.tool_schemas, extra_body=self.completion_callback.extra_body, verl_session_id=None, generation=0, sample_id=sample_id
                    )
                    self.pending_sample[sample_id] = _Sample(rollout_req, batch, gen_batch, generation=0)
                    for _ in range(n):
                        _rollout_req = deepcopy(rollout_req)
                        _rollout_req.verl_session_id = uuid4().hex
                        logger.debug(f"put {sample_id} to global data queue")
                        # raw_prompt: [{"role": "user", "content": ""}, ["role": "assistant", "content"], ...]
                        await self.global_data_queue.put(_rollout_req)
        except asyncio.CancelledError:
            print("data fetcher cancled")
        except Exception as e:
            print(f"exit data fetcher for exception: {e}")
        print(f"exit data fetcher, pending sample size: {len(self.pending_sample)}")
        self.data_fetcher_exit.set()

    def _generate_unique_id(self):
        while 1:
            sample_id = uuid4().hex
            if sample_id not in self.pending_sample.keys():
                return sample_id

    def init_async_data_fetcher(self, data_iter, renew):
        print(f"[start_async_data_fetcher]: data iter: {data_iter}，renew: {renew}, {self.data_iter},{data_iter}")
        if not renew:
            return
        if self.data_fetcher_actor is not None and self.data_fetcher_actor.done():
            self.data_fetcher_actor.cancel()

        self.data_fetcher_actor = asyncio.create_task(self._default_data_fetcher(data_iter))
        self.data_iter_length = len(data_iter)
        self.data_fetcher_exit.clear()
        self.done_sample_counter = 0

        def callback(task):
            if task.exception() is not None:
                letter = DeathLetter(
                    actor_meta=self.actor_meta,
                    async_task=task,
                )
                if self.death_letter is not None:
                    self.death_letter.put_nowait(letter)
                print(f"global data fetcher exit for execption: {task}")

        self.data_fetcher_actor.add_done_callback(callback)
        self.global_blocker.set()
        print(f"[start_async_data_fetcher]: data fetcher actor: {self.data_fetcher_actor}")

    def _lazy_init_global_resource(self, data_iter: Iterable, renew):
        print("_lazy_init_global_resource")
        super()._lazy_init_global_resource()
        self.init_async_data_fetcher(data_iter, renew)

    def _data_fetcher_done(self) -> bool:
        return self.data_fetcher_exit.is_set()

    async def cancel_all_req(self):
        # cancel all req and put then back to the global queue.
        # put it back to local queue for prefix cache only works for
        # algorithms that reuse stale model's kv cache, user can re-implement this method to do that.
        # here we only implement the on-policy one, which will drop all previous results.
        evts = []
        for actor in self.engine_call_actors:
            maybe_set_evt: asyncio.Event = actor.cancel_task()
            if not maybe_set_evt.is_set():
                evts.append(maybe_set_evt.wait())
        print(f"cancel req with length: {len(evts)}")
        # cancel tool-calls
        print("[MicroBatchChatCompletionScheduler] shut down completion callback")
        await self.completion_callback.shutdown()
        print("[MicroBatchChatCompletionScheduler] shut down completion callback done")
        await asyncio.gather(*evts)

    def _skip_if_stale_request(self, rollout_req: RolloutReq, stage: str):
        # FIXME should not share any variable, we should change to push mode
        sample_id = rollout_req.sample_id
        if sample_id in self.pending_sample.keys():
            if self.pending_sample[sample_id].generation == rollout_req.generation:
                # first time see this sample, pop it from pending sample.
                # TODO reduce handler won't hit this term, make sanity check if necessary.
                self.active_sample[sample_id] = self.pending_sample.pop(sample_id)
                return False
            else:
                # stale generation,skip this
                print(f"[MicroBatchChatCompletionScheduler] _consumer process get sample,stage: {stage}, skip pending sample, sample_id: {sample_id}")
                return True
        elif sample_id in self.active_sample.keys():
            #  in active sample, must be n_sample cases
            if self.active_sample[sample_id].generation != rollout_req.generation:
                # stale generation,skip this
                print(f"[MicroBatchChatCompletionScheduler] _consumer process get sample, stage: {stage}, skip active sample, sample_id: {sample_id}")
                return True
        else:
            # not in pending and active, should be a finished one but with stale generation
            # TODO better sanity check
            print(f"[MicroBatchChatCompletionScheduler] _consumer process get sample, stage: {stage}, sample_id: {sample_id} not in pending and active")
            return True
        return False

    async def stream_handle_rollout_req(self, addr, reduce_queue: asyncio.Queue, external_call: AsyncCallbackMixin, actor_meta: ActorMeta, rollout_req: RolloutReq):
        from verl.workers.rollout.chat_scheduler.requests import chat_completions_aiohttp

        if self._skip_if_stale_request(rollout_req, stage="handle_rollout_req"):
            return
        logger.debug(f"[MicroBatchChatCompletionScheduler] _consumer process get sample, addr: {addr}, actor_meta: {actor_meta}")
        request_id = uuid4().hex
        completions, exception, message = None, None, {}
        messages = rollout_req.messages
        try:
            # NOTE: OpenAI client uses httpx, seems to have performance issue in high concurrency requests.
            logger.debug(f"[MicroBatchChatCompletionScheduler] _consumer process get sample, submit to engine {addr}，sample_id: {rollout_req.sample_id}")
            completions = await chat_completions_aiohttp(
                address=addr,
                messages=messages,
                tools=rollout_req.tools_schema,
                extra_body=rollout_req.extra_body,
                extra_headers={"x-request-id": request_id},
                **rollout_req.sampling_params,
            )
            message = completions.choices[0].message.model_dump(exclude_unset=True, exclude_none=True)
        except asyncio.CancelledError as cancel_err:
            logger.debug(f"chat completion failed with exception: {cancel_err}")
            raise cancel_err
        except Exception as e:
            logger.warning(f"chat completion failed with exception: {e}")
            exception = e
        logger.debug(f"[MicroBatchChatCompletionScheduler] _consumer process get sample done,meesage: {message}", actor_meta)
        if "content" not in message:
            message["content"] = ""
        messages.append(message)
        resp = RolloutResp(request=rollout_req, completions=completions, exception=exception, req_id=request_id, messages=messages)
        try:
            if external_call.hit(resp):
                logger.debug(f"[id={completions.id},turn={len(messages)},finish_reason={completions.choices[0].finish_reason}] Call tools")
                external_call.put(CallsReq(rollout_resp=resp, actor_meta=actor_meta))
            else:
                logger.debug(f"[MicroBatchChatCompletionScheduler] _consumer process put sample to reduce_queue,idx: {actor_meta.actor_id}")
                reduce_queue.put_nowait(resp)
        except Exception as e:
            logger.warning(f"[MicroBatchChatCompletionScheduler] _consumer process put sample to reduce_queue failed,idx: {actor_meta.actor_id}, exception: {e}")
            resp.exception = e
            reduce_queue.put_nowait(resp)
        # print("[MicroBatchChatCompletionScheduler] _consumer process done")

    # maybe we can make this sink_queue as a pubsub proxy using zmq
    async def stream_handle_reduce_req(self, batch_size, n_sample, sink_queue: asyncio.Queue = None, format="ReduceResp"):
        batch_conversations = []
        # joiner_buffer worked as key for sample_id,value for result.
        # make sure n-sample arrived correctlly then ship to batch_conversations as result
        joiner_buffer: Dict[str, List[List[Dict[str, str]]]] = {}
        gen_batch_proto_list = []
        batch_proto_list = []
        counter = 0
        print_rate = 0.1
        _print_div = int(batch_size * 0.1)
        print(f"[stream_handle_reduce_req] _gather_result launch, current queue size: {self.reduce_data_queue.qsize()}, batch_size: {batch_size}, print_rate: {print_rate}")
        while counter < batch_size:
            if counter % _print_div == 0:
                print(f"[stream_handle_reduce_req] _gather_result counter: {counter}")
            rollout_resp: RolloutResp = await self.reduce_data_queue.get()
            if self._skip_if_stale_request(rollout_resp.request, stage="handle_reduce_req"):
                continue
            if sink_queue is not None:
                sink_queue.put(rollout_resp)
            sample_id = rollout_resp.request.sample_id
            if rollout_resp.exception is not None:
                # maybe skip or requeue?  error handling issue
                # should be handled by supervisor
                raise rollout_resp.exception
            if sample_id in joiner_buffer.keys():
                joiner_buffer[sample_id].append(rollout_resp.messages)
            else:
                joiner_buffer[sample_id] = [rollout_resp.messages]
            if len(joiner_buffer[sample_id]) == n_sample:
                _sample = self.active_sample.pop(sample_id)
                if format == "ReduceResp":
                    logger.debug(f"finished for samples: {sample_id}")
                    batch_conversations.append(ReduceResp(raw_prompt=rollout_resp.request.raw_prompt, messages=joiner_buffer[sample_id]))
                else:
                    batch_conversations.extend(joiner_buffer[sample_id])
                gen_batch_proto_list.append(_sample.gen_batch)
                batch_proto_list.append(_sample.batch)
                counter += 1
        print("[MicroBatchChatCompletionScheduler] _gather_result done for one batch，do collact function")
        batch = concat_data_proto(batch_proto_list)
        gen_batch = concat_data_proto(gen_batch_proto_list)
        return batch_conversations, gen_batch, batch

    async def generate_sequences(self, batch, **sampling_params):
        raise NotImplementedError

    def _requeue_preempt_req(self):
        print(f"ready to requeue active samples {len(self.active_sample)}")
        for sample_id in list(self.active_sample.keys()):  # 避免 RuntimeError
            _sample: _Sample = self.active_sample.pop(sample_id)
            _sample.generation += 1
            self.pending_sample[sample_id] = _sample
            for _ in range(self.config.n):
                req: RolloutReq = deepcopy(_sample.rollout_req)
                req.verl_session_id = uuid4().hex
                req.generation = _sample.generation
                self.global_data_queue.put_nowait(req)
        assert len(self.active_sample) == 0

    async def stream_generate_sequences(self, data_iter: Iterable, batch_size: int, renew=False) -> Tuple[bool, DataProto, DataProto, DataProto]:
        self.buffer_size = batch_size
        self._lazy_init_global_resource(data_iter, renew)
        self.wake_up_engine_actor()
        # detect wether there is any active request
        # they might be in tool-calls queue,
        pending_sample_length = len(self.pending_sample)
        if self._data_fetcher_done() and pending_sample_length == 0:
            return True, None, None, None
        last_batch, bsz = self.last_batch(self.buffer_size)
        if last_batch:
            last_batch = True
            self.buffer_size = bsz
            print(f"last batch for epoch, size: {bsz}")
        print(f"waiting for rollout done, self.buffer_size: {self.buffer_size}")
        batch_conversations, gen_batch, batch = await self.reduce_handler(self.buffer_size, n_sample=self.config.n, format=self.reduce_format)
        print(f"partial rollout done, cancel all left request, real size: {len(batch_conversations)}")
        await self.cancel_all_req()
        self._requeue_preempt_req()
        self.done_sample_counter += len(batch_conversations)
        print(f"[MicroBatchChatCompletionScheduler] generate_sequences done with {len(batch_conversations)} samples, done_sample_counter: {self.done_sample_counter}")
        gen_batch_output = self.completion_callback.new_postprocess(batch_conversations, n=self.config.n)
        self.last_batch_sanity_check(last_batch)
        return last_batch, gen_batch_output, gen_batch, batch

    def last_batch(self, expect_buffer_size) -> Tuple[bool, int]:
        # how do we know whether the elements in data_iter plus active_samples is less than buffer_size?
        if self.data_iter_length - self.done_sample_counter < expect_buffer_size:
            return True, self.data_iter_length - self.done_sample_counter
        else:
            return False, expect_buffer_size

    def last_batch_sanity_check(self, is_last_batch):
        if not is_last_batch:
            return
        assert self.done_sample_counter == self.data_iter_length
        assert len(self.pending_sample) == 0
        assert len(self.active_sample) == 0
        assert self.global_data_queue.empty()
        assert self.reduce_data_queue.empty()
