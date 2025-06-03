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
from copy import deepcopy
from dataclasses import dataclass
from queue import Queue
from typing import Any, Dict, List

import torch
from openai.types.chat.chat_completion import ChatCompletion
from tensordict import TensorDict

from verl.protocol import DataProto
from verl.workers.rollout.async_server import ChatCompletionScheduler


class NaiveChatCompletionScheduler(ChatCompletionScheduler):
    """
    A very naive implementation of ChatCompletionScheduler for demo purpose,
    only do single-turn chat completion.
    """

    async def generate_sequences(self, batch: DataProto, **sampling_params) -> DataProto:
        kwargs = dict(
            n=self.config.n,
            max_completion_tokens=self.config.response_length,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )

        do_sample = batch.meta_info.get("do_sample", True)
        is_validate = batch.meta_info.get("validate", False)
        if not do_sample or is_validate:
            kwargs["n"] = 1
            kwargs["temperature"] = 0

        kwargs.update(sampling_params)
        print(f"[NaiveChatCompletionScheduler] generate_sequences sampling params: {kwargs}")

        async def callback(completions: ChatCompletion, info: Dict[str, Any], exception: Exception):
            assert exception is None, f"exception: {exception}"
            conversation, batch_conversations, batch_index = (
                info["conversation"],
                info["batch_conversations"],
                info["batch_index"],
            )

            conversations = []
            for choice in completions.choices:
                chat = conversation.copy()
                chat.append({"role": choice.message.role, "content": choice.message.content})
                conversations.append(chat)
            batch_conversations[batch_index] = conversations

            # NOTE: we can call tools and resubmit chat completions here.
            # call_tools(completions, info)
            # await self.submit_chat_completions(callback2, ...)

        # TODO: we may need to control max concurrent requests here, or it will harm prefix cache hit rate.
        tasks, batch_conversations = [], [None] * len(batch)
        for batch_index, conversation in enumerate(batch.non_tensor_batch["raw_prompt"]):
            # raw_prompt: [{"role": "user", "content": ""}, ["role": "assistant", "content"], ...]
            tasks.append(
                asyncio.create_task(
                    self.submit_chat_completions(
                        callback=callback,
                        callback_additional_info={
                            "batch_conversations": batch_conversations,
                            "batch_index": batch_index,
                            "conversation": list(conversation),
                        },
                        model=self.model_name,
                        messages=conversation.tolist(),
                        **kwargs,
                    )
                )
            )
        await asyncio.gather(*tasks)
        print("[NaiveChatCompletionScheduler] generate_sequences done")

        return self._postprocess(batch, batch_conversations, kwargs["n"])

    def _postprocess(self, batch: DataProto, batch_conversations: List[List[List[Dict[str, str]]]], n: int) -> DataProto:
        # NOTE: consistent with batch version of generate_sequences in vllm_rollout_spmd.py
        # prompts: left pad
        # responses: right pad
        # input_ids: prompt + response
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]

        # prompts: [prompt] from input dataset
        prompts = [self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False) for prompt in batch.non_tensor_batch["raw_prompt"]]

        # flatten batch_conversations if n > 1
        assert len(batch_conversations) == len(prompts)
        batch_conversations = [conversation for conversations in batch_conversations for conversation in conversations]
        assert len(batch_conversations) == len(prompts) * n

        # sequences: [prompt + response]
        sequences = [self.tokenizer.apply_chat_template(conversation, add_generation_prompt=False, tokenize=False) for conversation in batch_conversations]

        # responses: [response]
        # TODO: mask out tools calling tokens?
        responses = [sequence[len(prompts[i // n]) :] for i, sequence in enumerate(sequences)]

        prompts = self.tokenizer(prompts, return_tensors="pt", padding="longest", padding_side="left")
        responses = self.tokenizer(responses, return_tensors="pt", padding="longest", padding_side="right")
        if n > 1:
            prompts["input_ids"] = prompts["input_ids"].repeat_interleave(n, dim=0)
            prompts["attention_mask"] = prompts["attention_mask"].repeat_interleave(n, dim=0)

        input_ids = torch.cat([prompts["input_ids"], responses["input_ids"]], dim=1)
        attention_mask = torch.cat([prompts["attention_mask"], responses["attention_mask"]], dim=1)
        position_ids = (attention_mask.cumsum(dim=1) - 1) * attention_mask

        batch = TensorDict(
            {
                "prompts": prompts["input_ids"],
                "responses": responses["input_ids"],
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=len(input_ids),
        )

        return DataProto(batch=batch)


@dataclass
class RolloutSample:
    completions: ChatCompletion
    info: Dict[str, Any]
    model_name: str
    conversation: List[Dict[str, str]]
    chat_complete_request: Dict[str, Any]
    exceptoin: Exception


class MicroBatchChatCompletionScheduler(NaiveChatCompletionScheduler):
    def __init__(self, config, model_path, server_addresses, max_cache_size=10000, max_inflight_req=8):
        super().__init__(config, model_path, server_addresses, max_cache_size)
        self.send_queue = asyncio.Queue()
        self.reduce_queue = asyncio.Queue()
        self.loop = asyncio.get_event_loop()
        self.max_inflight_req = max_inflight_req
        self.server_addresses = server_addresses
        self.proxy_agents_coros = self._init_proxy_group(self.server_addresses, self.send_queue, self.reduce_queue, self.max_inflight_req)
        print(self.proxy_agents_coros)

    def _init_proxy_group(self, addrs: List[str], send_queue: Queue, reduce_queue: Queue, max_inflight_req=4):
        # we use a group of coroutine to consume send_queue and produce reduce_queue
        # since the asyncio.Queue is not thread safe.
        # ideadly we have 1 get_elements coroutine to get element from send_queue and put to local_queue
        # max_inflight_req consumer coroutine to get element from local_queue and submit to vllm
        coros = []
        for addr in addrs:
            semaphore = asyncio.Semaphore(max_inflight_req)
            local_queue = asyncio.Queue(max_inflight_req)
            coros.append(self.get_element(addr, send_queue, local_queue, semaphore))
            coros.extend([self.process(local_queue, reduce_queue, semaphore, addr, i) for i in range(max_inflight_req)])
        for coro in coros:
            self.loop.create_task(coro)
        return coros

    async def get_element(self, server_addr, send_queue, local_queue, semaphore: asyncio.Semaphore):
        print("[MicroBatchChatCompletionScheduler] _consumer get_element start with idx: ", server_addr)
        while True:
            await semaphore.acquire()
            sample: RolloutSample = await send_queue.get()
            print("[MicroBatchChatCompletionScheduler] _consumer get_element get sample, put to local_queue", server_addr)
            await local_queue.put(sample)

    async def process(self, local_queue, reduce_queue, semaphore: asyncio.Semaphore, addr, idx):
        print("[MicroBatchChatCompletionScheduler] _consumer process start with idx: ", idx)
        while True:
            sample: RolloutSample = await local_queue.get()
            _sample = deepcopy(sample)
            print("[MicroBatchChatCompletionScheduler] _consumer process get sample, submit to vllm", addr)
            try:

                async def callback(completions, info, exception):
                    if exception is not None:
                        print("[MicroBatchChatCompletionScheduler] _consumer process callback get exception", idx)
                        await reduce_queue.put(RolloutSample(completions, info, None, None, None, exceptoin=exception))
                    else:
                        print("[MicroBatchChatCompletionScheduler] _consumer process callback get completions", idx)
                        await reduce_queue.put(RolloutSample(completions, info, None, None, None, exceptoin=None))

                await self.submit_chat_completions(callback=callback, address=addr, callback_additional_info=sample.info, model=sample.model_name, messages=sample.conversation, **sample.chat_complete_request)
                print("[MicroBatchChatCompletionScheduler] _consumer process submit to vllm done", idx)
            except Exception as e:
                print("[MicroBatchChatCompletionScheduler] _consumer process exception", idx, e)
                await reduce_queue.put(RolloutSample(None, sample.info, None, None, exceptoin=e))
            finally:
                semaphore.release()

    async def _gather_result(self, batch_size):
        batch_conversations = [None] * batch_size
        counter = 0
        while counter < batch_size:
            print("[MicroBatchChatCompletionScheduler] _gather_result counter: ", counter)
            sample: RolloutSample = await self.reduce_queue.get()
            counter += 1
            if sample.exceptoin is not None:
                # assert exception is None, f"exception: {exception}"
                raise sample.exceptoin
            conversation, batch_index = (
                sample.info["conversation"],
                sample.info["batch_index"],
            )
            conversations = []
            for choice in sample.completions.choices:
                chat = conversation.copy()
                chat.append({"role": choice.message.role, "content": choice.message.content})
                conversations.append(chat)
            batch_conversations[batch_index] = conversations
        return batch_conversations

    async def generate_sequences(self, batch: DataProto, **sampling_params) -> DataProto:
        kwargs = dict(
            n=self.config.n,
            max_completion_tokens=self.config.response_length,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )
        do_sample = batch.meta_info.get("do_sample", True)
        is_validate = batch.meta_info.get("validate", False)
        if not do_sample or is_validate:
            kwargs["n"] = 1
            kwargs["temperature"] = 0

        kwargs.update(sampling_params)
        print(f"[NaiveChatCompletionScheduler] generate_sequences sampling params: {kwargs}")
        for batch_index, conversation in enumerate(batch.non_tensor_batch["raw_prompt"]):
            await self.send_queue.put(
                RolloutSample(
                    completions=None,
                    info={
                        "batch_index": batch_index,
                        "conversation": list(conversation),
                    },
                    conversation=conversation.tolist(),
                    model_name=self.model_name,
                    chat_complete_request=kwargs,
                    exceptoin=None,
                )
            )
        batch_conversations = await self._gather_result(len(batch))
        print("[NaiveChatCompletionScheduler] generate_sequences done")

        return self._postprocess(batch, batch_conversations, kwargs["n"])
