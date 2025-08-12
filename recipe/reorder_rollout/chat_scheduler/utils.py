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
import functools
import logging
import os
from abc import abstractmethod
from dataclasses import dataclass
from heapq import heapify, heappop, heappush
from typing import Any, Optional, Protocol

import numpy as np
import torch
from tensordict import TensorDict

from verl.experimental.agent_loop.agent_loop import AgentLoopBase
from verl.experimental.agent_loop.utils import AgentLoopOutput
from verl.protocol import DataProto

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_QUEUE_LOGGING_LEVEL", "INFO"))


# copyed from http://code.activestate.com/recipes/522995-priority-dict-a-priority-queue-with-updatable-prio/


class priority_dict(dict):
    """Dictionary that can be used as a priority queue.
    Keys of the dictionary are items to be put into the queue, and values
    are their respective priorities. All dictionary methods work as expected.
    The advantage over a standard heapq-based priority queue is
    that priorities of items can be efficiently updated (amortized O(1))
    using code as 'thedict[item] = new_priority.'
    The 'smallest' method can be used to return the object with lowest
    priority, and 'pop_smallest' also removes it.
    The 'sorted_iter' method provides a destructive sorted iterator.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rebuild_heap()

    def _rebuild_heap(self):
        self._heap = [(v, k) for k, v in self.items()]
        heapify(self._heap)

    def smallest(self):
        """Return the item with the lowest priority.
        Raises IndexError if the object is empty.
        """

        heap = self._heap
        v, k = heap[0]
        while k not in self or self[k] != v:
            heappop(heap)
            v, k = heap[0]
        return k

    def pop_smallest(self):
        """Return the item with the lowest priority and remove it.
        Raises IndexError if the object is empty.
        """

        heap = self._heap
        v, k = heappop(heap)
        while k not in self or self[k] != v:
            v, k = heappop(heap)
        del self[k]
        return k

    def __setitem__(self, key, val):
        # We are not going to remove the previous value from the heap,
        # since this would have a cost O(n).

        super().__setitem__(key, val)

        if len(self._heap) < 2 * len(self):
            heappush(self._heap, (val, key))
        else:
            # When the heap grows larger than 2 * len(self), we rebuild it
            # from scratch to avoid wasting too much memory.
            self._rebuild_heap()

    def setdefault(self, key, val):
        if key not in self:
            self[key] = val
            return val
        return self[key]

    def update(self, *args, **kwargs):
        # Reimplementing dict.update is tricky -- see e.g.
        # http://mail.python.org/pipermail/python-ideas/2007-May/000744.html
        # We just rebuild the heap from scratch after passing to super.

        super().update(*args, **kwargs)
        self._rebuild_heap()

    def sorted_iter(self):
        """Sorted iterator of the priority dictionary items.
        Beware: this will destroy elements as they are returned.
        """

        while self:
            yield self.pop_smallest()


class QueueGroup:
    def __init__(self, num_queues: int, queus: list[asyncio.Queue]):
        # this queue warpper for work stealing
        # the heap is maintained as a max-heap, so we can steal work from the longest queue
        # this utility is not thread-safe, please use it in a single thread
        self._queues: list[asyncio.Queue] = queus if queus else [asyncio.Queue() for _ in range(num_queues)]
        self._heap = priority_dict({i: 0 for i in range(num_queues)})

    def __getitem__(self, idx: int) -> asyncio.Queue:
        return self._queues[idx]

    def __len__(self):
        return len(self._queues)

    def push(self, idx: int, item: Any):
        # we don't need lock here, because we are using a single thread
        self._queues[idx].put_nowait(item)
        # this heap return the smallest value, so we need to minus 1
        self._heap[idx] -= 1
        logger.debug(f"push {item} to queue {idx},heap: {self._heap}")

    async def pop(self, idx: int) -> Any:
        try:
            item = await self._queues[idx].get()
            # this heap return the smallest value, so we need to plus 1
            self._heap[idx] += 1
            return item
        except asyncio.CancelledError as err:
            logger.debug(f"pop from queue {idx} cancelled")
            raise asyncio.CancelledError(f"pop from queue {idx} cancelled") from err

    def pop_nowait(self, idx: int) -> Any:
        item = self._queues[idx].get_nowait()
        # this heap return the smallest value, so we need to plus 1
        self._heap[idx] += 1
        return item

    def pop_from_longest(self) -> Any:
        try:
            idx = self._heap.smallest()
            item = self._queues[idx].get_nowait()
            self._heap[idx] += 1
            logger.debug(f"pop {item} from queue {idx},heap: {self._heap}")
            return item
        except asyncio.QueueEmpty:
            logger.debug("heap is empty")
            # none of the queues are non-empty
        return None


class Message:
    pass


@dataclass
class ActorMeta:
    actor_id: int
    queue_group: QueueGroup
    local_id: int


@dataclass
class DeathLetter:
    actor_meta: ActorMeta
    async_task: asyncio.Task


class WorkFunc(Protocol):
    @abstractmethod
    async def __call__(self, meta: ActorMeta, message: Message) -> None:
        pass


class WorkStealingActor:
    def __init__(
        self,
        worker_id: int,
        local_id: int,
        local_queues: QueueGroup,
        global_queue: asyncio.Queue,
        work_fn: WorkFunc,
        enable_work_stealing: bool = True,
        death_sigal: asyncio.Queue = None,
        sink_queue: asyncio.Queue = None,
    ):
        self.sink_queue = sink_queue
        self.worker_id = worker_id
        self.local_id = local_id
        self.local_queues = local_queues
        self.global_queue = global_queue
        self.func = work_fn
        self.death_signal = death_sigal
        self.total_workers = len(local_queues)
        self.actor_meta = ActorMeta(worker_id, self.local_queues, self.local_id)
        self.enable_work_stealing = enable_work_stealing
        self._init_actor_coro()
        self.cur_task = None
        self.queue_task = []
        self.blocker: asyncio.Event = asyncio.Event()
        self.shutdown_done = asyncio.Event()
        self.wakeup()

    async def run(self):
        while True:
            try:
                task = await self.global_queue.get()
                await self.blocker.wait()
                coros = self.func(self.actor_meta, task)
                self.cur_task: asyncio.Task = asyncio.ensure_future(coros)
                await self.cur_task
            except asyncio.CancelledError:
                # this means the work function didn't hanlde the canceled error
                # we need to cancel the task
                if self.sink_queue is not None:
                    self.sink_queue.put_nowait(task)
                # requeue since err like this means coros not execute at all
                print(f"[WorkStealingActor] run, cancel task, task: {task}")
                self.global_queue.put_nowait(task)
            except Exception as e:
                import traceback

                traceback.print_exc()
                print(f"Fatal Error: task {task} failed, actor_meta: {self.actor_meta}, error: {e}")
            finally:
                self.cur_task = None
                self.global_queue.task_done()
                logger.debug(f"finish task, actor_meta: {self.actor_meta}")

    def _init_actor_coro(self):
        if self.death_signal is not None:

            def callback(task):
                if task.exception() is not None:
                    letter = DeathLetter(
                        actor_meta=self.actor_meta,
                        async_task=task,
                    )
                    self.death_signal.put_nowait(letter)

            self.coro = asyncio.create_task(self.run())
            self.coro.add_done_callback(callback)
        else:
            self.coro = asyncio.create_task(self.run())

    def cancel_task(self):
        # set blocker
        self.blocker.clear()
        evt = asyncio.Event()
        if self.cur_task is not None and not (self.cur_task.done() or self.cur_task.cancelled()):
            self.cur_task.add_done_callback(functools.partial(self._set_evt, evt=evt))
            self.cur_task.cancel()
        else:
            evt.set()
        return evt

    def wakeup(self):
        self.blocker.set()

    def _set_evt(self, task, evt: asyncio.Event):
        evt.set()


def list_of_dict_to_dict_of_list_nd_array(list_of_dict: list[dict]):
    if len(list_of_dict) == 0:
        return {}
    keys = list_of_dict[0].keys()
    output = {key: [] for key in keys}
    for data in list_of_dict:
        for key, item in data.items():
            assert key in output
            assert isinstance(item, np.ndarray)
            assert len(item) == 1
            output[key].append(item.tolist()[0])
    return output


def concat_data_proto(data: list["DataProto"]) -> "DataProto":
    """Concat a list of DataProto. The batch is concatenated among dim=0.
    The meta_info is assumed to be identical and will use the first one.

    Args:
        data (List[DataProto]): list of DataProto

    Returns:
        DataProto: concatenated DataProto
    """
    batch_lst = []
    non_tensors = {}
    for batch in data:
        batch_lst.append(batch.batch)

    new_batch = torch.cat(batch_lst, dim=0) if batch_lst[0] is not None else None

    non_tensor_batch = list_of_dict_to_dict_of_list_nd_array(list_of_dict=[d.non_tensor_batch for d in data])

    for key, val in non_tensor_batch.items():
        non_tensors[key] = np.array(val, dtype=object)
    cls = type(data[0]) if len(data) > 0 else DataProto
    return cls(batch=new_batch, non_tensor_batch=non_tensors, meta_info=data[0].meta_info)


# copy from agent-loop
def get_agent_loop_class(agent_name: str) -> type[AgentLoopBase]:
    # TODO: add tool agent registrary
    from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
    from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop

    if agent_name == "single_turn_agent":
        return SingleTurnAgentLoop
    elif agent_name == "tool_agent":
        return ToolAgentLoop
    raise ValueError(f"Unknown agent_name: {agent_name}")


class _MgrProxy:
    def __init__(self, routing_method):
        self.router = routing_method
        self.ray_awaitable = None
        self.server_handle = None
        self.request_id = None

    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
    ) -> list[int]:
        self.request_id = request_id
        self.server_handle = self.router(request_id)
        self.ray_awaitable = self.server_handle.generate_with_cancel.remote(
            request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params, image_data=image_data
        )
        result = await self.ray_awaitable
        self.ray_awaitable = None
        return result


def agent_loop_postprocess(
    tokenizer, inputs: list[AgentLoopOutput], max_prompt_length: int, max_response_length: int
) -> DataProto:
    # NOTE: consistent with batch version of generate_sequences in vllm_rollout_spmd.py
    # prompts: left pad
    # responses: right pad
    # input_ids: prompt + response
    # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
    # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]

    # prompts
    tokenizer.padding_side = "left"
    outputs = tokenizer.pad(
        [{"input_ids": input.prompt_ids} for input in inputs],
        padding="max_length",
        # max_length=self.config.actor_rollout_ref.rollout.prompt_length,
        max_length=max_prompt_length,
        return_tensors="pt",
        return_attention_mask=True,
    )
    prompt_ids, prompt_attention_mask = outputs["input_ids"], outputs["attention_mask"]

    # responses
    tokenizer.padding_side = "right"
    outputs = tokenizer.pad(
        [{"input_ids": input.response_ids} for input in inputs],
        padding="max_length",
        max_length=max_response_length,
        return_tensors="pt",
        return_attention_mask=True,
    )
    response_ids, response_attention_mask = outputs["input_ids"], outputs["attention_mask"]

    # response_mask
    outputs = tokenizer.pad(
        [{"input_ids": input.response_mask} for input in inputs],
        padding="max_length",
        max_length=max_response_length,
        return_tensors="pt",
        return_attention_mask=False,
    )
    response_mask = outputs["input_ids"]
    assert response_ids.shape == response_mask.shape, (
        f"mismatch in response_ids and response_mask shape: {response_ids.shape} vs {response_mask.shape}"
    )
    response_mask = response_mask * response_attention_mask

    input_ids = torch.cat([prompt_ids, response_ids], dim=1)
    attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=1)
    position_ids = (attention_mask.cumsum(dim=1) - 1) * attention_mask

    batch = TensorDict(
        {
            "prompts": prompt_ids,  # [bsz, prompt_length]
            "responses": response_ids,  # [bsz, response_length]
            "response_mask": response_mask,  # [bsz, response_length]
            "input_ids": input_ids,  # [bsz, prompt_length + response_length]
            "attention_mask": attention_mask,  # [bsz, prompt_length + response_length]
            "position_ids": position_ids,  # [bsz, prompt_length + response_length]
        },
        batch_size=len(input_ids),
    )

    num_turns = np.array([input.num_turns for input in inputs], dtype=np.int32)
    metrics = [input.metrics.model_dump() for input in inputs]
    return DataProto(batch=batch, non_tensor_batch={"__num_turns__": num_turns}, meta_info={"metrics": metrics})
