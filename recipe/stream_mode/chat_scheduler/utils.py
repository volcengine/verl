import asyncio
import functools
import logging
import os
from abc import abstractmethod
from dataclasses import dataclass
from heapq import heapify, heappop, heappush
from typing import Any, Dict, List, Protocol, Type

import numpy as np
import torch

from verl.experimental.agent_loop.agent_loop import AgentLoopBase
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
    def __init__(self, num_queues: int, queus: List[asyncio.Queue]):
        # this queue warpper for work stealing
        # the heap is maintained as a max-heap, so we can steal work from the longest queue
        # this utility is not thread-safe, please use it in a single thread
        self._queues: List[asyncio.Queue] = queus if queus else [asyncio.Queue() for _ in range(num_queues)]
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
    def __init__(self, worker_id: int, local_id: int, local_queues: QueueGroup, global_queue: asyncio.Queue, work_fn: WorkFunc, enable_work_stealing: bool = True, death_letter: asyncio.Queue = None):
        self.worker_id = worker_id
        self.local_id = local_id
        self.local_queues = local_queues
        self.global_queue = global_queue
        self.func = work_fn
        self.death_letter = death_letter
        self.total_workers = len(local_queues)
        self.actor_meta = ActorMeta(worker_id, self.local_queues, self.local_id)
        self.enable_work_stealing = enable_work_stealing
        self.queues_to_wait = self._build_priority_queue_list()
        self.coro = self._init_actor_coro()
        self.cur_task = None
        self.queue_task = []
        self.blocker: asyncio.Event = asyncio.Event()
        self.shutdown_flag = False
        self.shutdown_done = asyncio.Event()
        self.wakeup()

    def _build_priority_queue_list(self):
        base_queue = [functools.partial(self.local_queues.pop, self.worker_id), self.global_queue.get]
        if self.enable_work_stealing:
            return base_queue + [functools.partial(self.local_queues.pop, i) for i, _ in enumerate(self.local_queues) if i != self.worker_id]
        else:
            return base_queue

    async def run(self):
        # logger.info(f"start worker for actor, actor_meta: {self.actor_meta}")
        while not self.shutdown_flag:
            try:
                task = await self.get_task()
                logger.debug(f"get task {task} from queue, actor_meta: {self.actor_meta}")
                coros = self.func(self.actor_meta, task)
                self.cur_task: asyncio.Task = asyncio.ensure_future(coros)
                await self.cur_task
            except asyncio.CancelledError:
                # this means the work function didn't hanlde the canceled error
                # we need to cancel the task
                logger.debug(f"cancel task {task}, actor_meta: {self.actor_meta}")
            except Exception as e:
                logger.warning(f"Fatal Error: task {task} failed, actor_meta: {self.actor_meta}, error: {e}")
            finally:
                self.cur_task = None
                logger.debug(f"finish task, actor_meta: {self.actor_meta}")
        self.shutdown_done.set()
        logger.info(f"shutdown done with actor meta: {self.actor_meta}")

    def _init_actor_coro(self):
        if self.death_letter is not None:

            def callback(task):
                if task.exception() is not None:
                    letter = DeathLetter(
                        actor_meta=self.actor_meta,
                        async_task=task,
                    )
                    self.death_letter.put_nowait(letter)

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
            # logger.debug(self.cur_task.print_stack())
            self.cur_task.cancel()
        else:
            evt.set()
        return evt

    def wakeup(self):
        self.blocker.set()

    def _set_evt(self, task, evt: asyncio.Event):
        evt.set()

    async def shutdown(self):
        logger.info(f"ready to shut down actor: {self.actor_meta}")
        self.shutdown_flag = True
        event = []
        task_to_cancel = self.queue_task
        task_to_cancel.append(self.coro)
        if self.cur_task is not None:
            logger.info(f"append cur task to cancel: {self.cur_task}")
            task_to_cancel.append(self.cur_task)
        for task in task_to_cancel:
            if task is not None and (not task.done() or not task.cancelled()):
                evt = asyncio.Event()
                event.append(evt.wait())
                task.add_done_callback(functools.partial(self._set_evt, evt=evt))
                task.cancel()
        logger.debug("wait for done callback with length: ", self.actor_meta, len(event), flush=True)
        await asyncio.gather(*event)
        logger.debug("shutdown done for actor meta: ", self.actor_meta, flush=True)

    async def get_task(self):
        async def _get_task():
            try:
                logger.debug(f"try to get task from local queue, actor meta: {self.actor_meta}")
                return self.local_queues.pop_nowait(self.worker_id)
            except asyncio.QueueEmpty:
                pass
            try:
                logger.debug("try to get task from global queue")
                return self.global_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            if self.enable_work_stealing:
                logger.debug("try to steal task from other queues")
                task = self.local_queues.pop_from_longest()
                if task is not None:
                    return task

            self.queue_task = [asyncio.create_task(func()) for func in self.queues_to_wait]
            logger.debug(f"wait for task from all queue, actor_meta: {self.actor_meta}")
            done, _ = await asyncio.wait(self.queue_task, return_when=asyncio.FIRST_COMPLETED)
            for task in self.queue_task:
                if not task.done():
                    task.cancel()

            return list(done)[0].result()

        task = await _get_task()
        await self.blocker.wait()
        logger.debug(f"blocker acquire with actor_meta: {self.actor_meta},task: {task.sample_id}, generation: {task.generation}")
        return task


def list_of_dict_to_dict_of_list_nd_array(list_of_dict: list[dict]):
    if len(list_of_dict) == 0:
        return {}
    keys = list_of_dict[0].keys()
    output = {key: [] for key in keys}
    for data in list_of_dict:
        for key, item in data.items():
            assert key in output
            assert isinstance(item, np.ndarray)
            # import pytest
            # pytest.set_trace()
            assert len(item) == 1
            # unpack this
            output[key].append(item.tolist()[0])
    return output


def concat_data_proto(data: List["DataProto"]) -> "DataProto":
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
    # import pytest
    # pytest.set_trace()
    return cls(batch=new_batch, non_tensor_batch=non_tensors, meta_info=data[0].meta_info)


# copy from agent-loop
def get_agent_loop_class(agent_name: str) -> Type[AgentLoopBase]:
    # TODO: add tool agent registrary
    from verl.experimental.agent_loop.single_agent_loop import SingleTurnAgentLoop
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
        prompt_ids: List[int],
        sampling_params: Dict[str, Any],
    ) -> List[int]:
        self.request_id = request_id
        self.server_handle = self.router(request_id)
        self.ray_awaitable = self.server_handle.generate_with_cancel.remote(request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params)
        result = await self.ray_awaitable
        self.ray_awaitable = None
        return result
