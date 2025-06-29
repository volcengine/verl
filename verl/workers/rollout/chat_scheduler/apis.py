import asyncio
import functools
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import numpy as np
from openai.types.chat.chat_completion import ChatCompletion

from verl.workers.rollout.chat_scheduler.utils import ActorMeta


@dataclass
class RolloutReq:
    # this works for tool-calling, indicate for one-multi-turns request
    verl_session_id: Optional[str]
    # sample_id works for n-samples, n replicated requests share the same sample_id
    sample_id: Optional[str]
    model_name: str
    messages: List[Dict[str, str]]
    tools_schema: List[Dict[str, Any]]
    sampling_params: Dict[str, Any]
    extra_body: Dict[str, Any]
    raw_prompt: np.ndarray
    # maybe we can count the requeue times
    generation: int = 0


@dataclass
class RolloutResp:
    request: RolloutReq
    completions: ChatCompletion
    exception: Optional[Exception] = None
    req_id: str = None
    messages: List[Dict[str, str]] = None


@dataclass
class CallsReq:
    rollout_resp: RolloutResp
    actor_meta: ActorMeta


@dataclass
class ReduceResp:
    raw_prompt: np.ndarray
    messages: List[List[Dict[str, str]]]


@runtime_checkable
class AsyncCallbackMixin(Protocol):
    def put(self, req: RolloutResp) -> bool:
        # this method will be called in coroutine
        # this method should act as a message queue
        ...

    def hit(self, req: RolloutResp) -> bool:
        # make sure this is a short function
        # this will be run in a coroutine
        ...

    async def shutdown(self): ...

    async def cancel(self): ...

    async def wake_up(self): ...

    def init_plugin_callers(self): ...


class CoroExternalCallsPlugin(AsyncCallbackMixin):
    def __init__(self, num_workers=3, death_letter: Optional[asyncio.Queue] = None):
        self.plugin_queue = asyncio.Queue()
        self.num_workers = num_workers
        self.shut_down_flag = False
        self.shutdown_evt = asyncio.Event()
        self.death_letter: Optional[asyncio.Queue] = death_letter

    def init_plugin_callers(self):
        self.shut_down_flag = False
        if self.death_letter is not None:
            pass
            # def callback(task):
            #     if task.exception() is not None:
            #         letter = DeathLetter(
            #             actor_meta=self.actor_meta,
            #             async_task=task,
            #         )
            #         self.death_letter.put_nowait(letter)
            # self.coro = asyncio.create_task(self.run())
            # self.coro.add_done_callback(callback)
        else:
            print("init plugin callers for CoroExternalCallsPlugin with worker: ", self.num_workers)
            self.coros = [asyncio.create_task(self.run()) for _ in range(self.num_workers)]

    def wake_up(self):
        self.init_plugin_callers()

    def put(self, req: RolloutResp) -> bool:
        self.plugin_queue.put_nowait(req)

    async def shutdown(self):
        self.shut_down_flag = True

        def set_evt(task: asyncio.Task, evt: asyncio.Event):
            evt.set()

        evts = []
        for coro in self.coros:
            if not coro.done() or not coro.cancelled():
                evt = asyncio.Event()
                evts.append(evt.wait())
                coro.add_done_callback(functools.partial(set_evt, evt=evt))
                coro.cancel()
        print("waiting for CoroExternalCallsPlugin to shutdown, with length: ", len(evts))
        await asyncio.gather(*evts)
        print("shutdown coros for CoroExternalCallsPlugin")

    async def cancel(self):
        await self.shutdown()

    async def run(self):
        while not self.shut_down_flag:
            req: CallsReq = await self.plugin_queue.get()
            result = self(req)
            id = req.actor_meta.actor_id
            assert isinstance(result, RolloutReq), "tools should "
            req.actor_meta.queue_group.push(id, result)
