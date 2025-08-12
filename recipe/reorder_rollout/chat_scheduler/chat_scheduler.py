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
import heapq
import logging
import math
import os
import random
import traceback
from concurrent.futures.thread import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Protocol
from uuid import uuid4

import hydra
import numpy as np
import ray
import torch
from cachetools import LRUCache
from omegaconf import DictConfig
from tensordict import TensorDict
from typing_extensions import runtime_checkable

from recipe.reorder_rollout.chat_scheduler.apis import (
    ReduceResp,
    RolloutReq,
    RolloutResp,
)
from recipe.reorder_rollout.chat_scheduler.utils import (
    ActorMeta,
    DeathLetter,
    QueueGroup,
    WorkStealingActor,
    _MgrProxy,
    agent_loop_postprocess,
    concat_data_proto,
)
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopOutput,
    _DummyConfig,
    get_registry_detail,
    get_registry_keys,
    get_trajectory_info,
)
from verl.experimental.agent_loop.utils import agent_loop_perf
from verl.protocol import DataProto
from verl.utils.fs import copy_to_local
from verl.utils.rollout_trace import rollout_trace_attr
from verl.utils.tokenizer import hf_processor, hf_tokenizer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


@runtime_checkable
class ReorderSchedulerMixin(Protocol):
    async def reorder_generate_sequences(
        self, data_iter: Iterable, renew: bool
    ) -> tuple[bool, DataProto, DataProto, DataProto]: ...


class ChatCompletionScheduler:
    def __init__(
        self,
        config: DictConfig,
        server_handles: list[ray.actor.ActorHandle],
        max_cache_size: int = 10000,
    ):
        """
        Args:
            config: DictConfig.
            server_handles: List[ray.actor.ActorHandle], OpenAI compatible server addresses.
            max_cache_size: int, max cache size of request_id to address mapping.
        """
        self.config = config.actor_rollout_ref.rollout
        model_path = config.actor_rollout_ref.model.path
        self.model_name = "/".join(model_path.split("/")[-2:])

        self.server_handles = server_handles
        random.shuffle(self.server_handles)

        # Least requests load balancing
        self.weighted_serveres = [[0, (hash(server), server)] for server in server_handles]
        heapq.heapify(self.weighted_serveres)

        # LRU cache to map request_id to address
        self.request_id_to_server = LRUCache(maxsize=max_cache_size)

        self.background_tasks = set()

    def _routing(self, request_id: str, handle) -> ray.actor.ActorHandle:
        if handle is not None:
            return handle
        # TODO: implement server pressure awareness load balancing
        if request_id in self.request_id_to_server:
            return self.request_id_to_server[request_id]

        server = self.weighted_serveres[0][1][1]
        self.weighted_serveres[0][0] += 1
        heapq.heapreplace(self.weighted_serveres, self.weighted_serveres[0])
        self.request_id_to_server[request_id] = server
        return server

    async def generate_sequences(self, batch: DataProto) -> DataProto: ...


class _Queue(asyncio.Queue):
    def __init__(self, maxsize=0, blocker: asyncio.Event = None):
        super().__init__(maxsize)
        self.blocker: asyncio.Event = blocker

    async def get(self):
        re = await super().get()
        await self.blocker.wait()
        return re

    def put_nowait(self, item):
        super().put_nowait(item)


class MicroBatchScheduler(ChatCompletionScheduler):
    def __init__(
        self,
        config,
        server_handles,
        max_cache_size=10000,
        max_inflight_req=8,
        rollout_req_handler=None,
        reduce_handler=None,
        enable_work_stealing=False,
    ):
        super().__init__(config, server_handles, max_cache_size)
        self.original_config = config
        self.mirco_batch_config = config.actor_rollout_ref.rollout.chat_scheduler
        self.micro_batch_per_dp = (
            self.mirco_batch_config.micro_batch.max_inflight_req
            if self.mirco_batch_config.micro_batch.max_inflight_req is not None
            else max_inflight_req
        )
        self.server_handles = server_handles
        self.enable_work_stealing = (
            self.mirco_batch_config.micro_batch.enable_work_stealing
            if self.mirco_batch_config.micro_batch.enable_work_stealing is not None
            else enable_work_stealing
        )
        self.number_of_servers = len(server_handles)
        self.rollout_rate = 1
        self.rollout_req_handler = rollout_req_handler
        self.reduce_handler = reduce_handler
        self.initialized = False
        model_path = config.actor_rollout_ref.model.path
        self.model_name = "/".join(model_path.split("/")[-2:])
        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=True)
        self.processor = hf_processor(local_path, trust_remote_code=True)
        self.max_prompt_length = self.config.prompt_length
        self.max_response_length = self.config.response_length

    def set_rollout_rate(self, rate):
        assert rate <= 1 and rate > 0, "rollout rate must be in (0, 1]"
        self.rollout_rate = rate

    def _get_rollout_batch_size(self, data_batch_size):
        return int(data_batch_size * self.rollout_rate)

    def _lazy_init_global_resource(self):
        if self.initialized:
            return
        else:
            self.initialized = True
        # TODO use ZMQ to implement pub-sub for debug purpose
        self.loop = asyncio.get_event_loop()
        self.death_signal = asyncio.Queue()
        self.global_data_blocker = asyncio.Event()
        self.global_data_queue: _Queue = _Queue(0, self.global_data_blocker)
        self.local_data_queue_group = QueueGroup(
            self.number_of_servers, [asyncio.Queue() for _ in range(self.number_of_servers)]
        )
        self.reduce_data_queue = asyncio.Queue()
        # TODO better implement a supervisor-tree pattern, include dead-letter-queue
        # to monitor whether any actor exit unexpectly
        self.engine_call_actors: list[WorkStealingActor] = self._init_engine_call_actors(
            server_address=self.server_handles, max_inflight_req=self.micro_batch_per_dp
        )
        self._init_death_signal_consumer()
        logger.info(
            f"start MicroBatchChatCompletionScheduler, with max_inflight_req: {self.micro_batch_per_dp}, \
            enable_work_stealing: {self.enable_work_stealing}, server_handles: {self.server_handles}"
        )

    def _init_death_signal_consumer(self):
        async def consume_daeth_signal():
            while True:
                letter = await self.death_signal.get()
                print(f"[MicroBatchChatCompletionScheduler] consume death letter: {letter}")

        asyncio.create_task(consume_daeth_signal())

    def _init_engine_call_actors(self, server_address, max_inflight_req):
        # we use a group of coroutine to consume send_queue and produce reduce_queue
        # since the asyncio.Queue is not thread safe.
        # max_inflight_req consumer coroutine to get element from local_queue and submit to vllm
        actors = []
        self.error_sink_queue = asyncio.Queue()
        self.re_key_dict: dict[str, int] = {}
        counter = 0
        for idx, addr in enumerate(server_address):
            print(
                f"[MicroBatchChatCompletionScheduler] init engine call actor  \
                {addr}, max_inflight_req: {max_inflight_req}"
            )
            for _ in range(max_inflight_req):
                work_fn = functools.partial(
                    self.rollout_req_handler,
                    addr,
                    self.reduce_data_queue,
                )
                actor = WorkStealingActor(
                    worker_id=idx,
                    local_id=counter,
                    local_queues=self.local_data_queue_group,
                    global_queue=self.global_data_queue,
                    work_fn=work_fn,
                    enable_work_stealing=self.enable_work_stealing,
                    death_sigal=self.death_signal,
                    sink_queue=self.error_sink_queue,
                )
                actors.append(actor)
                counter += 1
        print(f"[MicroBatchChatCompletionScheduler] init engine call actors done, total: {len(actors)}")

        async def _get_sink_queue_re():
            while True:
                task = await self.error_sink_queue.get()
                if task.sample_id in self.re_key_dict:
                    self.re_key_dict[task.sample_id] += 1
                else:
                    self.re_key_dict[task.sample_id] = 1

        asyncio.create_task(_get_sink_queue_re())
        return actors

    def wake_up_engine_actor(
        self,
    ):
        for actor in self.engine_call_actors:
            actor.wakeup()

    async def generate_sequences(self, batch: DataProto) -> DataProto:
        """Generate sequences from agent loop.

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
        config = self.original_config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=1.0,
        )

        # override sampling params for validation
        if batch.meta_info.get("validate", False):
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["temperature"] = config.val_kwargs.temperature

        # by default, we assume it's a single turn agent
        if "agent_name" not in batch.non_tensor_batch:
            batch.non_tensor_batch["agent_name"] = np.array(["single_turn_agent"] * len(batch), dtype=object)

        tasks = []
        agent_names = batch.non_tensor_batch["agent_name"]
        raw_prompts = batch.non_tensor_batch["raw_prompt"]
        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(len(raw_prompts))

        trajectory_info = await get_trajectory_info(
            batch.meta_info.get("global_steps", -1), index, batch.meta_info.get("validate", False)
        )

        for agent_name, messages, trajectory in zip(agent_names, raw_prompts, trajectory_info, strict=True):
            proxy_mgr = _MgrProxy(routing_method=functools.partial(self._routing, handle=None))
            tasks.append(
                asyncio.create_task(self._run_agent_loop(agent_name, messages, sampling_params, trajectory, proxy_mgr))
            )
        outputs = await asyncio.gather(*tasks)
        output = agent_loop_postprocess(
            tokenizer=self.tokenizer,
            inputs=outputs,
            max_prompt_length=self.max_prompt_length,
            max_response_length=self.max_response_length,
        )
        return output

    async def _run_agent_loop(
        self,
        agent_name: str,
        messages: list[dict[str, Any]],
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        proxy_mgr: _MgrProxy,
        token_ids: list[int] = None,
    ) -> AgentLoopOutput:
        with rollout_trace_attr(
            step=trajectory["step"],
            sample_index=trajectory["sample_index"],
            rollout_n=trajectory["rollout_n"],
            validate=trajectory["validate"],
            name="agent_loop",
        ):
            registry_keys = get_registry_keys()
            assert agent_name in registry_keys, (
                f"Agent loop {agent_name} not registered, registered agent loops: {registry_keys}"
            )

            agent_loop_config = get_registry_detail(agent_name)
            agent_loop = hydra.utils.instantiate(
                config=agent_loop_config,
                trainer_config=_DummyConfig(config=self.original_config),
                server_manager=proxy_mgr,
                tokenizer=self.tokenizer,
                processor=self.processor,
            )
            kwargs = dict(
                raw_prompt=messages,
                token_ids=token_ids,
            )
            output = await agent_loop.run(sampling_params, **kwargs)
            return output


@dataclass
class _Sample:
    rollout_req: RolloutReq
    batch: DataProto
    gen_batch: DataProto
    generation: int
    model_name: str
    messages: list[dict[str, str]]
    sampling_params: dict[str, Any]
    agent_name: np.ndarray
    trajectory_info: list[dict[str, Any]]
    n: int
    # should be placed in another data-structure
    temp_buffer: dict[int, list[int]]
    agent_loop_dict: dict[int, bool]
    staleness: int


class PartialPolicy(Enum):
    DROP = "drop"
    KEEP = "keep"


class ReorderScheduler(MicroBatchScheduler, ReorderSchedulerMixin):
    def __init__(
        self,
        config,
        server_handles,
        max_inflight_req=8,
        rollout_req_handler=None,
        reduce_handler=None,
        enable_work_stealing=True,
        data_fetcher=None,
    ):
        self.data_loader_blocker = asyncio.Event()
        self.is_sync_batch = False
        self.data_iter = None
        self.data_fetcher = data_fetcher if data_fetcher else self._default_data_fetcher
        self.rollout_req_handler = rollout_req_handler if rollout_req_handler else self.reorder_hanlde_rollout_req
        self.reduce_handler = reduce_handler if reduce_handler else self.reorder_handle_reduce_req
        self.data_fetcher_actor = None
        self.data_fetcher_exit = asyncio.Event()
        self.all_sample: dict[str, _Sample] = {}
        self.pending_sample: dict[str, _Sample] = {}
        self.active_sample: dict[str, _Sample] = {}
        self.done_sample: dict[str, _Sample] = {}
        self.done_sample_counter = 0
        self.data_iter_length = 0
        self.batch_counter = 0
        self.global_data = 0
        self.drop_dict: dict[str, bool] = {}
        self.put_counter = 0
        self.requeue_id_counter: dict[str, int] = {}
        logger.info("init stream scheduler")
        super().__init__(
            config,
            server_handles,
            1000,
            max_inflight_req,
            self.rollout_req_handler,
            self.reduce_handler,
            enable_work_stealing,
        )
        self.prefetch_factor = self.config.chat_scheduler.prefetch_factor
        self.batch_size = config.data.train_batch_size
        self.synchronize_interval = self.config.chat_scheduler.synchronize_interval
        self.partial_policy: PartialPolicy = PartialPolicy(self.config.chat_scheduler.partial_policy)  # drop or
        self.max_in_mem_samples = self.prefetch_factor * self.batch_size
        self.print_rate = 0.1
        self._validate()
        print(
            f"init with resource: tokenizer: {self.tokenizer},model name: {self.model_name}, max \
            prompt length: {self.max_prompt_length}, max response length: {self.max_response_length} \
            max_in_mem_samples: {self.max_in_mem_samples} \
            batch_size: {self.batch_size} \
            prefetch_factor: {self.prefetch_factor} \
            synchronize_interval: {self.synchronize_interval} \
            partial_policy: {self.partial_policy}"
        )

    def _validate(self):
        assert self.prefetch_factor >= 1, "prefetch_factor must be >=1"

        if self.prefetch_factor > 2 and self.synchronize_interval is not None:
            raise ValueError("prefetch_factor must be 2 when synchronize_interval is not None")

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
        sampling_params = dict(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            repetition_penalty=1.0,
        )

        logger.info(f"[ReorderScheduler] generate_sequences sampling params: {sampling_params}")
        try:
            async for gen_next_batch in _Iter(data_iter):
                gen_batch, batch = gen_next_batch
                # by default, we assume it's a single turn agent
                if "agent_name" not in gen_batch.non_tensor_batch:
                    gen_batch.non_tensor_batch["agent_name"] = np.array(["single_turn_agent"], dtype=object)

                agent_names = gen_batch.non_tensor_batch["agent_name"]
                raw_prompts = gen_batch.non_tensor_batch["raw_prompt"]

                if "index" in batch.non_tensor_batch:
                    index = batch.non_tensor_batch["index"]
                else:
                    index = np.arange(len(raw_prompts))

                trajectory_info = await get_trajectory_info(
                    batch.meta_info.get("global_steps", -1), index, batch.meta_info.get("validate", False)
                )
                # assert len(batch) == 1
                for agent_name, messages, trajectory in zip(agent_names, raw_prompts, trajectory_info, strict=True):
                    sample_id = self._generate_unique_id()
                    rollout_req = RolloutReq(
                        generation=0,
                        sample_id=sample_id,
                    )
                    self.all_sample[sample_id] = _Sample(
                        rollout_req,
                        batch,
                        gen_batch,
                        generation=0,
                        model_name=self.model_name,
                        messages=messages,
                        agent_name=agent_name,
                        trajectory_info=trajectory,
                        sampling_params=sampling_params,
                        n=self.config.n,
                        temp_buffer={},
                        agent_loop_dict={},
                        staleness=0,
                    )
                    self.pending_sample[sample_id] = None
                    # for _ in range(n):
                    _rollout_req = deepcopy(rollout_req)
                    logger.debug(f"put {sample_id} to global data queue")
                    # raw_prompt: [{"role": "user", "content": ""}, ["role": "assistant", "content"], ...]
                    await self.global_data_queue.put(_rollout_req)
                await self._stop_fetch_if_max_in_mem_samples()
        except asyncio.CancelledError:
            logger.warning("data fetcher cancled")
        except Exception as e:
            traceback.print_exc()
            logger.warning(f"exit data fetcher for exception: {e}")
        logger.info(f"exit data fetcher, pending sample size: {len(self.pending_sample)}")
        self.data_fetcher_exit.set()

    async def _stop_fetch_if_max_in_mem_samples(self):
        if len(self.all_sample) >= self.max_in_mem_samples:
            self.data_loader_blocker.clear()
            print(
                f"stop fetch data if max in mem samples, pending sample size: {len(self.pending_sample)}, \
                active_sample: {len(self.active_sample)}, done_sample: {len(self.done_sample)}"
            )
            await self.data_loader_blocker.wait()
            print(f"resume fetch data after stop, pending sample size: {len(self.pending_sample)}")

    def _generate_unique_id(self):
        while 1:
            sample_id = uuid4().hex
            if sample_id not in self.pending_sample:
                return sample_id

    def init_async_data_fetcher(self, data_iter, renew):
        logger.info(f"[start_async_data_fetcher]: data iter: {data_iter}，renew: {renew}, {self.data_iter},{data_iter}")
        if not renew:
            return
        if self.data_fetcher_actor is not None and self.data_fetcher_actor.done():
            self.data_fetcher_actor.cancel()

        self.data_fetcher_actor = asyncio.create_task(self._default_data_fetcher(data_iter))
        self.data_iter_length = len(data_iter)
        self.data_fetcher_exit.clear()
        self.done_sample_counter = 0
        self.batch_counter = 0

        def callback(task):
            if task.exception() is not None:
                letter = DeathLetter(
                    actor_meta=ActorMeta(actor_id=0, queue_group=None, local_id=0),
                    async_task=task,
                )
                if self.death_signal is not None:
                    self.death_signal.put_nowait(letter)
                logger.exception(f"global data fetcher exit for execption: {task}, exception: {task.exception()}")

        self.data_fetcher_actor.add_done_callback(callback)
        self.data_loader_blocker.set()
        logger.info(f"[start_async_data_fetcher]: data fetcher actor: {self.data_fetcher_actor}")

    def _lazy_init_global_resource(self, data_iter: Iterable, renew):
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
        logger.info(f"[ReorderScheduler] cancel_all_req with length: {len(evts)}")
        await asyncio.gather(*evts)

    async def reorder_hanlde_rollout_req(
        self,
        server_handle,
        reduce_queue: asyncio.Queue,
        actor_meta: ActorMeta,
        rollout_req: RolloutReq,
    ):
        n_task_dict = {}
        mgr_dict = {}
        has_abort = False
        sample_id = rollout_req.sample_id
        sample = self.all_sample[sample_id]
        exception = None
        partial_sample = False
        try:
            # Only move from pending to active once
            assert sample_id in self.pending_sample, (
                f"sample_id: {sample_id},pending_sample: {self.pending_sample.keys()}"
            )
            self.pending_sample.pop(sample_id)
            self.active_sample[sample_id] = None
            logger.debug(f"[ReorderScheduler] _run_agent_loop, sample: {sample_id}")
            # apply rollout policy
            # TODO: into a sperate func
            if len(sample.agent_loop_dict) == sample.n:
                # done already, apply rollout policy
                if self.partial_policy == PartialPolicy.KEEP:
                    # partial rollout case, ship to reduce queue
                    logger.debug(
                        f"[ReorderScheduler] _run_agent_loop, sample: {sample_id}, \
                        partial_policy: {self.partial_policy},ship to reduce"
                    )
                    reduce_queue.put_nowait(
                        RolloutResp(
                            request=rollout_req,
                            exception=None,
                        )
                    )
                    return
                else:
                    # should drop all previous results
                    sample.agent_loop_dict = {}
                    sample.temp_buffer = {}
            for i in range(sample.n):
                message = None
                token_ids = None
                if i in sample.agent_loop_dict:
                    continue
                proxy_mgr = _MgrProxy(routing_method=functools.partial(self._routing, handle=server_handle))
                mgr_dict[i] = proxy_mgr
                message = sample.messages
                if i in sample.temp_buffer and self.partial_policy == PartialPolicy.KEEP:
                    token_ids = sample.temp_buffer[i]
                    partial_sample = True
                    logger.debug(
                        f"[ReorderScheduler] _run_agent_loop, sample: {sample_id}, \
                        partial_policy: {self.partial_policy},token_ids: {token_ids}"
                    )
                task = asyncio.create_task(
                    self._run_agent_loop(
                        sample.agent_name,
                        message,
                        sample.sampling_params,
                        trajectory=deepcopy(sample.trajectory_info),
                        proxy_mgr=proxy_mgr,
                        token_ids=token_ids,
                    )
                )
                n_task_dict[i] = task
            await asyncio.gather(*[val for _, val in n_task_dict.items()])
        except asyncio.CancelledError:
            for ids, mgr in mgr_dict.items():
                if mgr.ray_awaitable is not None:
                    has_abort = True
                    request_id = mgr.request_id
                    maybe_resp = await mgr.server_handle.cancel.remote(request_id)
                    if maybe_resp is not None:
                        token_ids = maybe_resp.outputs[0].token_ids
                        sample.temp_buffer[ids] = token_ids
            for ids in n_task_dict.keys():
                if not n_task_dict[ids].cancelled() and n_task_dict[ids].done():
                    logger.debug(
                        f"[ReorderScheduler] _run_agent_loop cancel, \
                          sample_id: {sample_id},ids: {n_task_dict[ids].result()}"
                    )
                    sample.agent_loop_dict[ids] = n_task_dict[ids].result()
            logger.debug(f"[ReorderScheduler] _run_agent_loop cancel, sample_id: {sample_id},has_abort: {has_abort}")
            return
        except Exception as e:
            traceback.print_exc()
            print(f"[ReorderScheduler] _run_agent_loop failed with exception: {e}")
            exception = e
        # Update joiner state
        for ids in n_task_dict.keys():
            if n_task_dict[ids].done():
                sample.agent_loop_dict[ids] = n_task_dict[ids].result()
        # we don't put it back to the global queue, since we will requeue it later
        # for active sample
        if has_abort:
            (f"[ReorderScheduler] _run_agent_loop abort, sample_id: {sample_id},has_abort: {has_abort}")
            return
        if partial_sample:
            logger.debug(f"partial rollout result: sample_id:{sample_id} result {sample.agent_loop_dict}")

        resp = RolloutResp(
            request=rollout_req,
            exception=exception,
        )
        logger.debug(f"[ReorderScheduler] _run_agent_loop done, resp: {resp}")
        reduce_queue.put_nowait(resp)
        return

    # maybe we can make this sink_queue as a pubsub proxy using zmq
    async def reorder_handle_reduce_req(self, batch_size, n_sample, sink_queue: asyncio.Queue = None):
        batch_agent_output = []
        gen_batch_proto_list = []
        batch_proto_list = []
        counter = 0
        _print_div = math.ceil(batch_size * self.print_rate)
        if self.print_rate == 0:
            _print_div = 1
        print(
            f"[ReorderScheduler] _gather_result launch, current queue size: \
            {self.reduce_data_queue.qsize()}, batch_size: {batch_size}, print_rate: {_print_div}"
        )
        while counter < batch_size:
            if counter % _print_div == 0:
                print(
                    f"[ReorderScheduler] _gather_result counter: {counter}，"
                    f"batch_size: {batch_size},pending samples: {len(self.pending_sample)},"
                    f"active_sample: {len(self.active_sample)}, queue size: {self.global_data_queue.qsize()},"
                    f" queue task: {self.global_data_queue._unfinished_tasks}"
                )
            rollout_resp: RolloutResp = await self.reduce_data_queue.get()
            sample_id = rollout_resp.request.sample_id
            sample = self.all_sample[sample_id]
            if rollout_resp.request.generation != sample.generation:
                # skip here, since it's stale generation
                # for partial rollout allow this, the handle_reduce_req will handle
                # this and send it directly to reduce queue with latest generation
                print(
                    f"stale generation, sample_id: {sample_id}, generation: \
                    {rollout_resp.request.generation}, sample_generation: {sample.generation}"
                )
                continue
            if sink_queue is not None:
                sink_queue.put(rollout_resp)
            assert sample_id in self.active_sample, f"sample_id: {sample_id}"
            if rollout_resp.exception is not None:
                # maybe skip or requeue?  error handling issue
                # should be handled by supervisor
                raise rollout_resp.exception
            self.active_sample.pop(sample_id)
            _sample = self.all_sample.get(sample_id)
            logger.debug(f"finished for samples: {sample_id}")
            loop_outputs: list[AgentLoopOutput] = [_sample.agent_loop_dict[i] for i in range(self.config.n)]
            reduce_resp = self._build_reduce_resp(loop_outputs)
            batch_agent_output.append(reduce_resp)
            gen_batch_proto_list.append(_sample.gen_batch)
            batch_proto_list.append(_sample.batch)
            self.done_sample[sample_id] = _sample
            _sample.staleness = _sample.generation
            counter += 1
        batch = concat_data_proto(batch_proto_list)
        gen_batch = concat_data_proto(gen_batch_proto_list)
        return batch_agent_output, gen_batch, batch

    def _build_reduce_resp(self, loop_outputs: list[AgentLoopOutput]):
        # prompts
        self.tokenizer.padding_side = "left"
        outputs = self.tokenizer.pad(
            [{"input_ids": input.prompt_ids} for input in loop_outputs],
            padding="max_length",
            max_length=self.max_prompt_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        prompt_ids, prompt_attention_mask = outputs["input_ids"], outputs["attention_mask"]

        # responses
        self.tokenizer.padding_side = "right"
        outputs = self.tokenizer.pad(
            [{"input_ids": input.response_ids} for input in loop_outputs],
            padding="max_length",
            max_length=self.max_response_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        response_ids, response_attention_mask = outputs["input_ids"], outputs["attention_mask"]

        # response_mask
        outputs = self.tokenizer.pad(
            [{"input_ids": input.response_mask} for input in loop_outputs],
            padding="max_length",
            max_length=self.max_response_length,
            return_tensors="pt",
            return_attention_mask=False,
        )
        response_mask = outputs["input_ids"]
        return ReduceResp(
            prompt_ids=prompt_ids,
            prompt_attention_mask=prompt_attention_mask,
            response_ids=response_ids,
            response_attention_mask=response_attention_mask,
            response_mask=response_mask,
            agent_loop_output_list=loop_outputs,
        )

    def _requeue_preempt_req(self):
        print(
            f"ready to requeue active samples {len(self.active_sample)},"
            f"current pending samples: {len(self.pending_sample)},"
            f"global_queue_size: {self.global_data_queue.qsize()}"
        )
        for sample_id in list(self.active_sample.keys()):
            self.active_sample.pop(sample_id)
            _sample: _Sample = self.all_sample.get(sample_id)
            _sample.generation += 1
            self.pending_sample[sample_id] = None
            req: RolloutReq = deepcopy(_sample.rollout_req)
            req.generation = _sample.generation
            self.global_data_queue.put_nowait(req)
        print(
            f"current queue size: {self.global_data_queue.qsize()},"
            f"queue undone task: {self.global_data_queue._unfinished_tasks}"
        )
        assert len(self.active_sample) == 0

    async def reorder_generate_sequences(
        self, data_iter: Iterable, renew=False
    ) -> tuple[bool, DataProto, DataProto, DataProto]:
        self._lazy_init_global_resource(data_iter, renew)
        self.global_data_blocker.set()
        self.wake_up_engine_actor()
        pending_sample_length = len(self.pending_sample)
        if self._data_fetcher_done() and pending_sample_length == 0:
            return True, None, None, None
        is_last_batch, bsz = self.handle_last_batch(self.batch_size)
        print(f"[ReorderScheduler] waiting for rollout done, bsz: {bsz}")
        batch_conversations, gen_batch, batch = await self.reduce_handler(bsz, n_sample=self.config.n)
        print(
            f"[ReorderScheduler] reorder rollout done, cancel all left request, real size: {len(batch_conversations)}"
        )
        await self.cancel_all_req()
        self._requeue_preempt_req()
        self.global_data_blocker.clear()
        done_sample_ids = list(self.done_sample.keys())
        for sample_id in done_sample_ids:
            self.done_sample.pop(sample_id)
            self.all_sample.pop(sample_id)
        assert len(self.done_sample) == 0
        self.done_sample_counter += len(batch_conversations)
        self.batch_counter += 1
        print(
            f"[ReorderScheduler] generate_sequences done with {len(batch_conversations)} samples, \
            done_sample_counter: {self.done_sample_counter}"
        )
        print(
            f"current data info: pending_sample: {len(self.pending_sample)}, active_sample: {len(self.active_sample)},"
            f"done_sample: {len(self.done_sample)}, data_len: {self.data_iter_length}"
        )
        gen_batch_output = self.handle_agent_loop_output(batch_conversations)
        if not is_last_batch:
            self.handle_sync_batch()
        self.last_batch_sanity_check(is_last_batch)
        timing = agent_loop_perf([gen_batch_output.meta_info["metrics"]], gen_batch_output)
        gen_batch_output.meta_info = {"timing": timing}
        return False, gen_batch_output, gen_batch, batch

    def handle_agent_loop_output(self, batch_conversations: list[ReduceResp]):
        agent_loop_output_list = [
            item for reduce_resp in batch_conversations for item in reduce_resp.agent_loop_output_list
        ]
        # cat all result
        response_ids = torch.cat([item.response_ids for item in batch_conversations], dim=0)
        response_mask = torch.cat([item.response_mask for item in batch_conversations], dim=0)
        response_attention_mask = torch.cat([item.response_attention_mask for item in batch_conversations], dim=0)
        prompt_ids = torch.cat([item.prompt_ids for item in batch_conversations], dim=0)
        prompt_attention_mask = torch.cat([item.prompt_attention_mask for item in batch_conversations], dim=0)
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

        num_turns = np.array([input.num_turns for input in agent_loop_output_list], dtype=np.int32)
        metrics = [input.metrics.model_dump() for input in agent_loop_output_list]
        return DataProto(batch=batch, non_tensor_batch={"__num_turns__": num_turns}, meta_info={"metrics": metrics})

    def handle_last_batch(self, expect_buffer_size) -> tuple[bool, int]:
        # how do we know whether the elements in data_iter plus active_samples is less than buffer_size?
        if self.data_iter_length - self.done_sample_counter < expect_buffer_size:
            return True, self.data_iter_length - self.done_sample_counter
        else:
            return False, expect_buffer_size

    def last_batch_sanity_check(self, is_last_batch):
        if not is_last_batch:
            return
        assert self.done_sample_counter == self.data_iter_length
        assert len(self.pending_sample) == 0, f"pending_sample not empty, {self.pending_sample}"
        assert len(self.active_sample) == 0, f"active_sample not empty, {self.active_sample}"
        assert self.global_data_queue.empty(), "global_data_queue not empty"
        assert self.reduce_data_queue.empty(), "reduce_data_queue not empty"

    def handle_sync_batch(self):
        self.done_sample = {}
        if self.synchronize_interval is not None and self._is_next_sync_batch(
            self.batch_counter, self.synchronize_interval
        ):
            logger.info(f"[ReorderScheduler] handle_sync_batch, next batch: {self.batch_counter} should be sync batch")
            # modify prefetch factor to one, so the dataloader will only fetch one batch
            self.max_in_mem_samples = self.batch_size
            self.is_sync_batch = True
            if len(self.pending_sample) >= self.batch_size:
                # no need to set blocker
                logger.info(f"[ReorderScheduler] pending samples: {len(self.pending_sample)},skip set blocker")
            else:
                # samples still needs to be fetched from disk, keep fetching
                logger.info(
                    f"[ReorderScheduler] resume fetch for sync batch, pending sample size: {len(self.pending_sample)}"
                )
                self.data_loader_blocker.set()
        else:
            self.max_in_mem_samples = self.batch_size * self.prefetch_factor
            self.is_sync_batch = False
            self.data_loader_blocker.set()
            logger.info(
                f"[ReorderScheduler] resume fetch for next batch, pending sample size: {len(self.pending_sample)}"
            )

    def _is_next_sync_batch(self, batch_numb, sync_interval) -> bool:
        batch_numb += 1
        return batch_numb % sync_interval == 0
