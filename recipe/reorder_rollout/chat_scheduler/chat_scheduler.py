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
from concurrent.futures.thread import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable, Protocol
from uuid import uuid4

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
    concat_data_proto,
    get_agent_loop_class,
)
from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
from verl.experimental.agent_loop.utils import agent_loop_perf, agent_loop_postprocess
from verl.protocol import DataProto
from verl.utils.fs import copy_to_local
from verl.utils.tokenizer import hf_tokenizer

logger = logging.getLogger(__name__)
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


class MicroBatchScheduler(ChatCompletionScheduler):
    def __init__(
        self,
        config,
        server_handles,
        max_cache_size=10000,
        max_inflight_req=8,
        rollout_req_handler=None,
        reduce_handler=None,
        enable_work_stealing=True,
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
        self.global_data_queue = asyncio.Queue()
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
                )
                actors.append(actor)
                counter += 1
        print(f"[MicroBatchChatCompletionScheduler] init engine call actors done, total: {len(actors)}")
        return actors

    def wake_up_engine_actor(
        self,
    ):
        for actor in self.engine_call_actors:
            actor.wakeup()

    async def shut_down_actors(self):
        print("shut down engine actors with length: ", len(self.engine_call_actors))
        for actor in self.engine_call_actors:
            print("ready to shutdown actor: ", actor.actor_meta)
            await actor.shutdown()
        print("[MicroBatchChatCompletionScheduler] shut down engine actor")

    async def generate_sequences(self, batch: DataProto) -> DataProto:
        """Generate sequences from agent loop.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
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

        n = 1 if batch.meta_info.get("validate", False) else config.n
        tasks = []

        # by default, we assume it's a single turn agent
        if "agent_name" not in batch.non_tensor_batch:
            batch.non_tensor_batch["agent_name"] = np.array(["single_turn_agent"] * len(batch), dtype=object)

        agent_names = batch.non_tensor_batch["agent_name"].repeat(n, axis=0)
        raw_prompts = batch.non_tensor_batch["raw_prompt"].repeat(n, axis=0)
        for agent_name, messages in zip(agent_names, raw_prompts, strict=False):
            proxy_mgr = _MgrProxy(routing_method=functools.partial(self._routing, handle=None))
            tasks.append(
                asyncio.create_task(self._run_agent_loop(agent_name, messages.tolist(), sampling_params, proxy_mgr))
            )
        print(f"length of tasks: {len(tasks)}")
        outputs = await asyncio.gather(*tasks)

        output = agent_loop_postprocess(self.tokenizer, outputs, self.max_prompt_length, self.max_response_length)
        print("[MicroBatchScheduler] generate_sequences done")
        # calculate performance metrics
        metrics = [output.meta_info["metrics"]]  # List[List[Dict[str, str]]]
        timing = agent_loop_perf(metrics, output)

        output.meta_info = {"timing": timing}

        return output

    async def _run_agent_loop(
        self, agent_name: str, messages: list[dict[str, Any]], sampling_params: dict[str, Any], proxy_mgr
    ) -> AgentLoopOutput:
        agent_loop_class = get_agent_loop_class(agent_name)
        agent_loop = agent_loop_class(self.original_config, proxy_mgr, self.tokenizer)
        output = await agent_loop.run(messages, sampling_params)
        return output


@dataclass
class _Sample:
    rollout_req: RolloutReq
    batch: DataProto
    gen_batch: DataProto
    generation: int


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
        self.pending_sample: dict[str, _Sample] = {}
        self.active_sample: dict[str, _Sample] = {}
        self.done_sample_counter = 0
        self.data_iter_length = 0
        self.batch_counter = 0
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
        self.max_prompt_length = self.config.prompt_length
        self.max_response_length = self.config.response_length
        self.prefetch_factor = self.config.chat_scheduler.prefetch_factor
        self.batch_size = config.data.train_batch_size
        self.synchorize_interval = self.config.chat_scheduler.synchorize_interval
        self.max_in_mem_samples = self.prefetch_factor * self.batch_size
        model_path = config.actor_rollout_ref.model.path
        self.model_name = "/".join(model_path.split("/")[-2:])
        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=True)
        self._validate()
        logger.info(
            f"init with resource: tokenizer: {self.tokenizer},model name: {self.model_name}, max \
            prompt length: {self.max_prompt_length}, max response length: {self.max_response_length} \
            max_in_mem_samples: {self.max_in_mem_samples} \
            batch_size: {self.batch_size} \
            prefetch_factor: {self.prefetch_factor} \
            synchorize_interval: {self.synchorize_interval}"
        )

    def _validate(self):
        assert self.prefetch_factor >= 1, "prefetch_factor must be >=1"

        if self.prefetch_factor > 2 and self.synchorize_interval is not None:
            raise ValueError("prefetch_factor must be 2 when synchorize_interval is not None")

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
        n = self.config.n

        logger.info(f"[ReorderScheduler] generate_sequences sampling params: {sampling_params}")
        try:
            async for gen_next_batch in _Iter(data_iter):
                gen_batch, batch = gen_next_batch
                # by default, we assume it's a single turn agent
                if "agent_name" not in gen_batch.non_tensor_batch:
                    gen_batch.non_tensor_batch["agent_name"] = np.array(["single_turn_agent"], dtype=object)

                agent_names = gen_batch.non_tensor_batch["agent_name"]
                raw_prompts = gen_batch.non_tensor_batch["raw_prompt"]
                # assert len(batch) == 1
                for agent_name, messages in zip(agent_names, raw_prompts, strict=False):
                    sample_id = self._generate_unique_id()
                    rollout_req = RolloutReq(
                        agent_name=agent_name,
                        messages=messages.tolist(),
                        model_name=self.model_name,
                        sampling_params=sampling_params,
                        generation=0,
                        sample_id=sample_id,
                    )
                    self.pending_sample[sample_id] = _Sample(rollout_req, batch, gen_batch, generation=0)
                    for _ in range(n):
                        _rollout_req = deepcopy(rollout_req)
                        _rollout_req.verl_session_id = uuid4().hex
                        logger.debug(f"put {sample_id} to global data queue")
                        # raw_prompt: [{"role": "user", "content": ""}, ["role": "assistant", "content"], ...]
                        await self.global_data_queue.put(_rollout_req)
                await self._stop_fetch_if_max_in_mem_samples()
        except asyncio.CancelledError:
            logger.warning("data fetcher cancled")
        except Exception as e:
            logger.warning(f"exit data fetcher for exception: {e}")
        logger.info(f"exit data fetcher, pending sample size: {len(self.pending_sample)}")
        self.data_fetcher_exit.set()

    async def _stop_fetch_if_max_in_mem_samples(self):
        if len(self.pending_sample) + len(self.active_sample) >= self.max_in_mem_samples:
            self.data_loader_blocker.clear()
            print(
                f"stop fetch data if max in mem samples, pending sample size: {len(self.pending_sample)}, \
                active_sample: {len(self.active_sample)}"
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

    def _skip_if_stale_request(self, rollout_req: RolloutReq, stage: str):
        sample_id = rollout_req.sample_id
        if sample_id in self.pending_sample:
            if self.pending_sample[sample_id].generation == rollout_req.generation:
                # first time see this sample, pop it from pending sample.
                # TODO reduce handler won't hit this term, make sanity check if necessary.
                self.active_sample[sample_id] = self.pending_sample.pop(sample_id)
                return False
            else:
                # stale generation,skip this
                logger.debug(
                    f"[ReorderScheduler] _consumer process get sample,stage: {stage}, skip  \
                    pending sample, sample_id: {sample_id}"
                )
                return True
        elif sample_id in self.active_sample.keys():
            #  in active sample, must be n_sample cases
            if self.active_sample[sample_id].generation != rollout_req.generation:
                # stale generation,skip this
                logger.debug(
                    f"[ReorderScheduler] _consumer process get sample, stage: {stage}, \
                    skip active sample, sample_id: {sample_id}"
                )
                return True
        else:
            # not in pending and active, should be a finished one but with stale generation
            # TODO better sanity check
            logger.debug(
                f"[ReorderScheduler] _consumer process get sample, stage: {stage}, \
                sample_id: {sample_id} not in pending and active"
            )
            return True
        return False

    async def reorder_hanlde_rollout_req(
        self,
        server_handle,
        reduce_queue: asyncio.Queue,
        actor_meta: ActorMeta,
        rollout_req: RolloutReq,
    ):
        if self._skip_if_stale_request(rollout_req, stage="handle_rollout_req"):
            return
        # agent loop need a async function implement generate_sequences interface.
        request_id = None
        proxy_mgr = _MgrProxy(routing_method=functools.partial(self._routing, handle=server_handle))
        agent_loop_output, exception = None, None
        try:
            logger.debug(
                f"[StreamScheduler] _consumer process get sample, \
                  submit to engine {server_handle}，sample_id: {rollout_req.sample_id}"
            )
            agent_loop_output = await self._run_agent_loop(
                rollout_req.agent_name, rollout_req.messages, rollout_req.sampling_params, proxy_mgr=proxy_mgr
            )
        except asyncio.CancelledError as cancel_err:
            request_id = proxy_mgr.request_id
            logger.debug(f"chat completion failed with exception: {cancel_err}")
            # do cancel:
            if proxy_mgr.ray_awaitable is not None:
                # need to figure out how tool calls needs to be canceled
                await proxy_mgr.server_handle.cancel.remote(request_id)
            raise
        except Exception as e:
            logger.warning(
                f"[ReorderScheduler] _consumer process get sample,chat completion failed with exception: {e}"
            )
            exception = e
        request_id = proxy_mgr.request_id
        resp = RolloutResp(
            request=rollout_req, exception=exception, req_id=request_id, agent_loop_output=agent_loop_output
        )
        try:
            logger.debug(f"[ReorderScheduler] _consumer process put sample to reduce_queue,idx: {actor_meta.actor_id}")
            reduce_queue.put_nowait(resp)
        except Exception as e:
            resp.exception = e
            reduce_queue.put_nowait(resp)

    # maybe we can make this sink_queue as a pubsub proxy using zmq
    async def reorder_handle_reduce_req(self, batch_size, n_sample, sink_queue: asyncio.Queue = None):
        batch_agent_output = []
        # joiner_buffer worked as key for sample_id,value for result.
        # make sure n-sample arrived correctlly then ship to batch_conversations as result
        joiner_buffer: dict[str, list[list[dict[str, str]]]] = {}
        gen_batch_proto_list = []
        batch_proto_list = []
        counter = 0
        print_rate = 0.1
        _print_div = math.ceil(batch_size * print_rate)
        if print_rate == 0:
            _print_div = 1
        print(
            f"[ReorderScheduler] _gather_result launch, current queue size: \
            {self.reduce_data_queue.qsize()}, batch_size: {batch_size}, print_rate: {print_rate}"
        )
        while counter < batch_size:
            if counter % _print_div == 0:
                print(f"[ReorderScheduler] _gather_result counter: {counter}")
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
                joiner_buffer[sample_id].append(rollout_resp.agent_loop_output)
            else:
                joiner_buffer[sample_id] = [rollout_resp.agent_loop_output]
            if len(joiner_buffer[sample_id]) == n_sample:
                _sample = self.active_sample.pop(sample_id)
                logger.debug(f"finished for samples: {sample_id}")
                loop_outputs: list[AgentLoopOutput] = joiner_buffer[sample_id]
                reduce_resp = self._build_reduce_resp(loop_outputs)
                batch_agent_output.append(reduce_resp)
                gen_batch_proto_list.append(_sample.gen_batch)
                batch_proto_list.append(_sample.batch)
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
        print(f"ready to requeue active samples {len(self.active_sample)}")
        for sample_id in list(self.active_sample.keys()):
            _sample: _Sample = self.active_sample.pop(sample_id)
            _sample.generation += 1
            self.pending_sample[sample_id] = _sample
            for _ in range(self.config.n):
                req: RolloutReq = deepcopy(_sample.rollout_req)
                req.verl_session_id = uuid4().hex
                req.generation = _sample.generation
                self.global_data_queue.put_nowait(req)
        assert len(self.active_sample) == 0

    async def reorder_generate_sequences(
        self, data_iter: Iterable, renew=False
    ) -> tuple[bool, DataProto, DataProto, DataProto]:
        self._lazy_init_global_resource(data_iter, renew)
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
        self.done_sample_counter += len(batch_conversations)
        self.batch_counter += 1
        print(
            f"[ReorderScheduler] generate_sequences done with {len(batch_conversations)} samples, \
            done_sample_counter: {self.done_sample_counter}"
        )
        gen_batch_output = self.handle_agent_loop_output(batch_conversations)
        if not is_last_batch:
            self.handle_sync_batch()
        self.last_batch_sanity_check(is_last_batch)
        timing = agent_loop_perf([gen_batch_output.meta_info["metrics"]], gen_batch_output)
        gen_batch_output.meta_info = {"timing": timing}
        return is_last_batch, gen_batch_output, gen_batch, batch

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
        if self.synchorize_interval is not None and self._is_next_sync_batch(
            self.batch_counter, self.synchorize_interval
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
