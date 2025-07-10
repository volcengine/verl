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
import functools
import heapq
import logging
import math
import os
import random
from concurrent.futures.thread import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Protocol, Tuple
from uuid import uuid4

import numpy as np
import ray
from cachetools import LRUCache
from omegaconf import DictConfig
from typing_extensions import runtime_checkable

from recipe.stream_mode.chat_scheduler.apis import (
    AsyncCallbackMixin,
    ReduceResp,
    RolloutReq,
    RolloutResp,
)
from recipe.stream_mode.chat_scheduler.utils import (
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
class StreamSchedulerMixin(Protocol):
    async def stream_generate_sequences(
        self, data_iter: Iterable, batch_size: int
    ) -> Tuple[bool, DataProto, DataProto, DataProto]: ...


class ChatCompletionScheduler:
    def __init__(
        self,
        config: DictConfig,
        server_handles: List[ray.actor.ActorHandle],
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


class MicroBatchScheduler(ChatCompletionScheduler):
    def __init__(
        self,
        config,
        server_handles,
        max_cache_size=10000,
        rollout_rate=1,
        max_inflight_req=8,
        rollout_req_handler=None,
        reduce_handler=None,
        enable_work_stealing=True,
    ):
        super().__init__(config, server_handles, max_cache_size)
        self.mirco_batch_config = config.actor_rollout_ref.rollout.chat_scheduler
        print(self.config)
        self.micro_batch_per_dp = (
            self.mirco_batch_config.micro_batch.max_inflight_req
            if self.mirco_batch_config.micro_batch.max_inflight_req
            else max_inflight_req
        )
        self.server_handles = server_handles
        self.enable_work_stealing = (
            self.mirco_batch_config.micro_batch.enable_work_stealing
            if self.mirco_batch_config.micro_batch.enable_work_stealing
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
        self.death_letter = asyncio.Queue()
        self.global_data_queue = asyncio.Queue()
        self.local_data_queue_group = QueueGroup(
            self.number_of_servers, [asyncio.Queue() for _ in range(self.number_of_servers)]
        )
        self.reduce_data_queue = asyncio.Queue()
        # TODO better implement a supervisor-tree pattern, include dead-letter-queue
        # to monitor whether any actor exit unexpectly
        self.engine_call_actors: List[WorkStealingActor] = self._init_engine_call_actors(
            server_address=self.server_handles, max_inflight_req=self.micro_batch_per_dp
        )
        self._init_death_letter_consumer()
        logger.info(
            f"start MicroBatchChatCompletionScheduler, with max_inflight_req: {self.micro_batch_per_dp}, \
            enable_work_stealing: {self.enable_work_stealing}, server_handles: {self.server_handles}"
        )

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
            print(
                f"[MicroBatchChatCompletionScheduler] init engine call actor  \
                {addr}, max_inflight_req: {max_inflight_req}"
            )
            for _ in range(max_inflight_req):
                work_fn = functools.partial(
                    self.rollout_req_handler,
                    addr,
                    self.reduce_data_queue,
                    self.completion_callback,
                )
                actor = WorkStealingActor(
                    worker_id=idx,
                    local_id=counter,
                    local_queues=self.local_data_queue_group,
                    global_queue=self.global_data_queue,
                    work_fn=work_fn,
                    enable_work_stealing=self.enable_work_stealing,
                    death_letter=self.death_letter,
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
        for agent_name, messages in zip(agent_names, raw_prompts):
            proxy_mgr = _MgrProxy(routing_method=functools.partial(self._routing, handle=None))
            tasks.append(
                asyncio.create_task(self._run_agent_loop(agent_name, messages.tolist(), sampling_params, proxy_mgr))
            )
        print(f"length of tasks: {len(tasks)}")
        outputs = await asyncio.gather(*tasks)

        output = agent_loop_postprocess(self.tokenizer, outputs, self.max_prompt_length, self.max_response_length)
        print("[StreamChatCompletionScheduler] generate_sequences done")
        # calculate performance metrics
        metrics = [output.meta_info["metrics"]]  # List[List[Dict[str, str]]]
        timing = agent_loop_perf(metrics, output)

        output.meta_info = {"timing": timing}

        return output


@dataclass
class _Sample:
    rollout_req: RolloutReq
    batch: DataProto
    gen_batch: DataProto
    generation: int


class StreamScheduler(MicroBatchScheduler, StreamSchedulerMixin):
    def __init__(
        self,
        config,
        server_handles,
        rollout_rate=1,
        max_inflight_req=8,
        rollout_req_handler=None,
        reduce_handler=None,
        enable_work_stealing=True,
        data_fetcher=None,
        batch_size=1024,
        prefetch_factor=2,
    ):
        self.original_config = config
        self.max_in_mem_samples = prefetch_factor * batch_size
        self.data_loader_blocker = asyncio.Event()
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
        logger.info("init stream scheduler")
        super().__init__(
            config,
            server_handles,
            1000,
            rollout_rate,
            max_inflight_req,
            self.rollout_req_handler,
            self.reduce_handler,
            enable_work_stealing,
        )
        self.max_prompt_length = self.config.prompt_length
        self.max_response_length = self.config.response_length
        model_path = config.actor_rollout_ref.model.path
        self.model_name = "/".join(model_path.split("/")[-2:])
        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=True)
        print(
            f"init with resource: tokenizer: {self.tokenizer},model name: {self.model_name}, max \
            prompt length: {self.max_prompt_length}, max response length: {self.max_response_length}"
        )

    async def _memory_monitor(self, blocker: asyncio.Event, max_in_mem_sample):
        await asyncio.sleep(1)
        in_memory_sample = len(self.pending_sample) + len(self.active_sample)
        if in_memory_sample > max_in_mem_sample:
            logger.debug(
                f"memory utilization {max_in_mem_sample} exceed threshold {self.data_memory_utils}, stop \
                data fetcher"
            )
            blocker.clear()
        else:
            # wake up data fetcher
            logger.debug(
                f"memory utilization {in_memory_sample} under threshold {max_in_mem_sample}, wake up data fetcher"
            )
            blocker.set()

    def start_memory_monitor(self):
        self.mem_monitor_actor = asyncio.create_task(
            self._memory_monitor(self.data_loader_blocker, self.max_in_mem_samples)
        )

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
        sampling_params = dict(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            repetition_penalty=1.0,
        )
        n = self.config.n

        print(f"[ChatCompletionScheduler] generate_sequences sampling params: {sampling_params}")
        try:
            async for gen_next_batch in _Iter(data_iter):
                gen_batch, batch = gen_next_batch
                # by default, we assume it's a single turn agent
                if "agent_name" not in gen_batch.non_tensor_batch:
                    gen_batch.non_tensor_batch["agent_name"] = np.array(["single_turn_agent"], dtype=object)

                agent_names = gen_batch.non_tensor_batch["agent_name"]
                raw_prompts = gen_batch.non_tensor_batch["raw_prompt"]
                # assert len(batch) == 1
                for agent_name, messages in zip(agent_names, raw_prompts):
                    sample_id = self._generate_unique_id()
                    rollout_req = RolloutReq(
                        agent_name=agent_name,
                        messages=messages.tolist(),
                        model_name=self.model_name,
                        sampling_params=sampling_params,
                        tools_schema=self.completion_callback.tool_schemas,
                        extra_body=self.completion_callback.extra_body,
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
        self.data_loader_blocker.set()
        print(f"[start_async_data_fetcher]: data fetcher actor: {self.data_fetcher_actor}")

    def _lazy_init_global_resource(self, data_iter: Iterable, renew):
        print("_lazy_init_global_resource")
        super()._lazy_init_global_resource()
        self.start_memory_monitor()
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
                logger.debug(
                    f"[StreamScheduler] _consumer process get sample,stage: {stage}, skip  \
                    pending sample, sample_id: {sample_id}"
                )
                return True
        elif sample_id in self.active_sample.keys():
            #  in active sample, must be n_sample cases
            if self.active_sample[sample_id].generation != rollout_req.generation:
                # stale generation,skip this
                logger.debug(
                    f"[StreamScheduler] _consumer process get sample, stage: {stage}, \
                    skip active sample, sample_id: {sample_id}"
                )
                return True
        else:
            # not in pending and active, should be a finished one but with stale generation
            # TODO better sanity check
            logger.debug(
                f"[StreamScheduler] _consumer process get sample, stage: {stage}, \
                sample_id: {sample_id} not in pending and active"
            )
            return True
        return False

    async def stream_handle_rollout_req(
        self,
        server_handle,
        reduce_queue: asyncio.Queue,
        external_call: AsyncCallbackMixin,
        actor_meta: ActorMeta,
        rollout_req: RolloutReq,
    ):
        if self._skip_if_stale_request(rollout_req, stage="handle_rollout_req"):
            return
        logger.debug(f"[StreamScheduler] _consumer process get sample, addr: {server_handle}, actor_meta: {actor_meta}")
        # agent loop need a async function implement generate_sequences interface.
        request_id = None
        proxy_mgr = _MgrProxy(routing_method=functools.partial(self._routing, handle=server_handle))
        agent_loop_output, exception = None, None
        try:
            # NOTE: OpenAI client uses httpx, seems to have performance issue in high concurrency requests.
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
                logger.debug(f"cancel with req: {request_id}")
            else:
                logger.debug(f"ray_awaitable none,: {request_id}")
            raise
        except Exception as e:
            logger.warning(f"[StreamScheduler] _consumer process get sample,chat completion failed with exception: {e}")
            exception = e
        request_id = proxy_mgr.request_id
        logger.debug(
            f"[StreamScheduler] _consumer process get sample \
              done,metrics: {agent_loop_output.metrics}, actor_meta: {actor_meta}"
        )
        resp = RolloutResp(
            request=rollout_req, exception=exception, req_id=request_id, agent_loop_output=agent_loop_output
        )
        try:
            logger.debug(f"[StreamScheduler] _consumer process put sample to reduce_queue,idx: {actor_meta.actor_id}")
            reduce_queue.put_nowait(resp)
        except Exception as e:
            logger.warning(
                f"[StreamScheduler] _consumer process put sample \
                to reduce_queue failed,idx: {actor_meta.actor_id}, exception: {e}"
            )
            resp.exception = e
            reduce_queue.put_nowait(resp)

    # maybe we can make this sink_queue as a pubsub proxy using zmq
    async def stream_handle_reduce_req(
        self, batch_size, n_sample, sink_queue: asyncio.Queue = None, format="ReduceResp"
    ):
        batch_agent_output = []
        # joiner_buffer worked as key for sample_id,value for result.
        # make sure n-sample arrived correctlly then ship to batch_conversations as result
        joiner_buffer: Dict[str, List[List[Dict[str, str]]]] = {}
        gen_batch_proto_list = []
        batch_proto_list = []
        counter = 0
        print_rate = 0.1
        _print_div = math.ceil(batch_size * print_rate)
        if print_rate == 0:
            _print_div = 1
        print(
            f"[stream_handle_reduce_req] _gather_result launch, current queue size: \
            {self.reduce_data_queue.qsize()}, batch_size: {batch_size}, print_rate: {print_rate}"
        )
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
                joiner_buffer[sample_id].append(rollout_resp.agent_loop_output)
            else:
                joiner_buffer[sample_id] = [rollout_resp.agent_loop_output]
            if len(joiner_buffer[sample_id]) == n_sample:
                _sample = self.active_sample.pop(sample_id)
                logger.debug(f"finished for samples: {sample_id}")
                batch_agent_output.append(ReduceResp(raw_prompt=None, agent_loop_output_list=joiner_buffer[sample_id]))
                gen_batch_proto_list.append(_sample.gen_batch)
                batch_proto_list.append(_sample.batch)
                counter += 1
        print("[MicroBatchChatCompletionScheduler] _gather_result done for one batch，do collact function")
        batch = concat_data_proto(batch_proto_list)
        gen_batch = concat_data_proto(gen_batch_proto_list)
        return batch_agent_output, gen_batch, batch

    async def _run_agent_loop(
        self, agent_name: str, messages: List[Dict[str, Any]], sampling_params: Dict[str, Any], proxy_mgr
    ) -> AgentLoopOutput:
        agent_loop_class = get_agent_loop_class(agent_name)
        agent_loop = agent_loop_class(self.original_config, proxy_mgr, self.tokenizer)
        output = await agent_loop.run(messages, sampling_params)
        return output

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

    async def stream_generate_sequences(
        self, data_iter: Iterable, batch_size: int, renew=False
    ) -> Tuple[bool, DataProto, DataProto, DataProto]:
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
        batch_conversations, gen_batch, batch = await self.reduce_handler(
            self.buffer_size, n_sample=self.config.n, format=self.reduce_format
        )
        print(f"partial rollout done, cancel all left request, real size: {len(batch_conversations)}")
        await self.cancel_all_req()
        self._requeue_preempt_req()
        self.done_sample_counter += len(batch_conversations)
        print(
            f"[MicroBatchChatCompletionScheduler] generate_sequences done with {len(batch_conversations)} samples, \
            done_sample_counter: {self.done_sample_counter}"
        )
        # TODO make an adaptor,unpack the agent_loop list and make sure they correspond with the batch order
        agent_loop_output_list = [
            item for reduce_resp in batch_conversations for item in reduce_resp.agent_loop_output_list
        ]
        gen_batch_output = agent_loop_postprocess(
            self.tokenizer,
            agent_loop_output_list,
            self.max_prompt_length,
            self.max_response_length,
        )
        self.last_batch_sanity_check(last_batch)
        metrics = [gen_batch_output.meta_info["metrics"]]  # List[List[Dict[str, str]]]
        timing = agent_loop_perf(metrics, gen_batch_output)

        gen_batch_output.meta_info = {"timing": timing}
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
        assert len(self.pending_sample) == 0, f"pending_sample not empty, {self.pending_sample}"
        assert len(self.active_sample) == 0, f"active_sample not empty, {self.active_sample}"
        assert self.global_data_queue.empty(), "global_data_queue not empty"
        assert self.reduce_data_queue.empty(), "reduce_data_queue not empty"
