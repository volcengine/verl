import asyncio
import copy
import os
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import ray
import torch
import torch.distributed as dist
import vllm.distributed.parallel_state as vllm_ps
import zmq
import zmq.asyncio
from omegaconf import DictConfig
from tensordict import TensorDict
from transformers import AutoConfig
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.executor.executor_base import DistributedExecutorBase
from vllm.executor.ray_utils import RayWorkerWrapper
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import ExecuteModelRequest
from vllm.utils import make_async, run_method

from verl import DataProto
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.torch_functional import get_response_mask
from verl.workers.rollout.base import BaseRollout

from .parallel_state import initialize_candidate_ctx, set_default_ctx


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id
    # is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


# Monkey patch for AsyncLLMEngine to support sleep and wakeup operations.
def sleep(self, level: int = 1):
    """
    Put the engine to sleep. The engine should not process any requests.
    The caller should guarantee that no requests are being processed
    during the sleep period, before `wake_up` is called.

    :param level: The sleep level. Level 1 sleep will offload the model
        weights and discard the kv cache. The content of kv cache is
        forgotten. Level 1 sleep is good for sleeping and waking up the
        engine to run the same model again. The model weights are backed
        up in CPU memory. Please make sure there's enough CPU memory to
        store the model weights. Level 2 sleep will discard both the model
        weights and the kv cache. The content of both the model weights
        and kv cache is forgotten. Level 2 sleep is good for sleeping and
        waking up the engine to run a different model or update the model,
        where previous model weights are not needed. It reduces CPU memory
        pressure.
    """
    self.engine.reset_prefix_cache()
    self.engine.sleep(level=level)


def wake_up(self):
    """
    Wake up the engine from sleep mode. See the :meth:`sleep` method
    for more details."""
    self.engine.wake_up()


AsyncLLMEngine.sleep = sleep
AsyncLLMEngine.wake_up = wake_up


def debug(msg: str):
    rank_str = ""
    if torch.distributed.is_initialized():
        rank_str = f"[rank:{torch.distributed.get_rank()}]"
    format_str = f"[DEBUG][{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]{rank_str} {msg}"
    print(format_str, flush=True)


class ExternalRayDistributedExecutor(DistributedExecutorBase):
    """An executor that uses Ray to launch engines,
    specially designed for torchrun-compatible launchers, for
    offline inference with tensor parallelism.
    """

    uses_ray: bool = True

    def _init_executor(self) -> None:
        """Initialize the worker and load the model."""
        print(f"{sorted(ray.util.list_named_actors())=}")
        fields = self.vllm_config.instance_id.split(":")
        # namespace, wg_prefix, global_rank = fields[0], fields[1], int(fields[2])
        _, _, global_rank = fields[0], fields[1], int(fields[2])

        assert self.vllm_config.parallel_config.pipeline_parallel_size == 1, "ExternalRayDistributedExecutor does not support pipeline parallelism."
        assert self.vllm_config.scheduler_config.delay_factor == 0.0, "ExternalRayDistributedExecutor needs deterministic execution, so itdoes not support delay_factor in scheduling"
        # assert not envs.VLLM_USE_V1, \
        #     ("V1 architecture cannot guarantee deterministic execution, "
        #     "so it is not supported in ExternalRayDistributedExecutor.")
        self.driver_worker = RayWorkerWrapper(vllm_config=self.vllm_config, rpc_rank=0)
        # engines are launched in torchrun-compatible launchers
        # so we can use the env:// method.
        # required env vars:
        # - RANK
        # - MASTER_ADDR
        # - MASTER_PORT
        distributed_init_method = "env://"
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        is_driver_worker = torch.distributed.get_rank() % self.parallel_config.tensor_parallel_size == 0
        kwargs = dict(
            vllm_config=self.vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )
        self.collective_rpc("init_worker", args=([kwargs],))
        self.collective_rpc("init_device")
        self.collective_rpc("load_model")

        actor_names = sorted(ray.util.list_named_actors())
        workers = [ray.get_actor(actor_name) for actor_name in actor_names]
        # This is the list of workers that are rank 0 of each TP group EXCEPT
        # global rank 0. These are the workers that will broadcast to the
        # rest of the workers.
        self.tp_driver_workers: List[RayWorkerWrapper] = []
        # This is the list of workers that are not drivers and not the first
        # worker in a TP group. These are the workers that will be
        # broadcasted to.
        self.non_driver_workers: List[RayWorkerWrapper] = []
        # FIXME(lkm):
        if is_driver_worker:
            self.non_driver_workers = workers[global_rank + 1 : global_rank + self.parallel_config.tensor_parallel_size]
            debug(f"is driver worker, non_driver_workers range: [{global_rank + 1} : {global_rank + self.parallel_config.tensor_parallel_size}]")

        self.parallel_worker_tasks: Optional[Union[Any, Awaitable[Any]]] = None
        self.driver_exec_method = make_async(self.driver_worker.execute_method)

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """
        Determine the number of available KV blocks.
        Add an additional all_reduce to get the min across all ranks.
        Note that even if we have the same `gpu_memory_utilization` and
        `swap_space`, the available memory in every rank might still
        differ because NCCL can take different amounts of memory in
        different ranks. Therefore, it is necessary to test if all ranks
        agree on the same KV cache configuration.
        """
        a, b = super().determine_num_available_blocks()
        from vllm.distributed.parallel_state import get_world_group

        cpu_group = get_world_group().cpu_group
        a_tensor = torch.tensor([a], device="cpu", dtype=torch.int64)
        b_tensor = torch.tensor([b], device="cpu", dtype=torch.int64)
        dist.all_reduce(a_tensor, group=cpu_group, op=dist.ReduceOp.MIN)
        dist.all_reduce(b_tensor, group=cpu_group, op=dist.ReduceOp.MIN)
        return a_tensor.item(), b_tensor.item()

    async def execute_model_async(self, execute_model_req: ExecuteModelRequest) -> List[SamplerOutput]:
        if self.parallel_worker_tasks is None:
            # Start model execution loop running in the parallel workers
            self.parallel_worker_tasks = asyncio.create_task(self._start_worker_execution_loop())

        # Only the driver worker returns the sampling results.
        return await self._driver_execute_model_async(execute_model_req)

    def _driver_execute_model(self, execute_model_req: Optional[ExecuteModelRequest]) -> Optional[List[SamplerOutput]]:
        """Run execute_model in the driver worker.

        Passing None will cause the driver to stop the model execution
        loop running in each of the remote workers.
        """
        return self.driver_worker.execute_method("execute_model", execute_model_req)

    def _run_workers(
        self,
        method: Union[str, Callable],
        *args,
        async_run_tensor_parallel_workers_only: bool = False,
        max_concurrent_workers: Optional[int] = None,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers. Can be used in the following
        ways:

        Args:
        - async_run_tensor_parallel_workers_only: If True the method will be
          run only in the remote TP workers, not the driver worker.
          It will also be run asynchronously and return a list of futures
          rather than blocking on the results.
        - args/kwargs: All workers share the same args/kwargs
        """
        if kwargs is None:
            kwargs = {}
        answer = run_method(self.driver_worker, method, args, kwargs)
        return [answer]

    def _wait_for_tasks_completion(self, parallel_worker_tasks: Any) -> None:
        """Wait for futures returned from _run_workers() with
        async_run_remote_workers_only to complete."""
        raise NotImplementedError

    async def _driver_execute_model_async(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None,
    ) -> List[SamplerOutput]:
        if not self.tp_driver_workers:
            try:
                return await self.driver_exec_method("execute_model", execute_model_req)
            except Exception as e:
                debug(f"tp_size:{vllm_ps.get_tensor_model_parallel_world_size()} Failed to execute model: {e}")
                raise e

    async def _start_worker_execution_loop(self):
        coros = [worker.actor_rollout_execute_method.remote("start_worker_execution_loop") for worker in self.non_driver_workers]
        return await asyncio.gather(*coros)

    def check_health(self) -> None:
        # Assume that the Ray workers are healthy.
        # TODO: check the health of the Ray workers
        return


class vLLMDynRollout(BaseRollout):
    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        self.config = config
        assert not config.enforce_eager, "vLLM dyn rollout only supports non-eager mode"

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)
        self.tensor_parallel_size = tensor_parallel_size

        rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
        if not rope_scaling_config:
            max_position_embeddings = None
            if hasattr(model_hf_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.max_position_embeddings
            elif hasattr(model_hf_config, "llm_config") and hasattr(model_hf_config.llm_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.llm_config.max_position_embeddings
            elif hasattr(model_hf_config, "text_config") and hasattr(model_hf_config.text_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.text_config.max_position_embeddings
            if max_position_embeddings is None:
                raise ValueError("max_position_embeddings not found in model_hf_config")

            assert max_position_embeddings >= config.prompt_length + config.response_length, "model context length should be greater than total sequence length"

        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )

        trust_remote_code = kwargs.get("trust_remote_code", False)
        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format

        model_config = AutoConfig.from_pretrained(model_path)
        num_kv_heads = getattr(model_config, "num_key_value_heads", None)
        tp_size_candidates = [i for i in range(1, 8) if num_kv_heads % i == 0 and i & (i - 1) == 0]
        self.candidated_inference_engine: Dict[int, AsyncLLMEngine] = {}
        self.rank = torch.distributed.get_rank()
        for tp_size in tp_size_candidates:
            with initialize_candidate_ctx(tp_size=tp_size):
                debug(f"Create inference engine with tp_size: {tp_size}")
                engine_args = AsyncEngineArgs(
                    model=model_path,
                    enable_sleep_mode=True,
                    tensor_parallel_size=tp_size,
                    distributed_executor_backend=ExternalRayDistributedExecutor,
                    dtype=config.dtype,
                    enforce_eager=config.enforce_eager,
                    gpu_memory_utilization=config.gpu_memory_utilization,
                    disable_custom_all_reduce=True,
                    disable_mm_preprocessor_cache=False,
                    skip_tokenizer_init=False,
                    max_model_len=max_model_len,
                    load_format=load_format,
                    disable_log_stats=config.disable_log_stats,
                    max_num_batched_tokens=max_num_batched_tokens,
                    enable_chunked_prefill=config.enable_chunked_prefill,
                    enable_prefix_caching=False,
                    trust_remote_code=trust_remote_code,
                    seed=config.get("seed", 0),
                )
                vllm_config = engine_args.create_engine_config()
                namespace = ray.get_runtime_context().namespace
                vllm_config.instance_id = f"{namespace}:rollout_worker:{self.rank}"
                inference_engine = AsyncLLMEngine.from_vllm_config(vllm_config)

                # Offload vllm model to reduce peak memory usage
                inference_engine.sleep(level=1)
                self.candidated_inference_engine[tp_size] = inference_engine

        set_default_ctx(tp_size=self.tensor_parallel_size)
        self.inference_engine = self.candidated_inference_engine[self.tensor_parallel_size]
        debug(f"RolloutWorker {self.rank} init done, candidate tp_size: {self.candidated_inference_engine.keys()}, default tp_size: {self.tensor_parallel_size}")

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        # replace vllm batching with manual request repeating
        # instead of letting vllm copy prompts internally, we now manually send N
        # separate requests to enable:
        # - better distribution across rollout instances
        # - independent processing per repeat
        self.n = 1
        if kwargs["n"] > 1:
            self.n = kwargs["n"]
            kwargs.pop("n")

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.tokenizer = tokenizer

        # below is used for scale up
        self.lock = asyncio.Lock()
        self._generate_task = None

        self.cancel_event = asyncio.Event()
        self.resume_event = asyncio.Event()
        self.scale_up_complete = asyncio.Event()

    def setup_buffer(self, buffer_addr: str):
        # connect to zmq buffer
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.PUSH)
        if "]" in buffer_addr:
            self.socket.setsockopt(zmq.IPV6, 1)
        self.socket.connect(buffer_addr)

    async def _send_result(self, uid: str, response: List[int]):
        await self.socket.send_json(
            {
                "uid": uid,
                "response": response,
            }
        )

    async def scale_up(self):
        num_unfinished_requests = self.get_num_unfinished_requests()
        if num_unfinished_requests > 0 and not (self._generate_task is None or self._generate_task.done()):
            debug("Scale up, cancel running generate task...")
            self.cancel_event.set()

            await self.resume_event.wait()
            self.resume_event.clear()

            # NOTE: the executor proc may be not finished, so we need to wait for it
            # sleep a while
            await asyncio.sleep(0.2)

        debug("Starting scale_up")
        # prev_tp_size = vllm_ps.get_tensor_model_parallel_world_size()
        prev_tp_rank = vllm_ps.get_tensor_model_parallel_rank()
        next_tp_size = vllm_ps.get_tensor_model_parallel_world_size() * 2
        self.inference_engine.sleep(level=1)
        self.inference_engine = self.candidated_inference_engine[next_tp_size]
        self.inference_engine.wake_up()
        debug("Scale_up complete")
        await make_async(vllm_ps.get_tp_group().barrier)()
        set_default_ctx(tp_size=next_tp_size)
        debug(f"Set tp_size={next_tp_size}")

        if num_unfinished_requests > 0 and not (self._generate_task is None or self._generate_task.done()):
            self.scale_up_complete.set()
        else:
            if prev_tp_rank == 0:
                if vllm_ps.get_tensor_model_parallel_rank() == 0:
                    # i am the new driver, create a new generate task
                    # debug(f"i am the new driver, create a new generate task")
                    self.generate_sequences(prompts=None)
                else:
                    # i am not the new driver, send an empty task list to driver
                    # debug(f"i am not the new driver, send an empty task list to driver")
                    await self.send_tasks_to_driver(pending_tasks=[])

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    async def _generate(self, prompts: DataProto, **kwargs) -> Any:
        if prompts:
            idx = prompts.batch["input_ids"]  # (bs, prompt_length)
            uids = prompts.non_tensor_batch["uid"]
            assert len(idx) == len(uids)

            batch_size = idx.size(0)

            non_tensor_batch = prompts.non_tensor_batch
            if "raw_prompt_ids" not in non_tensor_batch:
                non_tensor_batch["raw_prompt_ids"] = np.array([_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)

            if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
                raise RuntimeError("vllm sharding manager is not work properly.")

            vllm_inputs = [{"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")]

            # ensure the type of `prompt_token_ids` passed to vllm is list[int]
            # https://github.com/volcengine/verl/pull/772
            for input_data in vllm_inputs:
                if isinstance(input_data["prompt_token_ids"], np.ndarray):
                    input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
                elif not isinstance(input_data["prompt_token_ids"], list):
                    raise TypeError(f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}")

            # TODO(lkm):
            # do_sample = prompts.meta_info.get("do_sample", True)
            # is_validate = prompts.meta_info.get("validate", False)
            # if not do_sample:
            #     kwargs = {
            #         "best_of": 1,
            #         "top_p": 1.0,
            #         "top_k": -1,
            #         "min_p": 0.0,
            #         "temperature": 0,
            #         "n": 1,  # if greedy, only 1 response
            #     }
            # elif is_validate:
            #     # TODO: try **
            #     kwargs = {
            #         "top_k": self.config.val_kwargs.top_k,
            #         "top_p": self.config.val_kwargs.top_p,
            #         "temperature": self.config.val_kwargs.temperature,
            #         "n": 1,  # if validate, already repeat in ray_trainer
            #     }

            # self.update_sampling_params(**kwargs).__enter__()

            tasks = [asyncio.create_task(self._generate_sequence_task(uid, input_data, input_data["prompt_token_ids"], self.sampling_params)) for uid, input_data in zip(uids, vllm_inputs) for _ in range(self.n)]
        else:
            tasks = await self.recv_tasks_on_driver()

        debug(f"Start {len(tasks)} tasks")

        try:
            while tasks:
                if self.cancel_event.is_set():
                    # abort tasks
                    pending_tasks = []
                    finished_tasks = []

                    for task in tasks:
                        if not task.done():
                            task.cancel()
                            pending_tasks.append(task)
                        else:
                            finished_tasks.append(task)
                    tasks.clear()
                    debug(f"Cancel {len(pending_tasks)} tasks, {len(finished_tasks)} tasks have already finished.")

                    if pending_tasks:
                        # await asyncio.gather(*pending_tasks)
                        await asyncio.gather(*(pending_tasks + finished_tasks))

                    self.resume_event.set()

                    # do scale_up
                    await self.scale_up_complete.wait()
                    self.scale_up_complete.clear()
                    self.cancel_event.clear()

                    debug(f"Resume {len(pending_tasks)} tasks, {len(finished_tasks)} tasks have already finished.")
                    tasks = [] + finished_tasks
                    if vllm_ps.get_tensor_model_parallel_rank() == 0:
                        # resume tasks
                        for task in pending_tasks:
                            uid, temp_output, prompt_token_ids = await task
                            for output in temp_output.outputs:
                                sampling_params = copy.deepcopy(self.sampling_params)
                                sampling_params.n = 1
                                sampling_params.max_tokens = sampling_params.max_tokens - len(output.token_ids)
                                input_data = {"prompt_token_ids": temp_output.prompt_token_ids + list(output.token_ids)}
                                task = asyncio.create_task(self._generate_sequence_task(uid, input_data, prompt_token_ids, sampling_params))
                                tasks.append(task)

                        # receive pending_tasks from pre driver workers
                        tasks = tasks + await self.recv_tasks_on_driver()
                    else:
                        await self.send_tasks_to_driver(pending_tasks)

                    debug(f"Still have {len(tasks)} tasks...")

                # normal case
                done = []
                for task in tasks:
                    if task.done():
                        done.append(task)

                for task in done:
                    if not task.cancelled():
                        uid, result, org_prompt_token_ids = await task
                        for output in result.outputs:
                            input_ids = result.prompt_token_ids + list(output.token_ids)
                            # prompt = input_ids[: len(org_prompt_token_ids)]
                            response = input_ids[len(org_prompt_token_ids) :]
                            await self._send_result(uid, response)

                tasks = [task for task in tasks if not task.done()]

                await asyncio.sleep(0.1)

        finally:
            # Cleanup any remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            self._generate_task = None

    async def recv_tasks_on_driver(self):
        tasks = []
        # receive pending_tasks from pre driver workers
        src_rank = torch.distributed.get_rank() + vllm_ps.get_tensor_model_parallel_world_size() // 2
        debug(f"Receive pending_tasks from pre driver workers: rank={src_rank}")
        num_tasks_tensor = torch.zeros(1, dtype=torch.int32, device="cuda")
        torch.distributed.recv(num_tasks_tensor, src=src_rank)
        n_pending_tasks = num_tasks_tensor.item()
        for i in range(n_pending_tasks):
            # 1. recvice the uid
            objects = [None]
            torch.distributed.recv_object_list(objects, src=src_rank)
            uid = objects[0]

            # 2. receive the shape
            shape_tensor = torch.zeros(2, dtype=torch.int32, device="cuda")  # shape_tensor: [prompt_token_ids_len, output_tokens_ids_len]
            torch.distributed.recv(shape_tensor, src=src_rank)

            # 3. receive the token_ids
            token_ids = torch.zeros(shape_tensor.sum(), dtype=torch.long, device="cuda")
            torch.distributed.recv(token_ids, src=src_rank)
            token_ids = token_ids.tolist()

            # 4. receive the origin prompt_token_ids shape and data
            org_prompt_shape_tensor = torch.zeros(1, dtype=torch.int32, device="cuda")
            torch.distributed.recv(org_prompt_shape_tensor, src=src_rank)
            org_prompt_token_ids = torch.zeros(org_prompt_shape_tensor[0], dtype=torch.long, device="cuda")
            torch.distributed.recv(org_prompt_token_ids, src=src_rank)
            org_prompt_token_ids = org_prompt_token_ids.tolist()

            sampling_params = copy.deepcopy(self.sampling_params)
            sampling_params.n = 1
            sampling_params.max_tokens = sampling_params.max_tokens - shape_tensor[1].item()
            input_data = {"prompt_token_ids": token_ids}
            task = asyncio.create_task(self._generate_sequence_task(uid, input_data, org_prompt_token_ids, sampling_params))
            tasks.append(task)
            # debug(f"Recevive task {i+1}/{n_pending_tasks} from rank:{src_rank}, "
            #         f"org_prompt_token_ids_len: {org_prompt_shape_tensor[0].item()}, "
            #         f"prompt_token_ids_len: {shape_tensor[0].item()}, output_tokens_ids_len: {shape_tensor[1].item()}")
        return tasks

    async def send_tasks_to_driver(self, pending_tasks):
        # send pending_tasks from driver workers
        dst_rank = torch.distributed.get_rank() - vllm_ps.get_tensor_model_parallel_world_size() // 2
        # debug(f"Send pending_tasks to driver workers: rank={dst_rank}")
        n_pending_tasks = len(pending_tasks)
        num_tasks_tensor = torch.tensor([n_pending_tasks], dtype=torch.int32, device="cuda")
        torch.distributed.send(num_tasks_tensor, dst=dst_rank)
        for task in pending_tasks:
            uid, temp_output, org_prompt_token_ids = await task
            for output in temp_output.outputs:
                # 1. send the uid
                torch.distributed.send_object_list([uid], dst=dst_rank)

                # 2. send the shape
                shape = [len(temp_output.prompt_token_ids), len(output.token_ids)]
                shape_tensor = torch.tensor(shape, dtype=torch.int32, device="cuda")
                torch.distributed.send(shape_tensor, dst=dst_rank)

                # 3. send the token_ids
                token_ids = torch.tensor(temp_output.prompt_token_ids + list(output.token_ids), dtype=torch.long, device="cuda")
                torch.distributed.send(token_ids, dst=dst_rank)

                # 4. send the origin prompt_token_ids shape and data
                shape = [len(org_prompt_token_ids)]
                shape_tensor = torch.tensor(shape, dtype=torch.int32, device="cuda")
                torch.distributed.send(shape_tensor, dst=dst_rank)
                org_prompt_token_ids_tensor = torch.tensor(org_prompt_token_ids, dtype=torch.long, device="cuda")
                torch.distributed.send(org_prompt_token_ids_tensor, dst=dst_rank)

    async def _generate_sequence_task(
        self,
        uid: str,
        input_data: Dict[str, Any],
        prompt_token_ids: List[int],
        sampling_params: SamplingParams,
    ) -> Any:
        final_output = None
        request_id = str(uuid.uuid4())
        try:
            async for output in self.inference_engine.generate(
                prompt=input_data,
                sampling_params=sampling_params,
                request_id=request_id,
            ):
                final_output = output
            return uid, final_output, prompt_token_ids

        except asyncio.CancelledError:
            await self.inference_engine.abort(request_id)
            return uid, final_output, prompt_token_ids

        except Exception as e:
            await self.inference_engine.abort(request_id)
            raise RuntimeError(f"Generation failed for {request_id}: {str(e)}") from e

    def generate_sequences(self, prompts: DataProto) -> None:
        self._generate_task = asyncio.create_task(self._generate(prompts))

    def postprocess_generate_sequences(self, gen_batch_output: DataProto) -> DataProto:
        idx = gen_batch_output.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = gen_batch_output.batch["attention_mask"]
        position_ids = gen_batch_output.batch["position_ids"]
        response = gen_batch_output.batch["responses"]
        seq = torch.cat([idx, response], dim=-1)

        # used to construct attention_mask
        eos_token_id = gen_batch_output.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        non_tensor_batch = gen_batch_output.non_tensor_batch

        do_sample = gen_batch_output.meta_info.get("do_sample", True)

        if self.sampling_params.n > 1 and do_sample:
            raise NotImplementedError

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        # if position_ids.dim() == 3:  # qwen2vl mrope
        #     delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                # "rollout_log_probs": rollout_log_probs,  # we will recompute old log prob with actor
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    def reset_inference_engine(self, level: int = 1):
        if vllm_ps.get_tensor_model_parallel_world_size() != self.tensor_parallel_size:
            self.inference_engine.sleep(level=level)
            self.inference_engine = self.candidated_inference_engine[self.tensor_parallel_size]
            set_default_ctx(self.tensor_parallel_size)
        assert vllm_ps.get_tensor_model_parallel_world_size() == self.tensor_parallel_size
        log_gpu_memory_usage(f"rank:{torch.distributed.get_rank()} After reset_inference_engine")

    def sleep(self, level: int = 1):
        assert vllm_ps.get_tensor_model_parallel_world_size() == self.tensor_parallel_size
        self.inference_engine.sleep(level=level)
        log_gpu_memory_usage(f"rank:{torch.distributed.get_rank()} After sleep")

    def wake_up(self):
        assert vllm_ps.get_tensor_model_parallel_world_size() == self.tensor_parallel_size
        self.inference_engine.wake_up()
        log_gpu_memory_usage(f"rank:{torch.distributed.get_rank()} After wake_up")

    def get_num_unfinished_requests(self):
        return self.inference_engine.engine.get_num_unfinished_requests()

    def get_rank(self):
        return self.rank

    def execute_method(self, method, *args, **kwargs):
        return self.inference_engine.engine.model_executor.driver_worker.execute_method(method, *args, **kwargs)

    def execute_model(self, *args, **kwargs):
        return self.inference_engine.engine.model_executor.driver_worker.execute_model(*args, **kwargs)
