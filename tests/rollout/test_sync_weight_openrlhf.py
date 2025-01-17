from datetime import timedelta
from typing import Any, Optional, Union

import time
import os
import ray
import socket
import torch
import torch.distributed
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from torch.distributed.distributed_c10d import (
    Backend,
    PrefixStore,
    Store,
    _new_process_group_helper,
    _world,
    default_pg_timeout,
    rendezvous,
)
from vllm.worker.worker import Worker
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM


# Copy from pytorch to allow creating multiple main groups.
# https://github.com/pytorch/pytorch/blob/main/torch/distributed/distributed_c10d.py
def init_process_group(
    backend: Union[str, Backend] = None,
    init_method: Optional[str] = None,
    timeout: Optional[timedelta] = None,
    world_size: int = -1,
    rank: int = -1,
    store: Optional[Store] = None,
    group_name: str = None,
    pg_options: Optional[Any] = None,
):
    assert (store is None) or (init_method is None), "Cannot specify both init_method and store."

    if store is not None:
        assert world_size > 0, "world_size must be positive if using store"
        assert rank >= 0, "rank must be non-negative if using store"
    elif init_method is None:
        init_method = "env://"

    if backend:
        backend = Backend(backend)
    else:
        backend = Backend("undefined")

    if timeout is None:
        timeout = default_pg_timeout

    # backward compatible API
    if store is None:
        rendezvous_iterator = rendezvous(init_method, rank, world_size, timeout=timeout)
        store, rank, world_size = next(rendezvous_iterator)
        store.set_timeout(timeout)

        # Use a PrefixStore to avoid accidental overrides of keys used by
        # different systems (e.g. RPC) in case the store is multi-tenant.
        store = PrefixStore(group_name, store)

    pg, _ = _new_process_group_helper(
        world_size,
        rank,
        [],
        backend,
        store,
        group_name=group_name,
        pg_options=pg_options,
        timeout=timeout,
    )

    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}

    return pg


class WorkerWrap(Worker):
    def init_process_group(self, master_address, master_port, rank_offset, world_size, group_name, backend="nccl"):
        """Init torch process group for model weights update"""
        assert torch.distributed.is_initialized(), "default torch process group must be initialized"
        assert group_name != "", "group name must not be empty"

        rank = torch.distributed.get_rank() + rank_offset
        self._model_update_group = init_process_group(
            backend=backend,
            init_method=f"tcp://{master_address}:{master_port}",
            world_size=world_size,
            rank=rank,
            group_name=group_name,
        )
        print(
            f"init_process_group: master_address={master_address}, master_port={master_port}, ",
            f"rank={rank}, world_size={world_size}, group_name={group_name}",
        )

    def update_weight(self, name, dtype, shape, empty_cache=False):
        """Broadcast weight to all vllm workers from source rank 0 (actor model)"""
        # print(f"update_weight: {name}, dtype: {dtype}, shape: {shape}, rank: {torch.distributed.get_rank()}, world_size: {torch.distributed.get_world_size()}")
        # if torch.distributed.get_rank() == 0:
        #     print(f"update weight: {name}, dtype: {dtype}, shape: {shape}")

        assert dtype == self.model_config.dtype, f"mismatch dtype: src {dtype}, dst {self.model_config.dtype}"
        weight = torch.empty(shape, dtype=dtype, device="cuda")
        torch.distributed.broadcast(weight, 0, group=self._model_update_group)

        self.model_runner.model.load_weights(weights=[(name, weight)])

        del weight
        # TODO: should we empty cache if all weights have updated?
        # if empty_cache:
        #     torch.cuda.empty_cache()


@ray.remote
class LLMRayActor:
    def __init__(self, *args, **kwargs):
        import vllm

        self.__version__ = vllm.__version__
        assert self.__version__ >= "0.4.1", "OpenRLHF only supports vLLM >= 0.4.1"

        self.use_gpu_executor = kwargs["tensor_parallel_size"] == 1

        # See https://github.com/vllm-project/vllm/blob/main/vllm/executor/gpu_executor.py
        if self.use_gpu_executor:

            vllm.worker.worker.Worker = WorkerWrap
        else:
            # RayGPUExecutor
            # See the patch https://github.com/vllm-project/vllm/commit/479d69fad0538f04cb22bf13e76ff91cfeb8a4e5
            kwargs["worker_use_ray"] = True

            if vllm.__version__ > "0.4.1":
                RayWorkerWrapperPath = vllm.executor.ray_utils
            else:
                RayWorkerWrapperPath = vllm.engine.ray_utils

            class RayWorkerWrapper(RayWorkerWrapperPath.RayWorkerWrapper):
                def __init__(self, *args, **kwargs) -> None:
                    kwargs["worker_module_name"] = "test_sync_weight_openrlhf"
                    kwargs["worker_class_name"] = "WorkerWrap"
                    super().__init__(*args, **kwargs)

            RayWorkerWrapperPath.RayWorkerWrapper = RayWorkerWrapper

        self.llm = vllm.LLM(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.llm.generate(*args, **kwargs)

    def init_process_group(self, master_address, master_port, rank_offset, world_size, group_name, backend):
        if self.use_gpu_executor:
            return self.llm.llm_engine.model_executor.driver_worker.init_process_group(
                master_address, master_port, rank_offset, world_size, group_name, backend
            )
        else:
            return self.llm.llm_engine.model_executor._run_workers(
                "init_process_group", master_address, master_port, rank_offset, world_size, group_name, backend
            )

    def update_weight(self, name, dtype, shape, empty_cache=False):
        self.stop_remote_worker_execution_loop()

        if self.use_gpu_executor:
            return self.llm.llm_engine.model_executor.driver_worker.update_weight(name, dtype, shape, empty_cache)
        else:
            return self.llm.llm_engine.model_executor._run_workers("update_weight", name, dtype, shape, empty_cache)

    def stop_remote_worker_execution_loop(self):
        # Fix error for using 2 communication group
        # https://github.com/vllm-project/vllm/commit/eb6d3c264d0cd8e44dec16bca7947fbe96415ce9#diff-e1ad69e38e033accddfa5480ec808c4740eb39244d1ef51cc3407e20dde8cfd4
        if self.__version__ > "0.4.2":
            self.llm.llm_engine.model_executor.stop_remote_worker_execution_loop()


def create_vllm_engines(
    num_engines: int,
    tensor_parallel_size: int,
    pretrain: str,
    revision: str,
    seed: int,
    enable_prefix_caching: bool,
    max_model_len: int,
):
    vllm_engines = []
    for i in range(num_engines):
        # When tensor_parallel_size=1, vLLM init model in LLMEngine directly, assign 1 GPU for it.
        num_gpus = int(tensor_parallel_size == 1)
        scheduling_strategy = None

        if tensor_parallel_size > 1:
            bundles = [{"GPU": 1, "CPU": 1}] * tensor_parallel_size
            pg = placement_group(bundles)
            ray.get(pg.ready())

            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg, placement_group_capture_child_tasks=True, placement_group_bundle_index=0
            )
        print(f"vllm: {num_gpus=}, {num_engines=}")
        vllm_engines.append(
            LLMRayActor.options(
                num_cpus=1,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
            ).remote(
                pretrain,
                revision=revision,
                tokenizer_revision=revision,
                trust_remote_code=True,
                tensor_parallel_size=tensor_parallel_size,
                dtype="bfloat16",
                seed=seed + i,
                enable_prefix_caching=enable_prefix_caching,
                max_model_len=max_model_len,
            )
        )

    return vllm_engines


if __name__ == "__main__":

    local_cache_path = '~/.cache/verl/rlhf'
    local_cache_path = os.path.expanduser(local_cache_path)
    hdfs_path = 'Qwen/Qwen2-7B-Instruct'

    from verl.utils.fs import copy_local_path_from_hdfs
    local_model_path = copy_local_path_from_hdfs(src=hdfs_path, cache_dir=local_cache_path)
    # tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
    # actor_model_config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=True)
    # actor_model = AutoModelForCausalLM.from_pretrained(local_model_path, trust_remote_code=True)

    # llm = LLMRayActor.remote(model=actor_model, tokenizer=tokenizer, tensor_parallel_size=2)
    # # llm = LLMRayActor.remote("meta-llama/Llama-3.1-8B-Instruct", tensor_parallel_size=2)
    # output = ray.get(llm.generate.remote("San Franciso is a"))
    # print(f"output: {output}")

    vllm_tensor_parallel_size = 4
    vllm_num_engines = 1
    vllm_sync_backend = "nccl"
    # llm = LLMRayActor.remote("meta-llama/Llama-3.1-8B-Instruct", tensor_parallel_size=2)
    # output = ray.get(llm.generate.remote("San Franciso is a"))
    # print(f"output: {output}")
    
    vllm_engines = create_vllm_engines(
        vllm_num_engines,
        vllm_tensor_parallel_size,
        local_model_path,
        None,
        1,
        False,
        4096,
    )

    master_address = ray._private.services.get_node_ip_address()
    with socket.socket() as sock:
        sock.bind(("", 0))
        master_port = sock.getsockname()[1]
    vllm_num_engines, vllm_tensor_parallel_size = (
        vllm_num_engines,
        vllm_tensor_parallel_size,
    )
    world_size = vllm_num_engines * vllm_tensor_parallel_size + 1
    backend = vllm_sync_backend
    # https://github.com/OpenRLHF/OpenRLHF/issues/313
    # if vllm.__version__ > "0.4.2" and os.getenv("NCCL_P2P_DISABLE", "0") == "0":
    #     backend = "gloo"
    #     print(
    #         "Warning: using --vllm_sync_backend=gloo for vLLM version > 0.4.2 (or export NCCL_P2P_DISABLE=1)"
    #     )
    refs = [
        engine.init_process_group.remote(
            master_address,
            master_port,
            i * vllm_tensor_parallel_size + 1,
            world_size,
            "openrlhf",
            backend=backend,
        )
        for i, engine in enumerate(vllm_engines)
    ]
    model_update_group = init_process_group(
        backend=backend,
        init_method=f"tcp://{master_address}:{master_port}",
        world_size=world_size,
        rank=0,
        group_name="openrlhf",
    )
    ray.get(refs)
    torch.set_default_device("cuda:7")
    model = AutoModelForCausalLM.from_pretrained(local_model_path, torch_dtype=torch.bfloat16)
    model = model.to("cuda:7")
    def broadcast_to_vllm():
        # avoid OOM
        torch.cuda.empty_cache()
        count, num_params = 0, len(list(model.named_parameters()))
        refss = []
        for name, param in model.named_parameters():
            count += 1
            shape = param.shape
            refs = [
                engine.update_weight.remote(
                    name, dtype=param.dtype, shape=shape, empty_cache=count == num_params
                )
                for engine in vllm_engines
            ]
            refss.extend(refs)
            torch.distributed.broadcast(param.data, 0, group=model_update_group)
        ray.get(refss)

    # Warmup iterations
    for _ in range(10):
        torch.cuda.synchronize()
        broadcast_to_vllm()
        torch.cuda.synchronize()


    # Profile the function
    start_time = time.time()
    broadcast_to_vllm()
    torch.cuda.synchronize()  # Ensure all CUDA operations are complete
    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print(f"Weight sync time taken: {elapsed_time:.6f} seconds")
    print("broadcasted model to vllm")