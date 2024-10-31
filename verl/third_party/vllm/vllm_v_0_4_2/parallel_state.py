# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023 The vLLM team.
# Adapted from
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/parallel_state.py
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
"""Model and data parallel groups."""
import os
import torch
import torch.distributed
from typing import Optional

import vllm.distributed.parallel_state as ps

import vllm.envs as envs
from vllm.logger import init_logger

from torch.distributed.device_mesh import init_device_mesh

logger = init_logger(__name__)
"""
This version is strongly tied with Megatron to implement HybridEngine and weight sharing between vllm and Megatron.
- We assume the Megatron tp+dp+pp world is already established before calling this function.

"""

# Device mesh for using DTensor
_DEVICE_MESH = None

# Tensor model parallel group that the current rank belongs to.
_TP_DEVICE_GROUP = None
_TP_CPU_GROUP = None


# This method is for initializing the ParallelGroup when using HybridEngine
def initialize_parallel_state(
    distributed_init_method: str = "env://",
    backend: str = "nccl",
    tensor_model_parallel_size: int = 1,
    num_tp_per_train_tp: int = 1,
    pipeline_model_parallel_size: int = 1,
):
    # torch.distributed.all_reduce does not free the input tensor until
    # the synchronization point. This causes the memory usage to grow
    # as the number of all_reduce calls increases. This env var disables
    # this behavior.
    # Related issue:
    # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

    # NOTE(sgm): Modify for verl, Env vars will be set by TORCHRUN.
    rank = int(os.getenv("RANK", "-1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))

    # Use the world_size set by TORCHRUN
    world_size = int(os.getenv("WORLD_SIZE", "-1"))
    assert world_size != -1, "The world_size is set to -1, not initialized by TORCHRUN"
    ps.init_distributed_environment(world_size, rank, distributed_init_method, local_rank, backend)
    if torch.distributed.get_world_size() > 1:
        # NOTE: build a sepearate inference group with infer tp & micro dp
        initialize_model_parallel_for_vllm(tensor_model_parallel_size=tensor_model_parallel_size,
                                           num_tensor_model_parallel_groups_per_train_tp=num_tp_per_train_tp)
    else:
        initialize_model_parallel(tensor_model_parallel_size, pipeline_model_parallel_size, backend)


def ensure_model_parallel_initialized(
    tensor_model_parallel_size: int,
    pipeline_model_parallel_size: int = 1,
    backend: Optional[str] = None,
) -> None:
    """Helper to initialize model parallel groups if they are not initialized,
    or ensure tensor-parallel and pipeline-parallel sizes are equal to expected
    values if the model parallel groups are initialized.
    """
    # get the backend of _DEVICE_WORLD_GROUP
    backend = backend or torch.distributed.get_backend()
    if not model_parallel_is_initialized():
        initialize_model_parallel(tensor_model_parallel_size, pipeline_model_parallel_size, backend)
        return

    assert (get_tensor_model_parallel_world_size() == tensor_model_parallel_size), (
        "tensor parallel group already initialized, but of unexpected size: "
        f"{get_tensor_model_parallel_world_size()=} vs. "
        f"{tensor_model_parallel_size=}")
    # assert (get_pipeline_model_parallel_world_size(
    # ) == pipeline_model_parallel_size), (
    #     "pipeline parallel group already initialized, but of unexpected size: "
    #     f"{get_pipeline_model_parallel_world_size()=} vs. "
    #     f"{pipeline_model_parallel_size=}")


def model_parallel_is_initialized():
    """Check if tensor and pipeline parallel groups are initialized."""
    return (ps._TP_DEVICE_GROUP is not None)
    # and _PIPELINE_MODEL_PARALLEL_GROUP is not None)


def initialize_model_parallel_for_vllm(tensor_model_parallel_size: int,
                                       num_tensor_model_parallel_groups_per_train_tp: int = 1) -> None:
    from torch.distributed import new_group
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()

    assert isinstance(tensor_model_parallel_size, int)

    # assert num_tensor_model_parallel_groups_per_train_tp == 1 and not different_tp_group
    # assert num_tensor_model_parallel_groups_per_train_tp > 1 and different_tp_group

    # Build the tensor model-parallel groups.
    assert ps._TP_DEVICE_GROUP is None, ("tensor model parallel group is already initialized")

    global _TP_DEVICE_GROUP
    global _TP_CPU_GROUP
    global _DEVICE_MESH

    world_size: int = torch.distributed.get_world_size()

    rank = torch.distributed.get_rank()

    backend = torch.distributed.get_backend()

    num_tensor_model_parallel_groups = world_size // tensor_model_parallel_size

    if num_tensor_model_parallel_groups_per_train_tp == 1:
        # if tensor_model_parallel_size == train_tensor_parallel_size:
        # using the same tp group as Megatron/vllm
        for i in range(num_tensor_model_parallel_groups):
            ranks = range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size)
            group = torch.distributed.new_group(ranks, backend=backend)
            cpu_group = torch.distributed.new_group(ranks, backend="gloo")
            if rank in ranks:
                _TP_DEVICE_GROUP = group
                _TP_CPU_GROUP = cpu_group
                ps._TP_DEVICE_GROUP = group
                ps._TP_CPU_GROUP = cpu_group

        # no _MICRO_DATA_PARALLEL_GROUP
    else:
        # initialize a micro_dp group and a tp group
        # assume training tp=4, infer tp=2, then, weight is partitioned as
        # [1], [2], [3], [4] for training and [1,2], [1,2], [3,4], [3,4] for inference

        # Build the inference tp groups
        # train_tp = train_tensor_parallel_size
        train_tp = num_tensor_model_parallel_groups_per_train_tp * tensor_model_parallel_size
        # num_tensor_model_parallel_groups_per_train_tp = train_tp // tensor_model_parallel_size
        assert _TP_DEVICE_GROUP is None, ("tensor model parallel group is already initialized")
        for i in range(num_tensor_model_parallel_groups // num_tensor_model_parallel_groups_per_train_tp):
            start = train_tp * i
            end = train_tp * (i + 1)
            for j in range(num_tensor_model_parallel_groups_per_train_tp):
                ranks = list(range(start, end, num_tensor_model_parallel_groups_per_train_tp))
                for i in range(len(ranks)):
                    ranks[i] += j
                group = torch.distributed.new_group(ranks)
                cpu_group = torch.distributed.new_group(ranks, backend='gloo')
                if rank in ranks:
                    _TP_DEVICE_GROUP = group
                    _TP_CPU_GROUP = cpu_group
                    ps._TP_DEVICE_GROUP = _TP_DEVICE_GROUP
                    ps._TP_CPU_GROUP = cpu_group

    # Build the pipeline model-parallel groups.
    # global _PIPELINE_MODEL_PARALLEL_GROUP
    # global _PIPELINE_GLOBAL_RANKS
    # assert ps._PIPELINE_MODEL_PARALLEL_GROUP is None, ("pipeline model parallel group is already initialized")

    # ps._PIPELINE_MODEL_PARALLEL_GROUP = mpu.get_pipeline_model_parallel_group()
    # ps._PIPELINE_GLOBAL_RANKS = mpu.get_pipeline_model_parallel_ranks()


def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    backend: Optional[str] = None,
) -> None:
    """
    NOTE: This method is a hack from the open-sourced version without
    asertion of world_size = tp * pp
    
    Initialize model parallel groups.

    Arguments:
        tensor_model_parallel_size: number of GPUs used for tensor model
            parallelism.
        pipeline_model_parallel_size: number of GPUs used for pipeline model
            parallelism.

    Let's say we have a total of 8 GPUs denoted by g0 ... g7 and we
    use 2 GPUs to parallelize the model tensor, and 4 GPUs to parallelize
    the model pipeline. The present function will
    create 4 tensor model-parallel groups and 2 pipeline model-parallel groups:
        4 tensor model-parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7]
        2 pipeline model-parallel groups:
            [g0, g2, g4, g6], [g1, g3, g5, g7]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.
    """
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()
    # get the backend of _DEVICE_WORLD_GROUP
    backend = backend or torch.distributed.get_backend()

    # NOTE(sgm) we don't assert world_size == tp * pp
    # DP is not managed by vllm but by the veRL WorkerGroup

    num_tensor_model_parallel_groups: int = (world_size // tensor_model_parallel_size)
    num_pipeline_model_parallel_groups: int = (world_size // pipeline_model_parallel_size)
    rank = torch.distributed.get_rank()

    # Build device mesh for TP
    if num_tensor_model_parallel_groups > 1:
        device_mesh = init_device_mesh("cuda", (num_tensor_model_parallel_groups, tensor_model_parallel_size),
                                       mesh_dim_names=("replicate", "tp_shard"))
    else:
        device_mesh = init_device_mesh("cuda", (tensor_model_parallel_size,), mesh_dim_names=["tp_shard"])
    shard_group = device_mesh.get_group(mesh_dim="tp_shard")

    # Build the tensor model-parallel groups.
    global _TP_DEVICE_GROUP, _TP_CPU_GROUP
    global _DEVICE_MESH
    assert _TP_DEVICE_GROUP is None, ("tensor model parallel group is already initialized")
    assert _DEVICE_MESH is None, ("device mesh in vllm is already initialized")

    _DEVICE_MESH = device_mesh
    # for i in range(num_tensor_model_parallel_groups):
    #     ranks = range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size)
    # group = torch.distributed.new_group(ranks, backend=backend)
    # cpu_group = torch.distributed.new_group(ranks, backend="gloo")
    # assert torch.distributed.get_process_group_ranks(shard_group) == torch.distributed.get_process_group_ranks(cpu_group)
    # ranks = torch.distributed.get_process_group_ranks(shard_group)
    # cpu_group = torch.distributed.new_group(ranks, backend="gloo") # TODO: this will hang
    # cpu_group = torch.distributed.new_group(, backend="gloo")
    # if rank == 0:
    #     print(f'rank: {rank}')
    #     print(f'ranks: {ranks}')
    #     print(f'torch.distributed.get_process_group_ranks(shard_group): {torch.distributed.get_process_group_ranks(shard_group)}')
    # if rank in ranks:
    _TP_DEVICE_GROUP = shard_group
    ps._TP_DEVICE_GROUP = _TP_DEVICE_GROUP
    # ps._TP_CPU_GROUP = cpu_group # TODO: will hang when used with device mesh

    # TODO: init using device mesh
    # Build the pipeline model-parallel groups.
    assert ps._PIPELINE_MODEL_PARALLEL_GROUP is None, ("pipeline model parallel group is already initialized")
    for i in range(num_pipeline_model_parallel_groups):
        ranks = range(i, world_size, num_pipeline_model_parallel_groups)
        group = torch.distributed.new_group(ranks, backend=backend)
        if rank in ranks:
            ps._PIPELINE_MODEL_PARALLEL_GROUP = group
            ps._PIPELINE_GLOBAL_RANKS = ranks


"""
Device mesh utilities
"""


def get_device_mesh():
    assert _DEVICE_MESH is not None, ("device mesh is not initialized")
    return _DEVICE_MESH


"""
Tensor model parallel utilities
"""


def get_tensor_model_parallel_group():
    """Get the tensor model parallel group the caller rank belongs to."""
    assert _TP_DEVICE_GROUP is not None, ("tensor model parallel group is not initialized")
    return _TP_DEVICE_GROUP


def get_tensor_model_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    return torch.distributed.get_world_size(group=get_tensor_model_parallel_group())


def get_tensor_model_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    return torch.distributed.get_rank(group=get_tensor_model_parallel_group())


def get_tensor_model_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank
    in the tensor model parallel group."""
    global_rank = torch.distributed.get_rank()
    local_world_size = get_tensor_model_parallel_world_size()
    return (global_rank // local_world_size) * local_world_size
