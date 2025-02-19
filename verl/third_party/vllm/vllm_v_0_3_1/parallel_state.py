# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023 The vLLM team.
# Adapted from
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/parallel_state.py
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
"""Model and data parallel groups."""

import torch
import torch.distributed

import vllm.model_executor.parallel_utils.parallel_state as ps
"""
This version is strongly tied with Megatron to implement HybridEngine and weight sharing between vllm and Megatron.
- We assume the Megatron tp+dp+pp world is already established before calling this function.

"""

# Tensor model parallel group that the current rank belongs to.
_TENSOR_MODEL_PARALLEL_GROUP = None

# Micro Data parallel group. Micro data parallel group is additional dp group that origins from splitting training tp
# into infer_tp and micro_tp. By default, we use order micro_dp - tp
_MICRO_DATA_PARALLEL_GROUP = None


def initialize_model_parallel_from_megatron(
        tensor_model_parallel_size=None  # we set None for backward compatibility to set infer_tp = train_tp
) -> None:
    from megatron.core import parallel_state as mpu
    from megatron.distributed import new_group
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()

    if tensor_model_parallel_size is None:
        tensor_model_parallel_size = mpu.get_tensor_model_parallel_world_size()
    else:
        assert isinstance(tensor_model_parallel_size, int)

    # Build the tensor model-parallel groups.
    assert ps._TENSOR_MODEL_PARALLEL_GROUP is None, ("tensor model parallel group is already initialized")

    assert tensor_model_parallel_size <= mpu.get_tensor_model_parallel_world_size(
    ), 'Not implemented for infer_tp > train_tp'

    global _TENSOR_MODEL_PARALLEL_GROUP
    global _MICRO_DATA_PARALLEL_GROUP

    assert mpu.get_tensor_model_parallel_world_size() % tensor_model_parallel_size == 0

    micro_dp_size = mpu.get_tensor_model_parallel_world_size() // tensor_model_parallel_size

    world_size: int = torch.distributed.get_world_size()

    num_micro_dp_groups = world_size // micro_dp_size

    rank = torch.distributed.get_rank()

    # Build the micro dp groups.
    assert _MICRO_DATA_PARALLEL_GROUP is None, ("micro data parallel group is already initialized")
    for i in range(num_micro_dp_groups):
        ranks = range(i * micro_dp_size, (i + 1) * micro_dp_size)
        group = new_group(rank=rank, ranks=ranks, group_type='micro_dp')
        if rank in ranks:
            _MICRO_DATA_PARALLEL_GROUP = group

    if tensor_model_parallel_size == mpu.get_tensor_model_parallel_world_size():
        # using the same tp group as Megatron
        ps._TENSOR_MODEL_PARALLEL_GROUP = mpu.get_tensor_model_parallel_group()

        _TENSOR_MODEL_PARALLEL_GROUP = mpu.get_tensor_model_parallel_group()
        # no _MICRO_DATA_PARALLEL_GROUP
    else:
        # initialize a micro_dp group and a tp group
        # assume training tp=4, infer tp=2, then, weight is partitioned as
        # [1], [2], [3], [4] for training and [1,2], [1,2], [3,4], [3,4] for inference

        # Build the inference tp groups
        train_tp = mpu.get_tensor_model_parallel_world_size()
        num_tensor_model_parallel_groups_per_train_tp = train_tp // tensor_model_parallel_size
        num_tensor_model_parallel_groups = world_size // tensor_model_parallel_size
        assert _TENSOR_MODEL_PARALLEL_GROUP is None, ("tensor model parallel group is already initialized")
        for i in range(num_tensor_model_parallel_groups // num_tensor_model_parallel_groups_per_train_tp):
            start = train_tp * i
            end = train_tp * (i + 1)
            for j in range(num_tensor_model_parallel_groups_per_train_tp):
                ranks = list(range(start, end, num_tensor_model_parallel_groups_per_train_tp))
                for i in range(len(ranks)):
                    ranks[i] += j
                # group = torch.distributed.new_group(ranks)
                group = new_group(rank=rank, ranks=ranks, group_type='infer_tp')
                if rank in ranks:
                    _TENSOR_MODEL_PARALLEL_GROUP = group
                    ps._TENSOR_MODEL_PARALLEL_GROUP = _TENSOR_MODEL_PARALLEL_GROUP
    # Build the pipeline model-parallel groups.
    # global _PIPELINE_MODEL_PARALLEL_GROUP
    # global _PIPELINE_GLOBAL_RANKS
    # assert ps._PIPELINE_MODEL_PARALLEL_GROUP is None, ("pipeline model parallel group is already initialized")

    # ps._PIPELINE_MODEL_PARALLEL_GROUP = mpu.get_pipeline_model_parallel_group()
    # ps._PIPELINE_GLOBAL_RANKS = mpu.get_pipeline_model_parallel_ranks()


"""
Tensor model parallel utilities
"""


def get_tensor_model_parallel_group():
    """Get the tensor model parallel group the caller rank belongs to."""
    assert _TENSOR_MODEL_PARALLEL_GROUP is not None, ("tensor model parallel group is not initialized")
    return _TENSOR_MODEL_PARALLEL_GROUP


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


"""
Micro Data parallel group
"""


def get_micro_data_parallel_group():
    assert _MICRO_DATA_PARALLEL_GROUP is not None
    return _MICRO_DATA_PARALLEL_GROUP


def get_micro_data_parallel_world_size():
    return torch.distributed.get_world_size(group=get_micro_data_parallel_group())


def get_micro_data_parallel_rank():
    return torch.distributed.get_rank(group=get_micro_data_parallel_group())
