from typing import Any, Optional, List, Tuple

import torch
from torch import Tensor
import torch.distributed as dist
from torch.distributed import ProcessGroup

_ULYSSES_SEQUENCE_PARALLEL_GROUP = None

def set_ulysses_sequence_parallel_group(group: dist.ProcessGroup):
    """
    Set ulysses sequence parallel process group.
    """
    global _ULYSSES_SEQUENCE_PARALLEL_GROUP
    _ULYSSES_SEQUENCE_PARALLEL_GROUP = group

def get_ulysses_sequence_parallel_group() -> Optional[dist.ProcessGroup]:
    """
    Get ulysses sequence parallel process group.
    """
    global _ULYSSES_SEQUENCE_PARALLEL_GROUP
    assert _ULYSSES_SEQUENCE_PARALLEL_GROUP is not None, ("tensor model parallel group is not initialized")
    return _ULYSSES_SEQUENCE_PARALLEL_GROUP

def get_ulysses_sequence_parallel_world_size(group: ProcessGroup = None) -> int:
    """
    Get ulysses sequence parallel world size.
    """
    group = get_ulysses_sequence_parallel_group() if group is None else group
    return dist.get_world_size(group) if group else 1

def gather_seq_scatter_heads(
    x: Tensor,
    seq_dim: int,
    head_dim: int,
    unpadded_dim_size: int = 0,
    group: ProcessGroup = None,
) -> Tensor:
    """
    A func to sync embedding input with alltoall in sequence parallel
    gather sequence dimension and scatter head dim:
    e.g. seq_dim: 1, head_dim: 2
    [bsz, seq/n, h, ...] -> [bsz, seq, h/n, ...]
    """
    group = get_ulysses_sequence_parallel_group() if group is None else group
    if not group:
        return x
    sp_world = get_ulysses_sequence_parallel_world_size(group)
    x = SeqAllToAll.apply(group, x, head_dim, seq_dim)
    if unpadded_dim_size and unpadded_dim_size % sp_world != 0:
        padding_size = x.size(seq_dim) - unpadded_dim_size
        x = _unpad_tensor(x, seq_dim, padding_size)
        return x

def gather_heads_scatter_seq(x: Tensor, head_dim: int, seq_dim: int, group: ProcessGroup = None) -> Tensor:
    """
    A func to sync attention result with alltoall in sequence parallel
    gather head dimension and scatter seq dim:
    e.g. seq_dim: 1, head_dim: 2
    [bsz, seq, h/n, ...] -> [bsz, seq/n, h, ...]
    """
    group = get_ulysses_sequence_parallel_group() if group is None else group
    if not group:
        return x
    dim_size = x.size(seq_dim)
    sp_world = get_ulysses_sequence_parallel_world_size(group)
    if dim_size % sp_world != 0:
        padding_size = sp_world - (dim_size % sp_world)
        x = _pad_tensor(x, seq_dim, padding_size)
    return SeqAllToAll.apply(group, x, seq_dim, head_dim, False)

def _pad_tensor(x: Tensor, dim: int, padding_size: int) -> Tensor:
    shape = list(x.shape)
    shape[dim] = padding_size
    pad = torch.zeros(shape, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=dim)

def _unpad_tensor(x: Tensor, dim: int, padding_size: int) -> Tensor:
    slc = [slice(None)] * len(x.shape)
    slc[dim] = slice(0, -padding_size)
    return x[slc]

def all_to_all_tensor(local_tensor: Tensor, 
                    scatter_dim: int, 
                    gather_dim: int, 
                    group: Optional[dist.ProcessGroup] = None,
                    async_op: bool = False):
    group = get_ulysses_sequence_parallel_group() if group is None else group
    seq_world_size = dist.get_world_size(group=group)
    # scatter input
    input_tensor_list = [ t for t in torch.tensor_split(local_tensor, seq_world_size, dim=scatter_dim)]
    # gather output
    output_tensor_list = [ torch.empty_like(input_tensor_list[0]) for _ in range(seq_world_size) ]
    dist.all_to_all(output_tensor_list, input_tensor_list, group=group, async_op=async_op)
    return torch.cat(output_tensor_list, dim=gather_dim)


class SeqAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, 
                group: dist.ProcessGroup, 
                local_tensor: Tensor, 
                scatter_dim: int, 
                gather_dim: int, 
                async_op=False) -> Tensor:
        ctx.group = group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.async_op = async_op
        return all_to_all_tensor(local_tensor, scatter_dim, gather_dim, group, async_op)

    @staticmethod
    def backward(ctx: Any, 
                 *grad_outputs: Tensor
                 ) -> Tuple[None, Tensor, None, None, None]:
        return (
            None,
            all_to_all_tensor(grad_outputs[0], ctx.gather_dim, ctx.scatter_dim, ctx.group, ctx.async_op),
            None,
            None,
            None,
            None
        )