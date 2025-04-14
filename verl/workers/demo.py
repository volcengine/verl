import torch
import torch.distributed


def allgather_from_megatron_tp(tensor, dim):
    from megatron.core import parallel_state as mpu
    rank = mpu.get_tensor_model_parallel_rank()
    world_size = mpu.get_tensor_model_parallel_world_size()
    tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
    tensor_list[rank] = tensor
    torch.distributed.all_gather(tensor_list, tensor, group=mpu.get_tensor_model_parallel_group())
    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=dim).contiguous()
    return output


def broadcast_from_megatron_pp(tensor: torch.Tensor):
    from megatron.core import parallel_state as mpu
    # tensor is not None only in one of the pp ranks

    if tensor is not None:
        shape = tensor.shape
        dtype = tensor.dtype
        tensor_spec = (shape, dtype)
    else:
        tensor_spec = None

    tensor_spec_output = [None] * mpu.get_pipeline_model_parallel_world_size()
    torch.distributed.all_gather_object(object_list=tensor_spec_output,
                                        obj=tensor_spec,
                                        group=mpu.get_pipeline_model_parallel_group())

    # find the src rank
    target_tensor_spec = None
    src_rank = None

    for rank, tensor_spec in enumerate(tensor_spec_output):
        if tensor_spec is not None:
            if target_tensor_spec is None:
                target_tensor_spec = tensor_spec
            else:
                raise ValueError('A tensor exists on two pp ranks')
            src_rank = rank

    assert target_tensor_spec is not None

    if tensor is None:
        tensor = torch.empty(size=target_tensor_spec[0],
                             dtype=target_tensor_spec[1],
                             device=torch.cuda.current_device())

    global_rank = torch.distributed.get_global_rank(group=mpu.get_pipeline_model_parallel_group(), group_rank=src_rank)

    torch.distributed.broadcast(tensor=tensor, src=global_rank, group=mpu.get_pipeline_model_parallel_group())
    return tensor