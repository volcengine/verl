import os
import time
import torch

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffload
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.api import (
    ShardedStateDictConfig,
    ShardingStrategy,
    StateDictType,
)

from sglang.srt.entrypoints.verl_engine import VerlEngine
from transformers import AutoTokenizer

def main():
    assert torch.cuda.is_available(), "CUDA not available"
    print("Start init")
    local_rank, rank, world_size = initialize_global_process_group()
    print(f"RANK{rank}", local_rank, rank, world_size)
    
    if rank == 0:
        print(f"rank 0 init success")
    
    model_name = "Qwen/Qwen2-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    dp, tp, pp = 1, 4, 1
    kwargs = dict(mesh_shape=(dp, tp, pp), mesh_dim_names=["dp", "tp", "pp"])
    inference_device_mesh_cpu = init_device_mesh("cpu", **kwargs)
    print(f"{inference_device_mesh_cpu=}")
    for k in ["TORCHELASTIC_USE_AGENT_STORE"]:
        if k in os.environ:
            del os.environ[k]
    llm = VerlEngine(
        model_path=model_name,  # use model of same type but different weight to test update_weights
        dtype="bfloat16",
        mem_fraction_static=0.5,
        device_mesh_cpu=inference_device_mesh_cpu["tp"],
        base_gpu_id=0,
        gpu_id_step=1,
        enable_memory_saver=True,
        # trust_remote_code=True,
        # dist_init_addr='100.102.11.36:45654',
        # nnodes=2,
    )
    print('call release_memory_occupation', flush=True)
    llm.release_memory_occupation()
    print('sleep...', flush=True)
    time.sleep(3)
    print('call resume_memory_occupation', flush=True)
    llm.resume_memory_occupation()
    
    sampling_params = dict(
        temperature=0, top_p=1, n=1, max_new_tokens=16, ignore_eos=True
    )
    outputs = llm.generate("Who is the champion of 2022 World cup?", sampling_params=sampling_params)

    if torch.distributed.get_rank() == 0:
        print(f'SGlang response: {outputs["text"]}')
    
def initialize_global_process_group(timeout_second=36000):
    from datetime import timedelta

    import torch.distributed

    # NOTE MODIFIED should provide backend=None to have nccl+gloo
    # torch.distributed.init_process_group('nccl', timeout=timedelta(seconds=timeout_second))
    torch.distributed.init_process_group(timeout=timedelta(seconds=timeout_second))

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size

# torchrun --nnodes=2 --nproc_per_node=2 --master_addr=100.102.11.36 --master_port=34567 verlengine_multinode.py
# torchrun --nnodes=2 --nproc_per_node=2 --rdzv_id=4567 --rdzv_endpoint='100.102.11.36:34567' verlengine_multinode.py
if __name__ == "__main__":
    
    main()