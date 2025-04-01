import os
import time
import socket
import torch
from contextlib import closing

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
from sglang.srt.utils import broadcast_pyobj
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_local_ip():
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    return local_ip

def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

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

def main():
    assert torch.cuda.is_available(), "CUDA not available"
    print("Start init")
    local_rank, rank, world_size = initialize_global_process_group()
    print(f"RANK{rank}", local_rank, rank, world_size)
    
    "============================================= Parallel ============================================="
    dp, tp, pp = 1, 4, 1
    # dp, tp, pp = 2, 4, 1
    device_count = 2
    model_name = "Qwen/Qwen2-7B-Instruct"
    # model_name = "deepseek-ai/deepseek-llm-7b-chat"
    "============================================= Parallel ============================================="
    
    # tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    "============================================= FSDP ============================================="
    device_mesh = init_device_mesh(
        "cuda", mesh_shape=(dp*tp,), mesh_dim_names=["fsdp"]
    )
    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.float32,
    )
    with torch.device("cuda"):
        actor_model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True
        )
        actor_model.to(torch.bfloat16)
    fsdp_model = FSDP(
        actor_model,
        use_orig_params=True,
        auto_wrap_policy=None,
        device_id=torch.cuda.current_device(),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision,
        cpu_offload=CPUOffload(offload_params=False),
        sync_module_states=False,
        device_mesh=device_mesh,
    )
    FSDP.set_state_dict_type(
        fsdp_model,
        state_dict_type=StateDictType.SHARDED_STATE_DICT,
        state_dict_config=ShardedStateDictConfig(),
    )

    state_dict = fsdp_model.state_dict()
    "============================================= FSDP ============================================="
    
    "============================================= INIT PARALLEL ============================================="
    kwargs = dict(
        mesh_shape=(dp, tp, pp), 
        mesh_dim_names=["dp", "tp", "pp"]
    )
    inference_device_mesh_cpu = init_device_mesh("cpu", **kwargs)
    tp_rank = inference_device_mesh_cpu["tp"].get_local_rank()

    # NOTE: for each dp we have a TP0, within a TP group we need ip:port from TP0's worker for rdzv
    # see torch.distributed.init_process_group
    (ip, port) = (get_local_ip(), find_free_port()) if tp_rank == 0 else (None, None)

    [ip, port] = broadcast_pyobj(
        [ip, port],
        rank=tp_rank,
        dist_group=inference_device_mesh_cpu.get_group("tp"),
        src=inference_device_mesh_cpu["tp"].mesh[0].item()
    )
    # floor
    world_size = int(os.environ["WORLD_SIZE"])
    tp_size = tp
    nnodes = -(-tp_size // device_count)
    print(f"RANK{rank}: {ip}:{port} nnodes:{nnodes}")
    "============================================= INIT PARALLEL ============================================="

    # NOTE: otherwise would fail
    for k in ["TORCHELASTIC_USE_AGENT_STORE"]:
        if k in os.environ:
            del os.environ[k]
    
    print("Building VerlEngine...")
    llm = VerlEngine(
        model_path=model_name,  # use model of same type but different weight to test update_weights
        dtype="bfloat16",
        mem_fraction_static=0.3,
        device_mesh_cpu=inference_device_mesh_cpu["tp"],
        base_gpu_id=0,
        gpu_id_step=1,
        enable_memory_saver=True,
        trust_remote_code=True,
        dist_init_addr=f'{ip}:{port}',
        nnodes=nnodes,
    )
    print('call release_memory_occupation', flush=True)
    llm.release_memory_occupation()
    print('sleep...', flush=True)
    time.sleep(3)
    print('call resume_memory_occupation', flush=True)
    llm.resume_memory_occupation()

    print("updating rollout weights")
    llm.update_weights_from_tensor([(k, v) for k, v in state_dict.items()])

    "============================================= GEN ============================================="
    sampling_params = dict(
        temperature=0, top_p=1, n=1, max_new_tokens=16, ignore_eos=True
    )
    outputs = llm.generate("Introduce yourself.", sampling_params=sampling_params)

    if inference_device_mesh_cpu["tp"].get_local_rank() == 0:
        print("="*64)
        print(f'SGlang response: {outputs["text"]}')
        print("="*64)
    "============================================= GEN ============================================="
    

# torchrun --nnodes=2 --nproc_per_node=2 --master_addr=<NODE0 IP> --master_port=34567 --node_rank 0 torchrun_verlengine.py
# torchrun --nnodes=2 --nproc_per_node=2 --master_addr=<NODE0 IP> --master_port=34567 --node_rank 1 torchrun_verlengine.py
if __name__ == "__main__":
    main()