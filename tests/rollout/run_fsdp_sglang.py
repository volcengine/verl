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

import os
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision, CPUOffload
from torch.distributed.fsdp.api import ShardingStrategy, ShardedStateDictConfig, StateDictType
import torch
import multiprocessing

from verl.utils.fsdp_utils import offload_fsdp_model_to_cpu
from verl.utils.debug import log_gpu_memory_usage

from sglang.srt.entrypoints.verl_engine import VerlEngine


def main():
    assert torch.cuda.is_available(), 'CUDA must be present to run FSDP sglang example'
    tensor_model_parallel_size = 4

    torch.distributed.init_process_group()
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
    print(f"local_rank: {local_rank}, rank: {rank}, world_size: {world_size}, device: {torch.cuda.current_device()}")

    ## manually set the mp authkey
    multiprocessing.current_process().authkey = b'123456'

    local_cache_path = '~/.cache/verl/rlhf'
    local_cache_path = os.path.expanduser(local_cache_path)
    hdfs_path = 'Qwen/Qwen2-7B-Instruct'

    from verl.utils.fs import copy_to_local
    local_model_path = copy_to_local(src=hdfs_path, cache_dir=local_cache_path)
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
    actor_model_config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=True)
    with torch.device("cuda"):
        actor_model = AutoModelForCausalLM.from_pretrained(local_model_path, trust_remote_code=True)
        actor_model.to(torch.bfloat16)

    max_prompt_length = 16
    response_length = 32
    preencode_prompts = [
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    tokenizer.pad_token = tokenizer.eos_token
    prompts = tokenizer(preencode_prompts, return_tensors='pt', padding=True, padding_side='left')
    input_ids = prompts['input_ids']
    attention_mask = prompts['attention_mask']
    from verl.utils.torch_functional import pad_sequence_to_length
    input_ids = pad_sequence_to_length(input_ids, max_prompt_length, tokenizer.pad_token_id, left_pad=True).cuda()
    attention_mask = pad_sequence_to_length(attention_mask, max_prompt_length, 0, left_pad=True).cuda()

    from transformers import GenerationConfig
    generation_config = GenerationConfig(do_sample=False)
    actor_model.cuda()
    output = actor_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=32,
        # max_length=max_length,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        generation_config=generation_config,
        # renormalize_logits=True,
        output_scores=False,  # this is potentially very large
        return_dict_in_generate=True,
        use_cache=False)  # may OOM when use_cache = True
    seq = output.sequences
    response = seq[:, max_prompt_length:]
    hf_output = tokenizer.batch_decode(response)

    from torch.distributed.device_mesh import init_device_mesh
    device_mesh = init_device_mesh('cuda', mesh_shape=(world_size,), mesh_dim_names=['fsdp'])

    mixed_precision = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32)
    fsdp_model = FSDP(actor_model,
                      use_orig_params=True,
                      auto_wrap_policy=None,
                      device_id=torch.cuda.current_device(),
                      sharding_strategy=ShardingStrategy.FULL_SHARD,
                      mixed_precision=mixed_precision,
                      cpu_offload=CPUOffload(offload_params=True),
                      sync_module_states=False,
                      device_mesh=device_mesh)

    FSDP.set_state_dict_type(fsdp_model,
                             state_dict_type=StateDictType.SHARDED_STATE_DICT,
                             state_dict_config=ShardedStateDictConfig())

    actor_model.cpu()
    torch.cuda.empty_cache()

    sampling_params = {
        "max_new_tokens": response_length,
        "temperature": 0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
        "skip_special_tokens": True,
        "ignore_eos": False,
    }

    print(actor_model_config)

    ## sglang need to see all cuda devices
    for k in ['CUDA_VISIBLE_DEVICES', 'TORCHELASTIC_USE_AGENT_STORE']:
        if k in os.environ:
            print(f"del os.environ[{k}]")
            del os.environ[k]

    # init device mesh
    tp_size = tensor_model_parallel_size
    device_mesh_kwargs = dict(mesh_shape=(world_size // tp_size, tp_size, 1), mesh_dim_names=["dp", "tp", "pp"])
    print(f"device_mesh_kwargs: {device_mesh_kwargs}")
    device_mesh_cpu = init_device_mesh("cpu", **device_mesh_kwargs)

    # get tp_rank of this process in this tp group
    global_rank = device_mesh_cpu.get_rank()
    tp_size = device_mesh_cpu["tp"].mesh.size()[0]
    src_rank = global_rank // tp_size * tp_size

    llm = VerlEngine(
        model_path=local_model_path,
        load_format='dummy',
        dtype='bfloat16',
        mem_fraction_static=0.5,
        enable_memory_saver=True,
        # attention_backend='torch_native',
        # sampling_backend='pytorch',
        # disable_cuda_graph=True,
        # disable_custom_all_reduce=True,
        device_mesh_cpu=device_mesh_cpu["tp"],
        base_gpu_id=src_rank,
        gpu_id_step=1,
    )
    log_gpu_memory_usage('Before release memory occupation', None)
    llm.release_memory_occupation()
    torch.distributed.barrier()
    torch.cuda.empty_cache()
    log_gpu_memory_usage('After release memory occupation', None)

    offload_fsdp_model_to_cpu(fsdp_model)
    state_dict = fsdp_model.state_dict()
    for k, v in state_dict.items():
        state_dict[k] = v.cpu()
    log_gpu_memory_usage('After load fsdp model', None)

    llm.resume_memory_occupation()
    torch.distributed.barrier()
    log_gpu_memory_usage('After resume memory occupation', None, rank=1)

    llm.update_weights_from_tensor([(k, v) for k, v in state_dict.items()], load_format=None)
    log_gpu_memory_usage('After sync model weights', None)
    del state_dict
    torch.cuda.empty_cache()
    log_gpu_memory_usage('After del state_dict and empty_cache', None)

    input_ids = input_ids.cuda()
    attention_mask = attention_mask.cuda()
    idx_list = []
    batch_size = input_ids.shape[0]

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    from verl.workers.rollout.sglang_rollout.sglang_rollout import _pre_process_inputs
    for i in range(batch_size):
        idx_list.append(_pre_process_inputs(pad_token_id, input_ids[i]))

    outputs = llm.generate(prompt=None, input_ids=idx_list, sampling_params=sampling_params, return_logprob=False)
    sglang_output = [o['text'] for o in outputs]

    if llm._engine is not None:
        print(f"server args: {llm._engine.server_args}")
        log_gpu_memory_usage('Before free cache engine', None)
        llm._engine.tokenizer_manager.flush_cache()
        log_gpu_memory_usage('After free cache engine', None)
    llm.shutdown()

    if torch.distributed.get_rank() == 0:
        print(f"hf response: {hf_output}")
        print(f"sglang response: {sglang_output}")

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
