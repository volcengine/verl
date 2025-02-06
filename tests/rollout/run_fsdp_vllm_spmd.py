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

import torch.distributed as dist
from vllm.distributed.parallel_state import get_world_group
from verl.utils.distributed import initialize_global_process_group
from vllm import LLM
from verl.third_party.vllm.vllm_v_0_6_3.dtensor_weight_loaders import load_dtensor_weights

from vllm import SamplingParams
import time


def main():
    assert torch.cuda.is_available(), 'CUDA must be present to run FSDP vLLM example'
    local_rank, rank, world_size = initialize_global_process_group()

    local_cache_path = '~/.cache/verl/rlhf'
    local_cache_path = os.path.expanduser(local_cache_path)
    hdfs_path = 'Qwen/Qwen2-7B-Instruct'

    from verl.utils.fs import copy_local_path_from_hdfs
    local_model_path = copy_local_path_from_hdfs(src=hdfs_path, cache_dir=local_cache_path)
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
    prompts = tokenizer(preencode_prompts, return_tensors='pt', padding=True)
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

    print(f'hf response: {tokenizer.batch_decode(response)}')

    tensor_model_parallel_size = 4
    from torch.distributed.device_mesh import init_device_mesh
    device_mesh = init_device_mesh('cuda', mesh_shape=(world_size,), mesh_dim_names=['fsdp'])

    mixed_precision = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32)
    fsdp_model = FSDP(actor_model,
                      use_orig_params=True,
                      auto_wrap_policy=None,
                      device_id=torch.cuda.current_device(),
                      sharding_strategy=ShardingStrategy.FULL_SHARD,
                      mixed_precision=mixed_precision,
                      cpu_offload=CPUOffload(offload_params=False),
                      sync_module_states=False,
                      device_mesh=device_mesh)

    FSDP.set_state_dict_type(fsdp_model,
                             state_dict_type=StateDictType.SHARDED_STATE_DICT,
                             state_dict_config=ShardedStateDictConfig())

    state_dict = fsdp_model.state_dict()    
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    # print(actor_model_config)
    llm = LLM(model=local_model_path,
              enable_sleep_mode=True,
              tensor_parallel_size=tensor_model_parallel_size,
              distributed_executor_backend="external_launcher",
              dtype='bfloat16',
              gpu_memory_utilization=0.7)

    # # Warmup iterations
    # for _ in range(10):
    #     torch.cuda.synchronize()
    #     load_dtensor_weights(state_dict, llm.llm_engine.model_executor.driver_worker.worker.model_runner.model)
    #     torch.cuda.synchronize()
    #     dist.barrier()

    empty_state_dict = {}
    for key, value in state_dict.items():
        empty_state_dict[key] = torch.zeros_like(value)

    llm.sleep(level=2)
    llm.wake_up()
    load_dtensor_weights(empty_state_dict, llm.llm_engine.model_executor.driver_worker.worker.model_runner.model)
    load_dtensor_weights(state_dict, llm.llm_engine.model_executor.driver_worker.worker.model_runner.model)

    outputs = llm.generate(preencode_prompts, sampling_params)
    cpu_group = get_world_group().cpu_group
    torch_rank = dist.get_rank(group=cpu_group)

    def test_consistent_across_ranks(obj):
        if torch_rank == 0:
            dist.broadcast_object_list([obj], src=0, group=cpu_group)
        else:
            container = [None]
            dist.broadcast_object_list(container, src=0, group=cpu_group)
            assert container[0] == obj


    test_consistent_across_ranks(
        llm.llm_engine.vllm_config.cache_config.num_cpu_blocks)
    test_consistent_across_ranks(
        llm.llm_engine.vllm_config.cache_config.num_gpu_blocks)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        test_consistent_across_ranks(prompt)
        test_consistent_across_ranks(generated_text)
        print(f"Rank {torch_rank}, Prompt: {prompt!r}, "
            f"Generated text: {generated_text!r}")


if __name__ == "__main__":
    main()
