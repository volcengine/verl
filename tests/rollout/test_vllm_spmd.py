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

import torch
from torch.distributed.fsdp import CPUOffload, MixedPrecision
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import ShardedStateDictConfig, ShardingStrategy, StateDictType
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from verl.utils.distributed import initialize_global_process_group
from verl.utils.torch_functional import pad_sequence_to_length


def levenshtein(s1, s2):
    m, n = len(s1), len(s2)
    # Initialize matrix of zeros
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    # Initialize first column and first row of the matrix
    for i in range(m + 1):
        dp[i][0] = i  # Deletion from s1 to empty string
    for j in range(n + 1):
        dp[0][j] = j  # Insertion to s1 from empty string
    # Compute the Levenshtein distance matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1  # No cost if characters match
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # Deletion
                dp[i][j - 1] + 1,  # Insertion
                dp[i - 1][j - 1] + cost,  # Substitution
            )
    return dp[m][n]


def are_lists_similar(a, b):
    if len(a) != len(b):
        print("The lists are of different lengths.")
        return False

    total_length = 0
    total_diff = 0

    for s1, s2 in zip(a, b):
        max_len = max(len(s1), len(s2))
        total_length += max_len
        diff = levenshtein(s1, s2)
        total_diff += diff
        print(f"Comparing strings:\n{s1}\n{s2}\nDifference: {diff} characters\n")

    percentage_difference = (total_diff / total_length) * 100
    print(f"Total difference: {percentage_difference:.2f}%")

    return percentage_difference <= 15


def test_vllm_spmd():
    assert torch.cuda.device_count() >= 2, "At least 2 GPUs is required to run tp+dp tests."
    local_rank, rank, world_size = initialize_global_process_group()

    # Initialize model and token
    local_cache_path = "~/.cache/verl/rlhf"
    local_cache_path = os.path.expanduser(local_cache_path)
    hdfs_path = "Qwen/Qwen2-7B-Instruct"
    from verl.utils.fs import copy_to_local

    local_model_path = copy_to_local(src=hdfs_path, cache_dir=local_cache_path)
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, padding_side="left", trust_remote_code=True)

    actor_model = AutoModelForCausalLM.from_pretrained(local_model_path, trust_remote_code=True)
    actor_model.to(torch.bfloat16)

    # fill rollout config
    max_prompt_length = 16
    max_response_length = 32
    preencode_prompts = [
        "Who won the Champions League in 2019?",
        "The founder of Apple is",
        "What's your name",
    ]
    tokenizer.pad_token = tokenizer.eos_token
    prompts = tokenizer(preencode_prompts, return_tensors="pt", padding=True)
    input_ids = prompts["input_ids"]
    attention_mask = prompts["attention_mask"]

    input_ids = pad_sequence_to_length(input_ids, max_prompt_length, tokenizer.pad_token_id, left_pad=True)
    attention_mask = pad_sequence_to_length(attention_mask, max_prompt_length, 0, left_pad=True)

    print("start generation")
    input_ids = input_ids.cuda()
    attention_mask = attention_mask.cuda()

    temperature = 0
    top_p = 1
    kwargs = dict(n=1, temperature=temperature, top_p=top_p, max_tokens=max_response_length, logprobs=1, ignore_eos=True)

    tensor_parallel_size = 4

    from torch.distributed.device_mesh import init_device_mesh

    device_mesh = init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=["fsdp"])

    mixed_precision = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32)

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

    FSDP.set_state_dict_type(fsdp_model, state_dict_type=StateDictType.SHARDED_STATE_DICT, state_dict_config=ShardedStateDictConfig())

    state_dict = fsdp_model.state_dict()

    sampling_params = SamplingParams(**kwargs)
    llm = LLM(
        model=local_model_path,
        enable_sleep_mode=True,
        tensor_parallel_size=tensor_parallel_size,
        distributed_executor_backend="external_launcher",
        dtype="bfloat16",
        enforce_eager=True,
        gpu_memory_utilization=0.8,
        disable_custom_all_reduce=True,
        disable_mm_preprocessor_cache=True,
        skip_tokenizer_init=False,
        enable_prefix_caching=True,
        trust_remote_code=True,
        seed=1,
    )

    outputs = llm.generate(preencode_prompts, sampling_params=sampling_params, use_tqdm=False)
    vllm_response_tokens = []
    for output in outputs:
        generated_text = output.outputs[0].text
        vllm_response_tokens.append(generated_text)

    world_size = torch.distributed.get_world_size()
    model = llm.llm_engine.model_executor.driver_worker.worker.model_runner.model
    model.load_weights(((name, param.full_tensor() if world_size != 1 else param) for name, param in state_dict.items()))

    outputs = llm.generate(preencode_prompts, sampling_params=sampling_params, use_tqdm=False)
    verl_vllm_response_tokens = []
    for output in outputs:
        generated_text = output.outputs[0].text
        verl_vllm_response_tokens.append(generated_text)

    if torch.distributed.get_rank() == 0:
        print(f"vllm response: {vllm_response_tokens}")
        print(f"verl-vllm response: {verl_vllm_response_tokens}")
    assert are_lists_similar(vllm_response_tokens, verl_vllm_response_tokens), "Strings differ more than 10%:\n"
    print("Check Pass")
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    test_vllm_spmd()
