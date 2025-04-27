# Copyright 2023-2024 SGLang Team
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
# ==============================================================================
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
from datetime import timedelta

import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from verl.protocol import DataProto
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.model import compute_position_id_with_mask
from verl.utils.torch_functional import pad_sequence_to_length
from verl.workers.rollout.sglang_rollout import SGLangRollout
from verl.workers.sharding_manager.fsdp_sglang import FSDPSGLangShardingManager


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

    return percentage_difference <= 10


def initialize_global_process_group(timeout_second=36000):
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


def test_sglang_rollout():
    """测试 SGLang rollout 的生成能力"""
    # 初始化分布式环境
    assert torch.cuda.device_count() >= 2, "At least 2 GPUs required"
    local_rank, rank, world_size = initialize_global_process_group()

    # fill rollout config
    max_prompt_length = 16
    max_response_length = 16
    dtype = "bfloat16"
    tensor_parallel_size = 2
    CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not CUDA_VISIBLE_DEVICES:
        # CUDA_VISIBLE_DEVICES = ','.join(str(i) for i in range(tensor_parallel_size))
        CUDA_VISIBLE_DEVICES = str(local_rank)
        os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
        print(f"CUDA_VISIBLE_DEVICES is not set, set to {CUDA_VISIBLE_DEVICES}")
    local_model_path = "/user/longxiang1/models/Qwen/Qwen2.5-0.5B"

    sampling_params = dict(
        n=1,
        temperature=0,
        top_p=1,
        top_k=-1,
        max_new_tokens=max_response_length,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
        skip_special_tokens=True,
        spaces_between_special_tokens=True,
        ignore_eos=False,
    )

    rollout_config = OmegaConf.create(
        {
            "name": "sglang",
            "load_format": "dummy_dtensor",
            "enforce_eager": False,
            "free_cache_engine": False,
            "dtype": dtype,
            "gpu_memory_utilization": 0.5,
            "ignore_eos": False,
            "max_num_batched_tokens": 8192,
            "prompt_length": max_prompt_length,
            "response_length": max_response_length,
            "tensor_model_parallel_size": tensor_parallel_size,
            **sampling_params,
        }
    )

    # 准备模型和tokenizer
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    actor_model = AutoModelForCausalLM.from_pretrained(local_model_path, torch_dtype=dtype, device_map="cuda")

    # prepare input data
    preencode_prompts = [
        "Who won the Champions League in 2019?",
        "The founder of Apple is",
        "What's your name?",
    ]
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    prompts = tokenizer(preencode_prompts, return_tensors="pt", padding=True)
    input_ids = prompts["input_ids"]
    attention_mask = prompts["attention_mask"]
    position_ids = compute_position_id_with_mask(attention_mask)
    input_ids = pad_sequence_to_length(input_ids, max_prompt_length, tokenizer.pad_token_id, left_pad=True)
    attention_mask = pad_sequence_to_length(attention_mask, max_prompt_length, pad_token_id=0, left_pad=True)
    position_ids = pad_sequence_to_length(position_ids, max_prompt_length, pad_token_id=0, left_pad=True)

    fsdp_device_mesh = init_device_mesh("cuda", mesh_shape=(tensor_parallel_size,), mesh_dim_names=("fsdp",))

    inference_device_mesh_cpu = init_device_mesh(
        "cpu",
        mesh_shape=(world_size // tensor_parallel_size, tensor_parallel_size, 1),
        mesh_dim_names=("dp", "infer_tp", "pp"),
    )

    # generate HF baseline results
    generation_config = GenerationConfig(do_sample=False)
    output = actor_model.generate(
        input_ids=input_ids.cuda(),
        attention_mask=attention_mask.cuda(),
        max_new_tokens=max_response_length,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        generation_config=generation_config,
        output_scores=False,
        return_dict_in_generate=True,
        use_cache=False,
    )

    seq = output.sequences
    response = seq[:, max_prompt_length:]
    hf_response_tokens = tokenizer.batch_decode(response)
    print(f"HF response: {hf_response_tokens}")

    # initialize FSDP model
    fsdp_model = FSDP(
        actor_model,
        use_orig_params=True,
        device_id=fsdp_device_mesh["fsdp"].get_local_rank(),
        mixed_precision=MixedPrecision(param_dtype=getattr(torch, dtype)),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_mesh=fsdp_device_mesh,
    )
    print(f"FSDP model initialized on device {fsdp_model.device}")
    "======================= torchrun需要删掉这个 ======================="
    for k in ["TORCHELASTIC_USE_AGENT_STORE"]:
        if k in os.environ:
            del os.environ[k]
    "======================= torchrun需要删掉这个 ======================="

    # initialize rollout and sharding manager
    rollout = SGLangRollout(
        actor_module=local_model_path, config=rollout_config, tokenizer=tokenizer, model_hf_config=actor_model.config
    )
    print(f"Rollout initialized on rank {rank}")

    if world_size == 1:
        rollout_config.load_format = "dummy_hf"

    rollout_sharding_manager = FSDPSGLangShardingManager(
        module=fsdp_model,
        inference_engine=rollout.inference_engine,
        model_config=actor_model.config,
        full_params="hf" in rollout_config.load_format,
        device_mesh=inference_device_mesh_cpu,
    )
    print(f"Sharding manager initialized on rank {rank}")

    # generate SGLang results
    log_gpu_memory_usage("Before entering sharding manager", logger=None)
    with rollout_sharding_manager:
        prompt_dict = TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=input_ids.shape[0],
        )

        messages = np.asarray([{"role": "user", "content": prompt} for prompt in preencode_prompts])
        prompts = DataProto(batch=prompt_dict, non_tensor_batch={"messages": messages})

        prompts.meta_info.update(
            {
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
            }
        )

        prompts = rollout_sharding_manager.preprocess_data(prompts)
        log_gpu_memory_usage("Before generating sequences", logger=None)
        output = rollout.generate_sequences(prompts=prompts)
        log_gpu_memory_usage("After generating sequences", logger=None)
        output = rollout_sharding_manager.postprocess_data(output)
        sglang_output = output.to("cpu")
    log_gpu_memory_usage("After exiting sharding manager", logger=None)

    # compare results
    sglang_response_tokens = tokenizer.batch_decode(sglang_output.batch["responses"])
    print(f"SGLang response: {sglang_response_tokens}")

    assert are_lists_similar(hf_response_tokens, sglang_response_tokens), "Responses differ more than 10%"

    print("Test passed!")


if __name__ == "__main__":
    test_sglang_rollout()
