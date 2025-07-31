# Copyright 2023-2024 SGLang Team
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

import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from verl.utils.model import compute_position_id_with_mask
from verl.utils.torch_functional import pad_sequence_to_length


# ====================== utils ======================
def levenshtein(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n]


def are_lists_similar(a, b, threshold=10):
    if len(a) != len(b):
        print("The lists are of different lengths.")
        return False
    total_length = 0
    total_diff = 0
    for s1, s2 in zip(a, b, strict=True):
        max_len = max(len(s1), len(s2))
        total_length += max_len
        total_diff += levenshtein(s1, s2)
    percentage_difference = (total_diff / total_length) * 100
    print(f"Total difference: {percentage_difference:.2f}%")
    return percentage_difference <= threshold


def initialize_global_process_group(timeout_second=36000, spmd=False):
    import torch.distributed

    if not torch.distributed.is_initialized():  # Check if already initialized
        print("Initializing process group...")
        torch.distributed.init_process_group(timeout=timedelta(seconds=timeout_second))
    else:
        print("Process group already initialized.")

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not CUDA_VISIBLE_DEVICES:
        if spmd:
            # CUDA_VISIBLE_DEVICES = ','.join(str(i) for i in range(tensor_parallel_size))
            CUDA_VISIBLE_DEVICES = ",".join(str(i) for i in range(world_size))
        else:
            CUDA_VISIBLE_DEVICES = str(local_rank)
        os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
        print(f"CUDA_VISIBLE_DEVICES is not set, set to {CUDA_VISIBLE_DEVICES}")

    return local_rank, rank, world_size


def clean_torchelastic_env():
    for k in ["TORCHELASTIC_USE_AGENT_STORE"]:
        if k in os.environ:
            del os.environ[k]


def load_tokenizer_and_model(local_model_path, dtype="bfloat16"):
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(local_model_path, torch_dtype=getattr(torch, dtype), device_map="cuda")
    return tokenizer, model


def prepare_inputs(tokenizer, prompts, max_prompt_length):
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    tokenized = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = pad_sequence_to_length(tokenized["input_ids"], max_prompt_length, pad_token_id, left_pad=True)
    attention_mask = pad_sequence_to_length(
        tokenized["attention_mask"], max_prompt_length, pad_token_id=0, left_pad=True
    )
    position_ids = compute_position_id_with_mask(attention_mask)
    position_ids = pad_sequence_to_length(position_ids, max_prompt_length, pad_token_id=0, left_pad=True)
    return input_ids, attention_mask, position_ids


def generate_hf_output(model, input_ids, attention_mask, tokenizer, max_response_length):
    generation_config = GenerationConfig(do_sample=False)
    output = model.generate(
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
    response = seq[:, input_ids.shape[1] :]
    return tokenizer.batch_decode(response)


def get_rollout_config(
    max_response_length,
    max_prompt_length,
    dtype,
    tensor_parallel_size,
    tool_config_path=None,
    interaction_config_path=None,
):
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
            "mode": "sync",
            "load_format": "dummy_dtensor",
            "enforce_eager": False,
            "free_cache_engine": True,
            "dtype": dtype,
            "gpu_memory_utilization": 0.5,
            "ignore_eos": False,
            "max_num_batched_tokens": 8192,
            "prompt_length": max_prompt_length,
            "response_length": max_response_length,
            "tensor_model_parallel_size": tensor_parallel_size,
            # set to 128MB only for testing
            "update_weights_bucket_megabytes": 128,
            "multi_turn": {
                "max_assistant_turns": 4,
                "max_user_turns": 4,
                "enable": True,
                "tool_config_path": tool_config_path,
                "interaction_config_path": interaction_config_path,
                "use_inference_chat_template": False,
                "tokenization_sanity_check_mode": "strict",
            },
            "calculate_log_probs": False,
            "max_model_len": None,
            **sampling_params,
        }
    )

    return rollout_config
