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
import transformers

from verl.third_party.vllm import LLM, vllm_version
from verl.utils.model import update_model_config
from vllm import SamplingParams
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

from transformers import GenerationConfig

from verl.utils.torch_functional import pad_sequence_to_length
from verl.workers.rollout.vllm_rollout.vllm_rollout import _pre_process_inputs


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
                dp[i - 1][j - 1] + cost  # Substitution
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


def test_vllm_with_hf():
    assert torch.cuda.device_count() >= 2, 'At least 2 GPUs is required to run tp+dp tests.'

    # fill rollout config
    max_prompt_length = 16
    max_response_length = 16

    # Initialize model and token
    local_cache_path = '~/.cache/verl/rlhf'
    local_cache_path = os.path.expanduser(local_cache_path)
    hdfs_path = 'deepseek-ai/deepseek-llm-7b-chat'
    from verl.utils.fs import copy_local_path_from_hdfs
    local_model_path = copy_local_path_from_hdfs(src=hdfs_path, cache_dir=local_cache_path)
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)

    preencode_prompts = [
        "Who won the Champions League in 2019?",
        "The founder of Apple is",
        "What's your name",
    ]
    tokenizer.pad_token = tokenizer.eos_token
    prompts = tokenizer(preencode_prompts, return_tensors='pt', padding=True)
    input_ids = prompts['input_ids']
    attention_mask = prompts['attention_mask']

    input_ids = pad_sequence_to_length(input_ids, max_prompt_length, tokenizer.pad_token_id, left_pad=True)
    attention_mask = pad_sequence_to_length(attention_mask, max_prompt_length, 0, left_pad=True)

    actor_model = AutoModelForCausalLM.from_pretrained(local_model_path)
    actor_model.to(torch.bfloat16)

    actor_model_config = AutoConfig.from_pretrained(local_model_path)

    temperature = 0
    top_p = 1

    kwargs = dict(n=1,
                  temperature=temperature,
                  top_p=top_p,
                  max_tokens=max_response_length,
                  logprobs=1,
                  ignore_eos=True)

    if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
        kwargs['detokenize'] = False
    sampling_params = SamplingParams(**kwargs)

    tensor_parallel_size = 2

    llm = LLM(model=actor_model,
              tokenizer=tokenizer,
              model_hf_config=actor_model_config,
              tensor_parallel_size=tensor_parallel_size,
              dtype='bfloat16',
              gpu_memory_utilization=0.1,
              load_format='hf')

    print('start generation')
    input_ids = input_ids.cuda()
    attention_mask = attention_mask.cuda()
    batch_size = input_ids.size(0)

    idx_list = []
    # parse idx from torch.Tensor to List[List[str]]
    for i in range(batch_size):
        idx_list.append(_pre_process_inputs(tokenizer.pad_token_id, input_ids[i]))
    outputs = llm.generate(prompt_token_ids=idx_list, sampling_params=sampling_params, use_tqdm=False)
    vllm_output = outputs[0].cuda()
    llm.free_cache_engine()
    llm = None
    import gc
    torch.cuda.empty_cache()
    gc.collect()

    generation_config = GenerationConfig(do_sample=False)
    actor_model.cuda()
    output = actor_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_response_length,
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

    hf_response_tokens = tokenizer.batch_decode(response)
    vllm_response_tokens = tokenizer.batch_decode(vllm_output)

    print(f'hf response: {hf_response_tokens}')
    print(f'vllm response: {vllm_response_tokens}')
    assert are_lists_similar(hf_response_tokens, vllm_response_tokens), \
        f'Strings differ more than 10%:\n'
    print('Check Pass')


# if __name__ == "__main__":
#     test_vllm_with_hf()
