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


def test_vllm_with_hf():
    assert torch.cuda.device_count() >= 2, 'At least 2 GPUs is required to run tp+dp tests.'

    # fill rollout config
    max_prompt_length = 16
    max_response_length = 32

    # Initialize model and token
    local_cache_path = '~/.cache/verl/rlhf'
    local_cache_path = os.path.expanduser(local_cache_path)
    hdfs_path = 'deepseek-ai/deepseek-llm-7b-chat'
    from verl.utils.fs import copy_local_path_from_hdfs
    local_model_path = copy_local_path_from_hdfs(src=hdfs_path, cache_dir=local_cache_path)
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)

    preencode_prompts = [
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
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

    print(f'hf response: {tokenizer.batch_decode(response)}')
    print(f'vllm response: {tokenizer.batch_decode(vllm_output)}')
    assert torch.allclose(response, vllm_output), f'hf_response:{response} | vllm_response:{vllm_output}'
    assert False
    print('Check Pass')


# if __name__ == "__main__":
#     test_vllm_with_hf()
