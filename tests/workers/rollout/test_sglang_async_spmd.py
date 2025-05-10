# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
usage: torchrun --standalone --nnodes=1 \
    --nproc_per_node=2 $(which pytest) \
    -s test_sglang_async_spmd.py
"""

import asyncio

import torch
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.utils import broadcast_pyobj
from torch.distributed.device_mesh import init_device_mesh
from utils_sglang import (
    are_lists_similar,
    clean_torchelastic_env,
    generate_hf_output,
    initialize_global_process_group,
    load_tokenizer_and_model,
    prepare_inputs,
)


def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor):
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


def test_sglang_spmd():
    assert torch.cuda.device_count() >= 2
    initialize_global_process_group(spmd=True)
    clean_torchelastic_env()

    max_prompt_length = 16
    max_response_length = 16

    local_model_path = "Qwen/Qwen2.5-0.5B"
    tokenizer, actor_model = load_tokenizer_and_model(local_model_path)

    preencode_prompts = ["Who won the Champions League in 2019?", "The founder of Apple is", "What's your name?"]
    input_ids, attention_mask, _ = prepare_inputs(tokenizer, preencode_prompts, max_prompt_length)

    hf_response_tokens = generate_hf_output(actor_model, input_ids, attention_mask, tokenizer, max_response_length)

    tensor_parallel_size = 2
    inference_device_mesh_cpu = init_device_mesh("cpu", mesh_shape=(1, tensor_parallel_size, 1), mesh_dim_names=["dp", "tp", "pp"])
    tp_rank = inference_device_mesh_cpu["tp"].get_local_rank()

    if tp_rank == 0:
        llm = Engine(
            model_path=local_model_path,
            dtype="bfloat16",
            mem_fraction_static=0.5,
            enable_memory_saver=True,
            tp_size=inference_device_mesh_cpu["tp"].size(),
        )

        input_ids = input_ids.cuda()
        idx_list = []

        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        for i in range(input_ids.shape[0]):
            idx_list.append(_pre_process_inputs(pad_token_id, input_ids[i]))

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

        loop = asyncio.get_event_loop()
        outputs = loop.run_until_complete(llm.async_generate(input_ids=idx_list, sampling_params=sampling_params))
    else:
        outputs = None

    [outputs] = broadcast_pyobj(
        [outputs],
        rank=inference_device_mesh_cpu["tp"].get_local_rank(),
        src=inference_device_mesh_cpu["tp"].mesh[0].item(),
        dist_group=inference_device_mesh_cpu["tp"].get_group(),
        force_cpu_device=False,
    )

    sglang_response_tokens = [output["text"] for output in outputs]

    print(f"sglang response: {sglang_response_tokens}")
    assert are_lists_similar(hf_response_tokens, sglang_response_tokens)
    print("SPMD Test Passed!")

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
