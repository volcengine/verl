''' usage: torchrun --standalone --nnodes=1 --nproc_per_node=2 $(which pytest) -s test_async_sglang.py'''

import asyncio
import os
from datetime import timedelta

import numpy as np
import torch
from omegaconf import OmegaConf
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.utils import broadcast_pyobj
from tensordict import TensorDict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from verl.protocol import DataProto
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask
from verl.utils.torch_functional import pad_sequence_to_length
from verl.workers.rollout.sglang_rollout.async_sglang_rollout import AsyncSGLangRollout
from verl.workers.sharding_manager.fsdp_async_sglang import FSDPAsyncSGLangShardingManager

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


def are_lists_similar(a, b):
    if len(a) != len(b):
        print("The lists are of different lengths.")
        return False
    total_length = 0
    total_diff = 0
    for s1, s2 in zip(a, b):
        max_len = max(len(s1), len(s2))
        total_length += max_len
        total_diff += levenshtein(s1, s2)
    percentage_difference = (total_diff / total_length) * 100
    print(f"Total difference: {percentage_difference:.2f}%")
    return percentage_difference <= 10


def initialize_global_process_group(timeout_second=36000):
    import torch.distributed

    torch.distributed.init_process_group(timeout=timedelta(seconds=timeout_second))
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
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


def get_rollout_config(max_response_length, max_prompt_length, dtype, tensor_parallel_size):
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

    return rollout_config


# ====================== test_sglang_spmd ======================
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor):
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


def test_sglang_spmd():
    assert torch.cuda.device_count() >= 2
    initialize_global_process_group()
    clean_torchelastic_env()

    max_prompt_length = 16
    max_response_length = 16

    local_model_path = copy_to_local("Qwen/Qwen2-7B-Instruct", cache_dir=os.path.expanduser("~/.cache/verl/rlhf"))
    tokenizer, actor_model = load_tokenizer_and_model(local_model_path)

    preencode_prompts = ["Who won the Champions League in 2019?", "The founder of Apple is", "What's your name?"]
    input_ids, attention_mask, _ = prepare_inputs(tokenizer, preencode_prompts, max_prompt_length)

    hf_response_tokens = generate_hf_output(actor_model, input_ids, attention_mask, tokenizer, max_response_length)

    tensor_parallel_size = 2
    inference_device_mesh_cpu = init_device_mesh(
        "cpu", mesh_shape=(1, tensor_parallel_size, 1), mesh_dim_names=["dp", "tp", "pp"]
    )
    tp_rank = inference_device_mesh_cpu["tp"].get_local_rank()

    if tp_rank == 0:
        llm = Engine(
            model_path=local_model_path,
            dtype="bfloat16",
            mem_fraction_static=0.5,
            enable_memory_saver=True,
            tp_size=inference_device_mesh_cpu["tp"].size(),
        )
    else:
        llm = None

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

    outputs = (
        asyncio.run(llm.async_generate(input_ids=idx_list, sampling_params=sampling_params)) if tp_rank == 0 else None
    )
    [outputs] = broadcast_pyobj([outputs], rank=tp_rank, dist_group=inference_device_mesh_cpu["tp"].get_group(), src=0)
    sglang_response_tokens = [output["text"] for output in outputs]

    for output in outputs:
        print(f"{output=}")
        generated_text = output["text"]
        sglang_response_tokens.append(generated_text)

    print(f"sglang response: {sglang_response_tokens}")
    assert are_lists_similar(hf_response_tokens, sglang_response_tokens)
    print("SPMD Test Passed!")


# ====================== test_async_sglang_rollout ======================


def test_async_sglang_rollout():
    assert torch.cuda.device_count() >= 2
    initialize_global_process_group()
    clean_torchelastic_env()

    max_prompt_length = 16
    max_response_length = 16
    dtype = "bfloat16"
    tensor_parallel_size = 2
    local_model_path = "Qwen/Qwen2.5-0.5B"

    tokenizer, actor_model = load_tokenizer_and_model(local_model_path)

    preencode_prompts = ["Who won the Champions League in 2019?", "The founder of Apple is", "What's your name?"]
    input_ids, attention_mask, position_ids = prepare_inputs(tokenizer, preencode_prompts, max_prompt_length)

    hf_response_tokens = generate_hf_output(actor_model, input_ids, attention_mask, tokenizer, max_response_length)

    fsdp_device_mesh = init_device_mesh("cuda", (tensor_parallel_size,), ("fsdp",))
    inference_device_mesh_cpu = init_device_mesh("cpu", (2, tensor_parallel_size, 1), ("dp", "infer_tp", "pp"))

    fsdp_model = FSDP(
        actor_model,
        use_orig_params=True,
        device_id=fsdp_device_mesh["fsdp"].get_local_rank(),
        mixed_precision=MixedPrecision(param_dtype=getattr(torch, dtype)),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_mesh=fsdp_device_mesh,
    )

    rollout_config = get_rollout_config(max_response_length, max_prompt_length, dtype, tensor_parallel_size)
    rollout = AsyncSGLangRollout(
        actor_module=local_model_path, config=rollout_config, tokenizer=tokenizer, model_hf_config=actor_model.config
    )

    rollout_sharding_manager = FSDPAsyncSGLangShardingManager(
        module=fsdp_model,
        inference_engine=rollout._engine,
        model_config=actor_model.config,
        full_params=True,
        device_mesh=inference_device_mesh_cpu,
    )

    with rollout_sharding_manager:
        prompt_dict = TensorDict(
            {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids},
            batch_size=input_ids.shape[0],
        )
        prompts = DataProto(
            batch=prompt_dict,
            non_tensor_batch={
                "messages": np.asarray([{"role": "user", "content": prompt} for prompt in preencode_prompts])
            },
        )
        output = rollout.generate_sequences(prompts=prompts)
        sglang_output = output.to("cpu")

    sglang_response_tokens = tokenizer.batch_decode(sglang_output.batch["responses"])

    print(f"hf response: {hf_response_tokens}")
    print(f"sglang response: {sglang_response_tokens}")
    assert are_lists_similar(hf_response_tokens, sglang_response_tokens)
    print("Rollout Test Passed!")


# ====================== test_async_sglang_rollout_w_tool ======================


def test_async_sglang_rollout_w_tool():
    assert torch.cuda.device_count() >= 2
    initialize_global_process_group()
    clean_torchelastic_env()

    max_prompt_length = 32
    max_response_length = 16
    dtype = "bfloat16"
    tensor_parallel_size = 2
    local_model_path = "Qwen/Qwen2.5-0.5B"

    tokenizer, actor_model = load_tokenizer_and_model(local_model_path)

    preencode_prompts = [
        [{"role": "user", "content": prompt}]
        for prompt in [
            "Who won the Champions League in 2019?",
            "The founder of Apple is",
            "What's the best way to learn python?",
        ]
    ]
    prompts = [
        tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        for message in preencode_prompts
    ]
    input_ids, attention_mask, position_ids = prepare_inputs(tokenizer, prompts, max_prompt_length)

    hf_response_tokens = generate_hf_output(actor_model, input_ids, attention_mask, tokenizer, max_response_length)

    fsdp_device_mesh = init_device_mesh("cuda", (tensor_parallel_size,), ("fsdp",))
    inference_device_mesh_cpu = init_device_mesh("cpu", (2, tensor_parallel_size, 1), ("dp", "infer_tp", "pp"))

    fsdp_model = FSDP(
        actor_model,
        use_orig_params=True,
        device_id=fsdp_device_mesh["fsdp"].get_local_rank(),
        mixed_precision=MixedPrecision(param_dtype=getattr(torch, dtype)),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_mesh=fsdp_device_mesh,
    )

    rollout_config = get_rollout_config(max_response_length, max_prompt_length, dtype, tensor_parallel_size)
    rollout = AsyncSGLangRollout(
        actor_module=local_model_path, config=rollout_config, tokenizer=tokenizer, model_hf_config=actor_model.config
    )

    rollout_sharding_manager = FSDPAsyncSGLangShardingManager(
        module=fsdp_model,
        inference_engine=rollout._engine,
        model_config=actor_model.config,
        full_params=True,
        device_mesh=inference_device_mesh_cpu,
    )

    with rollout_sharding_manager:
        prompt_dict = TensorDict(
            {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids},
            batch_size=input_ids.shape[0],
        )
        prompts = DataProto(batch=prompt_dict, non_tensor_batch={"raw_prompt": np.asarray(preencode_prompts)})
        output = rollout.generate_sequences_with_tools(prompts=prompts)
        sglang_output = output.to("cpu")

    sglang_response_tokens = tokenizer.batch_decode(sglang_output.batch["responses"])

    print(f"hf response: {hf_response_tokens}")
    print(f"sglang response: {sglang_response_tokens}")
    assert are_lists_similar(hf_response_tokens, sglang_response_tokens)
    print("w tool Test Passed!")


if __name__ == "__main__":
    test_sglang_spmd()
    test_async_sglang_rollout()
    test_async_sglang_rollout_w_tool()
