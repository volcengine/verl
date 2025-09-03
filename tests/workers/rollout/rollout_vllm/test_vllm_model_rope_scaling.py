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

import gc
import os

import torch
import torch.distributed
import torch.distributed as dist
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from verl import DataProto
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.distributed import initialize_global_process_group
from verl.utils.model import compute_position_id_with_mask
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout


def test_vllm_rollout_with_yarn_position_embeddings():
    """
    Test the vLLM rollout with yarn position embeddings.
    """

    local_rank, rank, world_size = initialize_global_process_group()
    model_path = os.path.expanduser("~/models/OldKingMeister/Qwen2.5-1.5B-Instruct-YaRN")
    config = OmegaConf.create(
        {
            "name": "vllm",
            "prompt_length": 35000,
            "response_length": 512,
            "dtype": "bfloat16",
            "enforce_eager": True,
            "gpu_memory_utilization": 0.4,
            "enable_chunked_prefill": False,
            "free_cache_engine": False,
            "disable_log_stats": True,
            "max_model_len": 35000 + 512,
            "max_num_seqs": 1024,
            "load_format": "auto",
            "val_kwargs": {
                "top_k": -1,
                "top_p": 1.0,
                "temperature": 0,
                "n": 1,
                "do_sample": False,
            },
            "tensor_model_parallel_size": 4,
            "calculate_log_probs": False,
            "do_sample": False,
            "temperature": 0.0,
            "max_num_batched_tokens": 35000 + 512,
        }
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    # do_sample=False for temperate=0 deterministic
    input_dataproto = prepare_input_dataproto(tokenizer, config, validate=True, do_sample=False)

    rollout_config: RolloutConfig = omega_conf_to_dataclass(config, dataclass_type=RolloutConfig)
    model_config = HFModelConfig(path=model_path)
    model_config.tokenizer.pad_token = tokenizer.eos_token

    vllm_rollout = vLLMRollout(
        config=rollout_config,
        model_config=model_config,
        device_mesh=None,
    )
    # rollout
    rollout_response = vllm_rollout.generate_sequences(
        prompts=input_dataproto,
    )
    if rank == 0:
        print("VLLM Rollout Outputs:")
        print(tokenizer.batch_decode(rollout_response.batch["responses"][:], skip_special_tokens=False))
        for response in rollout_response.batch["responses"]:
            assert "<|im_end|>" in tokenizer.decode(response, skip_special_tokens=False), (
                "Response should contain <|im_end|> token"
            )
    print("Checks passed.")

    del vllm_rollout
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    dist.barrier()
    torch.distributed.destroy_process_group()


def prepare_input_dataproto(tokenizer, config, validate, do_sample=False):
    base_phrase = "Roses are red, sky is blue. " * 4096
    preencode_prompts = [
        # 32810 tokens > 32768 tokens
        [{"role": "user", "content": base_phrase + "Who won the Champions League in 2019?"}],
        [{"role": "user", "content": base_phrase + "The founder of Apple is"}],
        [{"role": "user", "content": base_phrase + "What's your name"}],
    ]
    formatted_prompts = [
        tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        for conversation in preencode_prompts
    ]
    prompts = tokenizer(formatted_prompts, return_tensors="pt", padding="max_length", max_length=config.prompt_length)
    input_dataproto = DataProto.from_dict(
        {
            "input_ids": prompts["input_ids"],
            "attention_mask": prompts["attention_mask"],
            "position_ids": compute_position_id_with_mask(prompts["attention_mask"]),
        },
        meta_info={
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "validate": validate,
            "do_sample": do_sample,
            "response_length": config.response_length,
            "temperature": config.temperature,
        },
    )
    return input_dataproto


if __name__ == "__main__":
    test_vllm_rollout_with_yarn_position_embeddings()
