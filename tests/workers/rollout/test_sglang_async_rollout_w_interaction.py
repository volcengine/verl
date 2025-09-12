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
    -s test_sglang_async_rollout_w_interaction.py
"""

import numpy as np
import torch
from tensordict import TensorDict
from utils_sglang import (
    are_lists_similar,
    clean_torchelastic_env,
    generate_hf_output,
    get_rollout_config,
    initialize_global_process_group,
    load_tokenizer_and_model,
    prepare_inputs,
)

from verl import DataProto
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.sglang_rollout.sglang_rollout import SGLangRollout


def test_async_sglang_rollout_w_interaction():
    import os

    assert torch.cuda.device_count() >= 2
    initialize_global_process_group()
    clean_torchelastic_env()

    max_prompt_length = 32
    max_response_length = 16
    dtype = "bfloat16"
    tensor_parallel_size = 2
    local_model_path = os.path.expanduser("~/models/Qwen/Qwen2.5-0.5B")

    tokenizer, actor_model = load_tokenizer_and_model(local_model_path)

    preencode_prompts = [
        [{"role": "user", "content": prompt, "tool_calls": None}]
        for prompt in [
            "Who won the Champions League in 2019?",
            "The founder of Apple is",
            "What's the best way to learn python?",
        ]
    ]
    interaction_kwargs = [
        {"name": "gsm8k", "query": "Who won the Champions League in 2019?", "ground_truth": "Real Madrid"},
        {"name": "gsm8k", "query": "The founder of Apple is", "ground_truth": "Steve Jobs"},
        {"name": "gsm8k", "query": "What's the best way to learn python?", "ground_truth": "Learn python from scratch"},
    ]
    prompts = [
        tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        for message in preencode_prompts
    ]
    input_ids, attention_mask, position_ids = prepare_inputs(tokenizer, prompts, max_prompt_length)

    hf_response_tokens = generate_hf_output(actor_model, input_ids, attention_mask, tokenizer, max_response_length)

    # Create a temporary interaction config file for testing
    import tempfile

    from omegaconf import OmegaConf

    interaction_config = {
        "interaction": [
            {"name": "gsm8k", "class_name": "verl.interactions.gsm8k_interaction.Gsm8kInteraction", "config": {}}
        ]
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        OmegaConf.save(interaction_config, f.name)
        interaction_config_path = f.name

    rollout_config = get_rollout_config(
        max_response_length, max_prompt_length, dtype, tensor_parallel_size, None, interaction_config_path
    )
    rollout_config: RolloutConfig = omega_conf_to_dataclass(rollout_config, dataclass_type=RolloutConfig)
    model_config = HFModelConfig(path=local_model_path)
    rollout = SGLangRollout(
        config=rollout_config,
        model_config=model_config,
        device_mesh=None,
    )

    prompt_dict = TensorDict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        },
        batch_size=input_ids.shape[0],
    )
    print(f"preprocessed {input_ids.shape=}")

    messages = np.asarray(preencode_prompts)
    prompts = DataProto(
        batch=prompt_dict,
        non_tensor_batch={"raw_prompt": messages, "interaction_kwargs": np.asarray(interaction_kwargs)},
    )

    prompts.meta_info.update(
        {
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        }
    )

    # log_gpu_memory_usage("Before generating sequences", logger=None)
    output = rollout.generate_sequences(prompts=prompts)
    print(f"generated {output.batch['responses'].shape=}")
    # log_gpu_memory_usage("After generating sequences", logger=None)

    sglang_output = output.to("cpu")

    sglang_response_tokens = tokenizer.batch_decode(sglang_output.batch["responses"])

    print(f"hf response: {hf_response_tokens}")
    print(f"sglang response: {sglang_response_tokens}")
    assert are_lists_similar(hf_response_tokens, sglang_response_tokens)
    print("SGLang w interaction Test Passed!")

    # Clean up temporary config file
    import os

    os.unlink(interaction_config_path)

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    test_async_sglang_rollout_w_interaction()
