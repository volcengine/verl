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
import ray
from hydra import compose, initialize_config_dir
from transformers import AutoModelForSequenceClassification
from verl.utils.model import compute_position_id_with_mask
from verl.experimental.reward import RewardModelManager
from verl.protocol import DataProto

def create_data_samples(tokenizer) -> DataProto:
    convs = [
        [
            {
                "role": "user",
                "content": "What is the range of the numeric output of a sigmoid node in a neural network?",
            },
            {"role": "assistant", "content": "Between -1 and 1."},
        ],
        [
            {
                "role": "user",
                "content": "What is the range of the numeric output of a sigmoid node in a neural network?",
            },
            {"role": "assistant", "content": "Between 0 and 1."},
        ],
        [
            {"role": "user", "content": "What is the capital of Australia?"},
            {
                "role": "assistant",
                "content": "Canberra is the capital city of Australia.",
            },
        ],
        [
            {"role": "user", "content": "What is the capital of Australia?"},
            {
                "role": "assistant",
                "content": "Sydney is the capital of Australia.",
            },
        ],
    ]

    prompt_length, response_length = 1024, 4096
    pad_token_id = tokenizer.pad_token_id
    prompts, responses, input_ids, attention_masks = [], [], [], []
    for conv in convs:
        prompt_tokens = tokenizer.apply_chat_template(conv[:1], tokenize=True)
        response_tokens = tokenizer.apply_chat_template(conv, tokenize=True)[len(prompt_tokens) :]

        padded_prompt = [pad_token_id] * (prompt_length - len(prompt_tokens)) + prompt_tokens
        padded_response = response_tokens + [pad_token_id] * (response_length - len(response_tokens))
        attention_mask = (
            [0] * (prompt_length - len(prompt_tokens))
            + [1] * len(prompt_tokens)
            + [1] * len(response_tokens)
            + [0] * (response_length - len(response_tokens))
        )
        prompts.append(torch.tensor(padded_prompt))
        responses.append(torch.tensor(padded_response))
        input_ids.append(torch.tensor(padded_prompt + padded_response))
        attention_masks.append(torch.tensor(attention_mask))

    prompts = torch.stack(prompts)
    responses = torch.stack(responses)
    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)
    position_ids = compute_position_id_with_mask(attention_masks)

    return DataProto.from_dict(
        tensors={
            "prompts": prompts,
            "responses": responses,
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "position_ids": position_ids,
        },
    )



def test_reward_model_manager():
    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
                "VLLM_USE_V1": "1",
            }
        }
    )
    with initialize_config_dir(config_dir=os.path.abspath("recipe/fapo/config")):
        config = compose("rm_config")

    model_path = os.path.expanduser("~/models/Skywork/Skywork-Reward-V2-Llama-3.2-1B")

    config.reward_model.reward_manager = "dapo"
    config.reward_model.enable = True
    config.reward_model.enable_resource_pool = True
    config.reward_model.n_gpus_per_node = 8
    config.reward_model.nnodes = 1
    config.reward_model.model.path = model_path
    config.reward_model.rollout.name = os.getenv("ROLLOUT_NAME", "vllm")
    config.reward_model.rollout.gpu_memory_utilization = 0.9
    config.reward_model.rollout.tensor_model_parallel_size = 2
    config.reward_model.rollout.skip_tokenizer_init = False
    config.reward_model.rollout.prompt_length = 2048
    config.reward_model.rollout.response_length = 4096

    # 1. init reward model manager
    reward_model_manager = RewardModelManager(config.reward_model)
    tokenizer = reward_model_manager.tokenizer

    # 2. init test data
    data = create_data_samples(tokenizer)

    # 3. generate responses
    responses = reward_model_manager.compute_rm_score(data)

    for idx, (conv, response) in enumerate(zip(convs, responses, strict=False)):
        print(f"Problem {idx}:\n{conv['problem']}\n")
        print(f"AI Solution {idx}:\n{conv['solution']}\n")
        print(f"GRM Response {idx}:\n{response['grm_response']}\n")
        print("=" * 50 + "\n")

    ray.shutdown()
