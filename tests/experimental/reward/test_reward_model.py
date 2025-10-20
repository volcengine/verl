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

import ray
from hydra import compose, initialize_config_dir

from verl.experimental.reward import RewardModelManager
from verl.protocol import DataProto

GRM_PROMPT_TEMPLATE = """
You are given a problem and a proposed solution.

Problem:
{problem}

Solution:
{solution}

Please evaluate how well the solution addresses the problem. 
Give a score from 1 to 10, where:
- 1 means the solution is completely irrelevant or incorrect.
- 5 means the solution is partially correct but incomplete or not well reasoned.
- 10 means the solution is fully correct, well-reasoned, and directly solves the problem.

Only output the score as a single number (integer).
""".strip()


def create_data_samples() -> DataProto:
    convs = [
        {
            "problem": "What is the range of the numeric output of a sigmoid node in a neural network?",
            "solution": "Between -1 and 1.",
        },
        {
            "problem": "What is the range of the numeric output of a sigmoid node in a neural network?",
            "solution": "Between 0 and 1.",
        },
        {
            "problem": "What is the capital of Australia?",
            "solution": "Canberra is the capital city of Australia.",
        },
        {
            "problem": "What is the capital of Australia?",
            "solution": "Sydney is the capital city of Australia.",
        },
    ]

    messages = [[{"role": "user", "content": GRM_PROMPT_TEMPLATE.format(**conv)}] for conv in convs]
    prompts = DataProto.from_dict(
        non_tensors={
            "raw_prompt": messages,
        }
    )
    return convs, prompts


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

    model_path = os.path.expanduser("~/models/Qwen/Qwen2.5-0.5B-Instruct")

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

    # 2. init test data
    convs, prompts = create_data_samples()

    # 3. generate responses
    sampling_params = {
        "max_tokens": 4096,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
    }
    results = reward_model_manager.generate_sequences(prompts, sampling_params)
    responses = [result.choices[0].message.content for result in results]

    for idx, (conv, response) in enumerate(zip(convs, responses, strict=False)):
        print(f"Problem {idx}:\n{conv['problem']}\n")
        print(f"AI Solution {idx}:\n{conv['solution']}\n")
        print(f"GRM Response {idx}:\n{response}\n")
        print("=" * 50 + "\n")

    ray.shutdown()
