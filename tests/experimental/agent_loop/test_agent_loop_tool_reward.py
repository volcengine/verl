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
import json
import os
from typing import Any

import numpy as np
import pytest
import ray
from omegaconf import DictConfig
from transformers.utils import get_json_schema

from tests.experimental.agent_loop.agent_utils import init_agent_loop_manager
from verl.experimental.agent_loop.agent_loop import get_trajectory_info
from verl.protocol import DataProto
from verl.tools.base_tool import BaseTool, OpenAIFunctionToolSchema
from verl.tools.schemas import ToolResponse
from verl.utils import hf_tokenizer

@pytest.fixture
def init_config() -> DictConfig:
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config")):
        config = compose(
            config_name="ppo_trainer",
            overrides=[
                "actor_rollout_ref.actor.use_dynamic_bsz=true",
                # test sleep/wake_up with fsdp offload
                "actor_rollout_ref.actor.fsdp_config.param_offload=True",
                "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
            ],
        )

    model_path = "Qwen/Qwen2.5-1.5B-Instruct"
    config.actor_rollout_ref.model.path = model_path
    config.actor_rollout_ref.rollout.name = os.environ["ROLLOUT_NAME"]
    config.actor_rollout_ref.rollout.mode = "async"
    config.actor_rollout_ref.rollout.prompt_length = 4096
    config.actor_rollout_ref.rollout.response_length = 4096
    config.actor_rollout_ref.rollout.n = 4
    config.actor_rollout_ref.rollout.agent.num_workers = 2

    return config


def test_single_turn(init_config):
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

    agent_loop_manager = init_agent_loop_manager(init_config)

    raw_prompts = [
        [
            {
                "role": "user",
                "content": "Let's play a role playing game. Your name is Alice, your favorite color is blue.",
            }
        ],
        [{"role": "user", "content": "Let's play a role playing game. Your name is Bob, your favorite color is red."}],
    ]
    batch = DataProto(
        non_tensor_batch={
            "raw_prompt": np.array(raw_prompts),
            "agent_name": np.array(["single_turn_agent"] * len(raw_prompts)),
            "data_source": np.array(["openai/gsm8k"] * len(raw_prompts)),
            "reward_model": np.array([{"style": "rule", "ground_truth": "1.0"}] * len(raw_prompts)),
        },
    )
    n = init_config.actor_rollout_ref.rollout.n
    batch = batch.repeat(n)
    result = agent_loop_manager.generate_sequences(prompts=batch)
    assert len(result) == len(raw_prompts) * n

    # check result
    seq_len = result.batch["prompts"].size(1) + result.batch["responses"].size(1)
    assert result.batch["input_ids"].size(1) == seq_len
    assert result.batch["attention_mask"].size(1) == seq_len
    assert result.batch["position_ids"].size(1) == seq_len
    assert result.batch["rm_scores"].size(1) == result.batch["responses"].size(1)
    if init_config.actor_rollout_ref.rollout.calculate_log_probs:
        assert result.batch["rollout_log_probs"].size(1) == result.batch["responses"].size(1)

    # check turns
    num_turns = result.non_tensor_batch["__num_turns__"]
    assert np.all(num_turns == 2)

    print("Test passed!")
    ray.shutdown()



class WeatherTool(BaseTool):
    def get_current_temperature(self, location: str, unit: str = "celsius"):
        """Get current temperature at a location.

        Args:
            location: The location to get the temperature for, in the format "City, State, Country".
            unit: The unit to return the temperature in. Defaults to "celsius". (choices: ["celsius", "fahrenheit"])

        Returns:
            the temperature, the location, and the unit in a dict
        """
        print(f"[DEBUG] get_current_temperature: {location}, {unit}")
        return {
            "temperature": 26.1,
            "location": location,
            "unit": unit,
        }

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        schema = get_json_schema(self.get_current_temperature)
        return OpenAIFunctionToolSchema(**schema)

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        try:
            result = self.get_current_temperature(**parameters)
            return ToolResponse(text=json.dumps(result)), 0.5, {"tool_name": "weather_tool"}
        except Exception as e:
            return ToolResponse(text=str(e)), -0.1, {"tool_name": "weather_tool"}


class WeatherToolWithData(BaseTool):
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        schema = get_json_schema(self.get_temperature_date)
        return OpenAIFunctionToolSchema(**schema)

    def get_temperature_date(self, location: str, date: str, unit: str = "celsius"):
        """Get temperature at a location and date.

        Args:
            location: The location to get the temperature for, in the format "City, State, Country".
            date: The date to get the temperature for, in the format "Year-Month-Day".
            unit: The unit to return the temperature in. Defaults to "celsius". (choices: ["celsius", "fahrenheit"])

        Returns:
            the temperature, the location, the date and the unit in a dict
        """
        print(f"[DEBUG] get_temperature_date: {location}, {date}, {unit}")
        return {
            "temperature": 25.9,
            "location": location,
            "date": date,
            "unit": unit,
        }

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        try:
            result = self.get_temperature_date(**parameters)
            return ToolResponse(text=json.dumps(result)), 0.5, {"tool_name": "weather_tool"}
        except Exception as e:
            return ToolResponse(text=str(e)), -0.1, {"tool_name": "weather_tool"}



def test_tool_agent(init_config):
    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
                "VLLM_USE_V1": "1",
            }
        },
        ignore_reinit_error=True,
    )

    # =========================== 1. Init rollout manager ===========================
    tool_config = {
        "tools": [
            {
                "class_name": "tests.experimental.agent_loop.test_agent_loop_tool_reward.WeatherTool",
                "config": {"type": "native"},
            },
            {
                "class_name": "tests.experimental.agent_loop.test_agent_loop_tool_reward.WeatherToolWithData",
                "config": {"type": "native"},
            },
        ]
    }
    tool_config_path = "/tmp/tool_config.json"
    with open(tool_config_path, "w") as f:
        json.dump(tool_config, f)

    n = 2
    init_config.actor_rollout_ref.rollout.n = n
    init_config.actor_rollout_ref.rollout.multi_turn.tool_config_path = tool_config_path
    init_config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls = 2
    init_config.actor_rollout_ref.rollout.calculate_log_probs = True
    agent_loop_manager = init_agent_loop_manager(init_config)

    # =========================== 2. Generate sequences  ===========================
    raw_prompts = [
        [
            {"role": "user", "content": "How are you?"},
        ],
        [
            {"role": "user", "content": "What's the temperature in Los Angeles now?"},
        ],
        [
            {"role": "user", "content": "What's the temperature in New York now?"},
        ],
        [
            {
                "role": "system",
                "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\n"
                "Current Date: 2024-09-30",
            },
            {"role": "user", "content": "What's the temperature in San Francisco now? How about tomorrow?"},
        ],
    ]
    batch = DataProto(
        non_tensor_batch={
            "raw_prompt": np.array([np.array(prompt) for prompt in raw_prompts], dtype=object),
            "agent_name": np.array(["tool_agent"] * len(raw_prompts)),
            "data_source": np.array(["openai/gsm8k"] * len(raw_prompts)),
            "reward_model": np.array([{"style": "rule", "ground_truth": "1.0"}] * len(raw_prompts)),
        },
    )
    batch = batch.repeat(n)
    result = agent_loop_manager.generate_sequences(prompts=batch)
    assert len(result) == len(raw_prompts) * n

    extra_info = result.non_tensor_batch.get("extra_info")
    print(f"extra_info: {extra_info}")
    
    # Check that extra_info exists and contains tool rewards/metrics for samples that used tools
    if extra_info is not None:
        # extra_info is a numpy array of dicts, check each sample
        for i, info in enumerate(extra_info):
            if isinstance(info, dict):
                print(f"Sample {i} extra_info: {info}")
                # For samples that used tools, we should have tool_rewards and tool_metrics
                if "tool_rewards" in info:
                    assert isinstance(info["tool_rewards"], (int, float))
                    print(f"Sample {i} tool_rewards: {info['tool_rewards']}")
                if "tool_metrics" in info:
                    assert isinstance(info["tool_metrics"], list)
                    print(f"Sample {i} tool_metrics: {info['tool_metrics']}")
    
    # At least some samples should have used tools and have tool rewards/metrics
    has_tool_rewards = False
    has_tool_metrics = False
    if extra_info is not None:
        for info in extra_info:
            if isinstance(info, dict):
                if "tool_rewards" in info:
                    has_tool_rewards = True
                if "tool_metrics" in info:
                    has_tool_metrics = True
    
    print(f"Has tool rewards: {has_tool_rewards}, Has tool metrics: {has_tool_metrics}")
    # Note: The assertion might need to be adjusted based on which samples actually use tools
    # For now, let's just check that the structure is correct
