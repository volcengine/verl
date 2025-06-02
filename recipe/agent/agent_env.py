# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import uuid
import json
import requests
from transformers import PreTrainedTokenizer
from typing import List, Union, Optional, Literal
import verl.utils.torch_functional as verl_F
from .utils import convert_assistant_message_to_openai, get_task_prompt, convert_chat_message_from_openai, get_tools_schema, get_agent_system_prompt, get_allow_parallel_tool_call
from verl.utils.model import compute_position_id_with_mask

def initialize_env(environment_endpoint: str, env_name: str, seed: int, env_kwargs: Union[dict, str]):
    """Initialize a remote environment instance and get initial state.

    Args:
        environment_endpoint (str): Base URL of the environment server
        env_name (str): Name of environment to initialize
        seed (int): Random seed for environment
        env_kwargs (dict): Additional keyword arguments for environment initialization

    Returns:
        dict: Environment initialization info containing:
            - env_id (str): Unique ID for this environment instance
            - observation (list): Initial observation as chat message list

    Raises:
        AssertionError: If required fields are missing from server responses
    """
    if isinstance(env_kwargs, str):
        env_kwargs = json.loads(env_kwargs)

    payload = {"env_name": env_name, "seed": seed, "env_kwargs": env_kwargs}
    init_response = requests.post(f"{environment_endpoint}/api/environment/initialize", json=payload)
    init_obj = init_response.json()
    assert 'env_id' in init_obj, f"Environment initialization failed: {init_obj} as env_id is not present"
    env_id = init_obj['env_id']
    assert 'observation' in init_obj, f"Environment initialization failed: {init_obj} as observation is not present"
    assert 'info' in init_obj, f"Environment initialization failed: {init_obj} as info is not present"  # info is only for the gymnasium convention

    return {
        'env_id': env_id,  # the environment id
        'observation': init_obj['observation'],  # a chat message list as the initial observation
    }


def reset_env(environment_endpoint: str, env_id: str, seed: int, options: Optional[Union[dict, str]] = None):
    """Reset the environment and get the initial observation."""
    if isinstance(options, str):
        options = json.loads(options)
    payload = {"env_id": env_id, "seed": seed, "options": options}
    reset_response = requests.post(f"{environment_endpoint}/api/environment/{env_id}/reset", json=payload)
    reset_obj = reset_response.json()
    assert 'observation' in reset_obj, f"Environment reset failed: {reset_obj} as observation is not present"
    assert 'info' in reset_obj, f"Environment reset failed: {reset_obj} as info is not present"  # info is only for the gymnasium convention
    return {
        'observation': reset_obj['observation'],  # a chat message list as the initial observation
        'info':
            reset_obj['info']  # info is only for the gymnasium convention
    }


def close_env(environment_endpoint: str, env_id: str):
    """Close the environment."""
    payload = {"env_id": env_id}
    close_response = requests.post(f"{environment_endpoint}/api/environment/{env_id}/close", json=payload)
    close_obj = close_response.json()


def step_env(environment_endpoint: str, env_id: str, action: dict):
    """
    Step the environment with the action.
    
    The step response is a dictionary with the following keys:
    - observation: list of chat messages
    - reward: float
    - done: bool
    - truncated: bool
    - info: dict
    """
    payload = {"action": action}
    step_response = requests.post(f"{environment_endpoint}/api/environment/{env_id}/step", json=payload)
    step_obj = step_response.json()
    return step_obj

# Environment class
class AgentEnv:

    def __init__(self, environment_endpoint: str, env_name: str, seed: int, env_kwargs: Union[dict, str],
                 agent_prompt_style: Literal['qwen2_5'], tokenizer: PreTrainedTokenizer, max_prompt_length: int,
                 truncation: Literal['error', 'ignore', 'max_length']):
        self.environment_endpoint = environment_endpoint
        self.env_name = env_name
        self.seed = seed
        if isinstance(env_kwargs, str):
            self.env_kwargs = json.loads(env_kwargs)
        else:
            self.env_kwargs = env_kwargs
        self.agent_prompt_style = agent_prompt_style
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation

        self.chat = []
        self.env_id = None
        self.action_turn = []
        self.reward_by_action_turn = []

    def initialize(self):
        env_meta_data = initialize_env(self.environment_endpoint, self.env_name, self.seed, self.env_kwargs)
        if self.env_id is not None:
            close_env(self.environment_endpoint, self.env_id)
        self.env_id = env_meta_data['env_id']
        env_meta_data = reset_env(self.environment_endpoint, self.env_id, self.seed, self.env_kwargs)

        # get the initial observation
        observation = env_meta_data['observation']
        # get the task prompt
        task_prompt = get_task_prompt(self.environment_endpoint, self.env_id)
        # get the allow parallel tool call
        allow_parallel_tool_call = get_allow_parallel_tool_call(self.environment_endpoint, self.env_id)
        # get the tools schema
        tools_schema = get_tools_schema(self.environment_endpoint, self.env_id)
        system_prompt = get_agent_system_prompt(self.agent_prompt_style, task_prompt, tools_schema,
                                                allow_parallel_tool_call)
        self.chat = [{"role": "system", "content": system_prompt}, *observation]

    def tokenize_chat(self, add_generation_prompt: bool = False):
        return_dict = {}
        prompt = convert_chat_message_from_openai(self.agent_prompt_style, self.chat)
        prompt_with_chat_template = self.tokenizer.apply_chat_template(prompt,
                                                                       add_generation_prompt=add_generation_prompt,
                                                                       tokenize=False)
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.max_prompt_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.truncation)

        position_ids = compute_position_id_with_mask(attention_mask)

        return_dict['input_ids'] = input_ids[0]
        return_dict['attention_mask'] = attention_mask[0]
        return_dict['position_ids'] = position_ids[0]
        return_dict['raw_prompt_ids'] = self.tokenizer.encode(prompt_with_chat_template, add_special_tokens=False)

        if not add_generation_prompt:
            # Only compute the response mask and tokenwise reward for the non-generation prompt tokens
            # As they are usually needed for things after all the rollouts are done
            # Also the logic in get_response_mask_and_tokenwise_reward is based on the assumption that NO generation prompt is added
            # If you also want to compute the response mask and tokenwise reward for the generation prompt tokens,
            # simply removing the if statement does not work.
            model_generated_mask, tokenwise_reward = get_model_generated_mask_and_tokenwise_reward(
                self.tokenizer, self.agent_prompt_style, self.chat, input_ids[0], attention_mask[0], self.action_turn,
                self.reward_by_action_turn)
            return_dict['model_generated_mask'] = model_generated_mask
            # Not that this is different from the response_mask implemented elsewhere
            # response_mask is only for the raw_prompt_ids, usually refers to the last assistant message?
            # while model_generated_mask is for all the tokens + padding tokens, basically having the same length as input_ids
            return_dict['tokenwise_reward'] = tokenwise_reward
        return return_dict

    def step(self, assistant_message_str: str, tool_parsing_error_reward: float):
        """
        Step the environment with the assistant message.
        """
        try:
            assistant_message = convert_assistant_message_to_openai(self.agent_prompt_style, assistant_message_str)
            step_obj = step_env(self.environment_endpoint, self.env_id, assistant_message)
            observation = step_obj['observation']
            reward = step_obj['reward']
            done = step_obj['done']
            truncated = step_obj['truncated']
            info = step_obj['info']
        except Exception as e:
            # Remove the leading newline characters
            # This is to make sure the turn parsing is correct, this is however may cause misalignment with the original assistant message
            assistant_message_str = assistant_message_str.lstrip()
            assistant_message = {"role": "assistant", "content": assistant_message_str}
            reward = tool_parsing_error_reward
            done = True
            truncated = False
            info = {}
            error_message = f"Failed to parse the assistant message: {assistant_message_str}, error: {e}"
            observation = [{"role": "user", "content": error_message}]
        self.chat.append(assistant_message)
        self.action_turn.append(len(self.chat) - 1)
        self.reward_by_action_turn.append(reward)
        self.chat += observation
        return observation, reward, done, truncated, info

    def __del__(self):
        if self.env_id is not None:
            close_env(self.environment_endpoint, self.env_id)
