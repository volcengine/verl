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

import uuid
from omegaconf import ListConfig
import os
from typing import List, Union, Optional, Literal
import copy
import requests
import json
import pandas as pd
import textwrap
from collections import defaultdict

import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F

## Utils for interacting with the environment

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

    payload = {
        "env_name": env_name,
        "seed": seed,
        "env_kwargs": env_kwargs
    }
    init_response = requests.post(f"{environment_endpoint}/api/environment/initialize", json=payload)
    init_obj = init_response.json()
    assert 'env_id' in init_obj, f"Environment initialization failed: {init_obj} as env_id is not present"
    env_id = init_obj['env_id']
    assert 'observation' in init_obj, f"Environment initialization failed: {init_obj} as observation is not present"
    assert 'info' in init_obj, f"Environment initialization failed: {init_obj} as info is not present" # info is only for the gymnasium convention

    return {
        'env_id': env_id, # the environment id
        'observation': init_obj['observation'], # a chat message list as the initial observation
    }

def reset_env(environment_endpoint: str, env_id: str, seed: int, options: Optional[Union[dict, str]] = None):
    """Reset the environment and get the initial observation."""
    if isinstance(options, str):
        options = json.loads(options)
    payload = {
        "env_id": env_id,
        "seed": seed,
        "options": options
    }
    reset_response = requests.post(f"{environment_endpoint}/api/environment/{env_id}/reset", json=payload)
    reset_obj = reset_response.json()
    assert 'observation' in reset_obj, f"Environment reset failed: {reset_obj} as observation is not present"
    assert 'info' in reset_obj, f"Environment reset failed: {reset_obj} as info is not present" # info is only for the gymnasium convention
    return {
        'observation': reset_obj['observation'], # a chat message list as the initial observation
        'info': reset_obj['info'] # info is only for the gymnasium convention
    }

def close_env(environment_endpoint: str, env_id: str):
    """Close the environment."""
    payload = {
        "env_id": env_id
    }
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
    payload = {
        "action": action
    }
    step_response = requests.post(f"{environment_endpoint}/api/environment/{env_id}/step", json=payload)
    step_obj = step_response.json()
    return step_obj

def get_task_prompt(environment_endpoint: str, env_id: str) -> str:
    """Retrieve the task prompt for a given environment ID."""
    prompt_response = requests.get(f"{environment_endpoint}/api/environment/{env_id}/task-prompt")
    prompt_obj = prompt_response.json()
    assert 'task_prompt' in prompt_obj, f"Failed to get task prompt: {prompt_obj} as task_prompt is not present"
    return prompt_obj['task_prompt']

def get_allow_parallel_tool_call(environment_endpoint: str, env_id: str) -> bool:
    """Retrieve the allow_parallel_tool_call for a given environment ID."""
    prompt_response = requests.get(f"{environment_endpoint}/api/environment/{env_id}/allow-parallel-tool-call")
    prompt_obj = prompt_response.json()
    if 'allow_parallel_tool_call' in prompt_obj:
        return prompt_obj['allow_parallel_tool_call']
    else:
        # By default, it does not support parallel tool call.
        return False

def get_tools_schema(environment_endpoint: str, env_id: str) -> dict:
    """
    Retrieve the tools schema for a given environment ID.
    The tools schema follows the openai format.
    """
    schema_response = requests.get(f"{environment_endpoint}/api/environment/{env_id}/tools-schema-openai")
    schema_obj = schema_response.json()
    assert 'tools_schema' in schema_obj, f"Failed to get tools schema: {schema_obj} as tools_schema is not present"
    return schema_obj['tools_schema']

## Utils for llm specific chat template

def get_agent_system_prompt(
    agent_prompt_style: Literal['qwen2_5'],
    task_prompt: str,
    tools_schema: dict,
    allow_parallel_tool_call: bool,
    ) -> str:
    """
    Get the system prompt for the agent.
    """
    if agent_prompt_style == 'qwen2_5':
        # See https://qwen.readthedocs.io/en/latest/framework/function_call.html#qwen2-5-function-calling-templates
        if True:
            # TODO: The single function call prompt is not working on qwen 2.5 7b instruction model. Will fix it in the future.
            system_prompt = task_prompt.rstrip() + "\n" + textwrap.dedent("""
            # Tools

            You may call one or more functions to assist with the user query.

            You are provided with function signatures within <tools></tools> XML tags:
            <tools>
            """).rstrip()
        else:
            system_prompt = task_prompt.rstrip() + "\n" + textwrap.dedent("""
            # Tools

            You may call one functions to assist with the user query.

            You are provided with function signatures within <tools></tools> XML tags:
            <tools>
            """).rstrip()
        for tool in tools_schema:
            system_prompt += '\n' + json.dumps(tool)
        system_prompt += textwrap.dedent("""
        </tools>

        For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
        <tool_call>
        {"name": <function-name>, "arguments": <args-json-object>}
        </tool_call>
        """)
    else:
        raise ValueError(f"Agent prompt style {agent_prompt_style} is not supported")

    return system_prompt

def convert_chat_message_from_openai(
    agent_prompt_style: Literal['qwen2_5'],
    chat: list[dict],
) -> list[dict]:
    """
    Convert the chat message from openai format to the agent prompt style format.
    """
    new_chat = []
    if agent_prompt_style == 'qwen2_5':
        for message in chat:
            if message['role'] == 'system':
                assert isinstance(message['content'], str), f"System message must be a string, got {type(message['content'])}, which is not supported"
                new_chat.append(message)
            elif message['role'] == 'user':
                if isinstance(message['content'], str):
                    new_chat.append(message)
                else:
                    raise ValueError(f"User message must be a string, got {type(message['content'])}, which is not supported")
            elif message['role'] == 'tool':
                assert isinstance(message['content'], str), f"Tool message must be a string, got {type(message['content'])}, which is not supported"
                new_chat.append({
                    "role": "user",
                    "content": "<tool_response>\n" + message['content'] + "\n</tool_response>"
                })
            elif message['role'] == 'assistant':
                assert message['content'] is None or isinstance(message['content'], str), f"Assistant message must be a string, got {type(message['content'])}, which is not supported"
                if "tool_calls" not in message:
                    new_chat.append(message)
                else:
                    content = ""
                    if message['content'] is not None and message['content'].strip() != "":
                        content += message['content'] + "\n"
                        
                    for tool_call in message['tool_calls']:
                        content += "<tool_call>\n" + json.dumps({
                            "name": tool_call['function']['name'],
                            "arguments": json.loads(tool_call['function']['arguments'])
                        }) + "\n</tool_call>\n"
                    content = content.rstrip()
                    new_chat.append({
                        "role": "assistant",
                        "content": content
                    })
            else:
                raise ValueError(f"Message role {message['role']} is not supported")
    else:
        raise ValueError(f"Agent prompt style {agent_prompt_style} is not supported")
    return new_chat

def convert_assistant_message_to_openai(
    agent_prompt_style: Literal['qwen2_5'],
    assistant_message_str: str,
) -> list[dict]:
    """
    Convert the one single chat message from the agent prompt style to the openai format.
    """
    assistant_message = {
        "role": "assistant",
    }
    if agent_prompt_style == 'qwen2_5':
        content = ""
        if not assistant_message_str.startswith('<tool_call>'):
            content = assistant_message_str.split('<tool_call>',1)[0]
        assistant_message['content'] = content
        tool_calls = []
        while '<tool_call>' and '</tool_call>' in assistant_message_str:
            tool_obj = json.loads(assistant_message_str.split('<tool_call>',1)[1].split('</tool_call>',1)[0].strip())
            assert 'name' in tool_obj and 'arguments' in tool_obj, f"Tool call must contain name and arguments, got {tool_obj}"
            tool_obj = {
                "type": "function_call",
                "id": str(uuid.uuid4()),
                "function": {
                    "name": tool_obj['name'],
                    "arguments": json.dumps(tool_obj['arguments'])
                }
            }
            tool_calls.append(tool_obj)
            assistant_message_str = assistant_message_str.split('<tool_call>',1)[1].split('</tool_call>',1)[1]
        if tool_calls:
            assistant_message['tool_calls'] = tool_calls
    else:
        raise ValueError(f"Agent prompt style {agent_prompt_style} is not supported")
    return assistant_message

def get_model_generated_mask_and_tokenwise_reward(
    tokenizer: PreTrainedTokenizer,
    agent_prompt_style: Literal['qwen2_5'],
    chat: list[dict],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    action_turn: list[int],
    reward_by_action_turn: list[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get the model-generated mask and tokenwise reward for the chat.
    Not returning the turn_ids is designed, which might be confusing as not all the tokens are generated by the model in that turn, e.g., <|im_start|>assistant\n
    The turn_ids might cause future misuse.
    """
    assert agent_prompt_style == 'qwen2_5', f"Agent prompt style {agent_prompt_style} is not supported"
    model_generated_mask = torch.zeros_like(attention_mask)
    tokenwise_reward = torch.zeros_like(input_ids).float()
    assert len(action_turn) == len(reward_by_action_turn)

    bos_token_id = tokenizer.convert_tokens_to_ids('<|im_start|>')
    eos_token_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
    generation_prompt_ids = tokenizer.encode('<|im_start|>assistant\n', return_tensors='pt')[0]
    generation_suffix_ids_list = [tokenizer.encode(suffix, return_tensors='pt')[0] for suffix in ['<|im_end|>', '<|im_end|>\n']]

    # Get the position ids of the bos_token_id and eos_token_id
    turn_ids = (input_ids == bos_token_id).long().cumsum(dim=-1) - 1

    if len(chat) != turn_ids[-1].item() + 1:
        print("chat", chat)
        print("turn_ids", turn_ids)

    assert len(chat) == turn_ids[-1].item() + 1, f"The length of the chat {len(chat)} is not equal to the last turn id {turn_ids[-1].item()} + 1. This may due to the additional bos_token_id and eos_token_id added in the chat template."

    for turn, reward in zip(action_turn, reward_by_action_turn):
        # get turn_start_index and turn_end_index
        turn_mask = (turn_ids == turn)
        turn_input_ids = input_ids[turn_mask]
        assert (turn_input_ids[:len(generation_prompt_ids)] == generation_prompt_ids).all(), f"The first {len(generation_prompt_ids)} tokens of the turn {turn} are not equal to the generation prompt ids: {turn_input_ids[:len(generation_prompt_ids)]} != {generation_prompt_ids}"
        # try to match the generation suffix ids
        for generation_suffix_ids in generation_suffix_ids_list:
            if (turn_input_ids[-len(generation_suffix_ids):] == generation_suffix_ids).all():
                break
        else:
            raise ValueError(f"No generation suffix ids found for the turn {turn}, turn_input_ids ends with: {turn_input_ids[-10:]}, and generation_suffix_ids_list is {generation_suffix_ids_list}")
        
        # Find true tokens generated by model, i.e., exclude the generation prompt and generation suffix
        turn_indices = torch.where(turn_mask)[0]
        turn_start_idx = turn_indices[len(generation_prompt_ids)]
        turn_end_idx = turn_indices[-(len(generation_suffix_ids) + 1)]
        
        # Set response mask to 1 for the model-generated tokens (excluding prompt and suffix)
        # The add 1 here means including the ending token
        model_generated_mask[turn_start_idx:turn_end_idx+1] = 1
        
        # Assign reward to all tokens in this turn that were generated by the model
        tokenwise_reward[turn_end_idx] = reward

    return model_generated_mask, tokenwise_reward

# Environment class
class AgentEnv:
    def __init__(
            self, 
            environment_endpoint: str, 
            env_name: str, 
            seed: int, 
            env_kwargs: Union[dict, str], 
            agent_prompt_style: Literal['qwen2_5'],
            tokenizer: PreTrainedTokenizer,
            max_prompt_length: int,
            truncation: Literal['error', 'ignore', 'max_length']
            ):
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
        system_prompt = get_agent_system_prompt(self.agent_prompt_style, task_prompt, tools_schema, allow_parallel_tool_call)
        self.chat = [
            {"role": "system", "content": system_prompt},
            *observation
        ]

    def tokenize_chat(self, add_generation_prompt: bool = False):
        return_dict = {}
        prompt = convert_chat_message_from_openai(self.agent_prompt_style, self.chat)
        prompt_with_chat_template = self.tokenizer.apply_chat_template(prompt, add_generation_prompt=add_generation_prompt, tokenize=False)
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
                self.tokenizer, 
                self.agent_prompt_style, 
                self.chat, 
                input_ids[0], 
                attention_mask[0], 
                self.action_turn, 
                self.reward_by_action_turn
            )
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
            assistant_message = {
                "role": "assistant",
                "content": assistant_message_str
            }
            reward = tool_parsing_error_reward
            done = True
            truncated = False
            info = {}
            error_message = f"Failed to parse the assistant message: {assistant_message_str}, error: {e}"
            observation = [{
                "role": "user",
                "content": error_message
            }]
        self.chat.append(assistant_message)
        self.action_turn.append(len(self.chat) - 1)
        self.reward_by_action_turn.append(reward)
        self.chat += observation
        return observation, reward, done, truncated, info
        
    def __del__(self):
        if self.env_id is not None:
            close_env(self.environment_endpoint, self.env_id)


def collate_fn(data_list: list[dict]) -> dict:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}


class RLAgentDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(self,
                 environment_endpoint: str,
                 parquet_files: Union[str, List[str]],
                 tokenizer: PreTrainedTokenizer,
                 processor: Optional[ProcessorMixin] = None,
                 max_prompt_length=4096,
                 cache_dir='~/.cache/verl/rlhf',
                 truncation='error'):
        self.environment_endpoint = environment_endpoint
        # Test if the environment endpoint is valid
        try:
            response = requests.get(environment_endpoint)
            response.raise_for_status()
        except Exception as e:
            raise ValueError(f"Invalid environment endpoint: {environment_endpoint}")

        if not isinstance(parquet_files, (List, ListConfig)):
            parquet_files = [parquet_files]

        self.parquet_files = copy.deepcopy(parquet_files)
        self.original_parquet_files = copy.deepcopy(parquet_files)  # use for resume
        self.cache_dir = os.path.expanduser(cache_dir)
        self.tokenizer = tokenizer
        self.processor = processor

        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        # TODO: implement the resume feature
        # whether to store the dataset in state_dict()
        # default not store
        # self.serialize_dataset = False
        self._download()
        self._read_files_and_initialize_env()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local
        parquet_files = self.parquet_files if not use_origin_parquet else self.original_parquet_files
        for i, parquet_file in enumerate(parquet_files):
            self.parquet_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir)

    def _read_files_and_initialize_env(self):
        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)

        print(f'dataset len: {len(self.dataframe)}')

    def resume_dataset_state(self):
        raise NotImplementedError("Resume dataset state is not implemented for RLAgentDataset")
        # self.serialize_dataset = False if hasattr(self, 'original_parquet_files') else True
        # # resume dataframe if not it's serialized in data.pt
        # if not self.serialize_dataset:
        #     self._download(use_origin_parquet=True)  # download and resume from original parquet files
        #     self._read_files_and_tokenize()
        # else:
        #     print(r'old dataloader ckpt file is used, please train from scratch for better ckpt performance')

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe.iloc[item].to_dict()
        row_dict['index'] = torch.tensor(item, dtype=torch.int64)
        return row_dict

    def __getstate__(self):
        raise NotImplementedError("Serialize dataset is not implemented for RLAgentDataset")
        # if not self.serialize_dataset:
        #     state = self.__dict__.copy()

        #     if 'dataframe' in state:
        #         del state['dataframe']
        #     return state
        # return self.__dict__.copy()