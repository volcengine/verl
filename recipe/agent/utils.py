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


import json
import torch
import requests
import numpy as np
from collections import defaultdict
from typing import List, Union, Optional, Literal
from transformers import PreTrainedTokenizer

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
        # Remove the leading newline characters
        # This is to make sure the turn parsing is correct, this is however may cause misalignment with the original assistant message
        assistant_message_str = assistant_message_str.lstrip()
        content = ""
        if not assistant_message_str.startswith('<tool_call>'):
            content = assistant_message_str.split('<tool_call>', 1)[0]
        assistant_message['content'] = content
        tool_calls = []
        while '<tool_call>' and '</tool_call>' in assistant_message_str:
            tool_obj = json.loads(assistant_message_str.split('<tool_call>', 1)[1].split('</tool_call>', 1)[0].strip())
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
            assistant_message_str = assistant_message_str.split('<tool_call>', 1)[1].split('</tool_call>', 1)[1]
        if tool_calls:
            assistant_message['tool_calls'] = tool_calls
    else:
        raise ValueError(f"Agent prompt style {agent_prompt_style} is not supported")
    return assistant_message

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
                assert isinstance(
                    message['content'],
                    str), f"System message must be a string, got {type(message['content'])}, which is not supported"
                new_chat.append(message)
            elif message['role'] == 'user':
                if isinstance(message['content'], str):
                    new_chat.append(message)
                else:
                    raise ValueError(
                        f"User message must be a string, got {type(message['content'])}, which is not supported")
            elif message['role'] == 'tool':
                assert isinstance(
                    message['content'],
                    str), f"Tool message must be a string, got {type(message['content'])}, which is not supported"
                new_chat.append({
                    "role": "user",
                    "content": "<tool_response>\n" + message['content'] + "\n</tool_response>"
                })
            elif message['role'] == 'assistant':
                assert message['content'] is None or isinstance(
                    message['content'],
                    str), f"Assistant message must be a string, got {type(message['content'])}, which is not supported"
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
                    new_chat.append({"role": "assistant", "content": content})
            else:
                raise ValueError(f"Message role {message['role']} is not supported")
    else:
        raise ValueError(f"Agent prompt style {agent_prompt_style} is not supported")
    return new_chat

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
    generation_suffix_ids_list = [tokenizer.encode(suffix, return_tensors='pt')[0] for suffix in ['<|im_end|>\n']]
    # generation_suffix_ids_list = [tokenizer.encode(suffix, return_tensors='pt')[0] for suffix in ['<|im_end|>', '<|im_end|>\n']]

    # Get the position ids of the bos_token_id and eos_token_id
    turn_ids = (input_ids == bos_token_id).long().cumsum(dim=-1) - 1

    if len(chat) != turn_ids[-1].item() + 1:
        print("chat", chat)
        print("turn_ids", turn_ids)

    assert len(chat) == turn_ids[-1].item(
    ) + 1, f"The length of the chat {len(chat)} is not equal to the last turn id {turn_ids[-1].item()} + 1. This may due to the additional bos_token_id and eos_token_id added in the chat template."

    for turn, reward in zip(action_turn, reward_by_action_turn):
        # get turn_start_index and turn_end_index
        turn_mask = ((turn_ids == turn) & attention_mask.bool())
        turn_input_ids = input_ids[turn_mask]
        assert (turn_input_ids[:len(generation_prompt_ids)] == generation_prompt_ids).all(
        ), f"The first {len(generation_prompt_ids)} tokens of the turn {turn} are not equal to the generation prompt ids: {turn_input_ids[:len(generation_prompt_ids)]} != {generation_prompt_ids}.\n{turn_input_ids=}\n{chat=}"
        # try to match the generation suffix ids
        for generation_suffix_ids in generation_suffix_ids_list:
            if (turn_input_ids[-len(generation_suffix_ids):] == generation_suffix_ids).all():
                break
        else:
            raise ValueError(
                f"No generation suffix ids found for the turn {turn}, turn_input_ids ends with: {turn_input_ids[-10:]}, and generation_suffix_ids_list is {generation_suffix_ids_list}"
            )

        # Find true tokens generated by model, i.e., exclude the generation prompt and generation suffix
        turn_indices = torch.where(turn_mask)[0]
        turn_start_idx = turn_indices[len(generation_prompt_ids)]
        turn_end_idx = turn_indices[-(len(generation_suffix_ids) + 1)]

        # Set response mask to 1 for the model-generated tokens (excluding prompt and suffix)
        # The add 1 here means including the ending token
        model_generated_mask[turn_start_idx:turn_end_idx + 1] = 1

        # Assign reward to all tokens in this turn that were generated by the model
        tokenwise_reward[turn_end_idx] = reward

    return model_generated_mask, tokenwise_reward

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