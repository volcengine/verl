# Copyright 2024 CollabLLM Ltd. and/or its affiliates
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

import asyncio
import copy
import logging
import os
from typing import Any, Optional
from uuid import uuid4

from recipe.collabllm.utils import remove_think_block
from verl.interactions.base import BaseInteraction
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

TERMINATION_SIGNAL = "[[TERMINATE CHAT]]"
USER_PROMPT_TEMPLATE = """You are role-playing as a human USER interacting with an AI collaborator to complete a specific task. Your goal is to generate realistic, natural responses that a user might give in this scenario.

## Input Information:
You will be provided with:
- Task Description: The type of task you are trying to accomplish.
- Complete Prompt or Reference Goal: This field may include the complete user request/query or a reference answer to user's request. Use this field to understand the user's intent, requirements, or what would count as a satisfactory outcome.
- Chat History: The ongoing conversation between you (as the user) and the AI

Inputs:
<|The Start of Task Description (Not visible to the AI)|>
{task_desc}
<|The End of Task Description|>

<|The Start of Complete Prompt or Reference Goal (Not visible to the AI)|>
{single_turn_prompt}
<|The End of Complete Prompt or Reference Goal|>

<|The Start of Chat History|>
{chat_history}
<|The End of Chat History|>


## Guidelines:
- Stay in Character: Role-play as a human USER. You are NOT an AI. Maintain a consistent personality throughout the chat.
- Minimize Effort: IMPORTANT! As a user, avoid being too detailed in your responses. Provide vague or incomplete demands in the early stages of the conversation to minimize your effort. Let the AI ask for clarification rather than providing everything upfront.
- Knowledge Background: Reflect the user's knowledge level in the role-playing. If the user is less knowledgeable about a task, they might not notice incorrect statements. Ask questions that demonstrate your current understanding and areas of confusion.
- Occasionally Make Mistakes: Real-world users might misspell words, provide incorrect dates, give wrong information, or ask unclear questions. Simulate this behavior to reflect natural interactions.
- Mention Personal Preferences: Include preferences or constraints that might influence your requests or responses. For example, "I prefer short answers," "I need this done quickly," or "I like detailed comments in code."
- Goal-Oriented: Keep the chat focused on your intent. Avoid small talk or digressions. Redirect the chat back to the main objective if it starts to stray.

## Output Format:
You should output a JSON object with three entries:
- "current_answer" (str): Briefly summerize the AI's current solution to the task.
- "thought" (str): Output your thought process as a user deciding what to say next. Consider:
1. Have you obtained a satisfactory solution from the AI? If yes, you can terminate this chat.
2. If not, what specific part of the problem or solution are you struggling with?
3. Has the AI asked you to perform a task or answer a question? If so, how should you approach it?
4. Are you noticing any patterns or potential misunderstandings that need clarification?
5. If you're stuck, how can you phrase your question to get the most helpful response while demonstrating your current understanding?
- "response" (str): Based on your thought process, respond to the AI as the user you are role-playing. Stop immediately when the user's response is completed.

## Important Notes:
- Respond Based on Previous Messages: Your responses should be based on the context of the current chat history. Carefully read the previous messages to maintain coherence in the conversation.
- Conversation Flow: If "Current Chat History" is empty, start the conversation from scratch with an initial request. Otherwise, continue based on the existing conversation.
- Don't Copy Input Directly: Use the provided information for understanding context only. Avoid copying target queries or any provided information directly in your responses.
- Completion Signal: Use "{termination_signal}" as your response when you believe your goal has been solved or if you determine the AI cannot help further.
- Double check if the JSON object is formatted correctly. Ensure that all fields are present and properly structured.

Remember to stay in character as a user throughout your response, and follow the instructions and guidelines carefully."""  # noqa


class CollabLLMInteraction(BaseInteraction):
    """A demo interaction for calculating the reward of CollabLLM.

    - `start_interaction`: start a interaction instance for a trajectory.
    - `generate_response`: generate the response of the assistant.
    - `calculate_score`: calculate the score of the interaction.
    - `finalize_interaction`: finalize the interaction instance.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        _config = copy.deepcopy(config)

        _config.pop("enable_log", None)

        self.name = _config.pop("name")
        self.user_model = _config.pop("user_model")

        self.termination_signal = _config.pop("termination_signal", TERMINATION_SIGNAL)
        self.num_retries = _config.pop("num_retries", 3)

        self.user_model_kwargs = _config

        self._instance_dict = {}

    async def start_interaction(
        self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs
    ) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": ground_truth,
            "reward": 0.0,
        }
        self.interaction_kwargs = kwargs
        assert "single_turn_prompt" in kwargs, "single_turn_prompt is required in interaction_kwargs"
        return instance_id

    @rollout_trace_op
    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], **kwargs
    ) -> tuple[bool, str, float, dict]:
        assert messages[-1]["role"] in ["system", "assistant"], (
            "Last message input to the user model must be from system or assistant role"
        )

        import litellm

        chat_history = self._parse_messages(messages, strip_sys_prompt=True)
        prompt = USER_PROMPT_TEMPLATE.format(
            task_desc=self.interaction_kwargs.get("task_desc", "general assistance task"),
            single_turn_prompt=self.interaction_kwargs["single_turn_prompt"],
            chat_history=chat_history,
            termination_signal=self.termination_signal,
        )
        response = ""
        for i in range(self.num_retries):
            try:
                full_response = (
                    (
                        await litellm.acompletion(
                            model=self.user_model,
                            messages=[{"role": "user", "content": prompt}],
                            **self.user_model_kwargs,
                        )
                    )
                    .choices[0]
                    .message.content
                )
            except litellm.RateLimitError as e:
                logger.warning(f"[CollabLLMInteraction] hit RateLimitError: {e}. Retrying...")
                await asyncio.sleep(max(2**i, 60))
                continue
            except Exception as e:
                logger.exception(f"An unexpected error occurred in CollabLLMAgentLoop: {e}")
                continue

            try:
                if isinstance(full_response, str):
                    full_response = extract_json(full_response)
            except Exception as e:
                logger.warning(f"[CollabLLMInteraction] Error extracting JSON: {e}. Retrying...")
                continue

            if isinstance(full_response, dict):
                keys = full_response.keys()
                if {"current_answer", "thought", "response"}.issubset(keys):
                    response = full_response.pop("response")
                    if isinstance(response, str):
                        break
                    else:
                        logger.warning(
                            f"[CollabLLMInteraction] got an invaild response {response} full_response {full_response}. \
                                Retrying..."
                        )
                        continue
                else:
                    logger.warning(f"[CollabLLMInteraction] Keys {keys} do not match expected keys. Retrying...")
                    continue

        self._instance_dict[instance_id]["response"] = response
        logger.debug(f"[CollabLLMInteraction] User: {response}")
        should_terminate_sequence = self.termination_signal in response
        reward = 0.0

        return should_terminate_sequence, response, reward, {}

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]

    def _parse_messages(self, messages, strip_sys_prompt=True):
        if messages is None:
            return ""

        if strip_sys_prompt:
            messages = [msg for msg in messages if msg["role"] != "system"]

        messages = [remove_think_block(msg) for msg in messages]

        chat = "\n".join(f"**{m['role'].capitalize()}**: {m['content']}" for m in messages)

        return chat


def extract_json(s):
    def convert_value(value):
        true_values = {"true": True, "false": False, "null": None}
        value_lower = value.lower()
        if value_lower in true_values:
            return true_values[value_lower]
        try:
            if "." in value or "e" in value.lower():
                return float(value)
            else:
                return int(value)
        except ValueError:
            return value  # Return as string if not a number

    def parse_number(s, pos):
        start = pos
        while pos < len(s) and s[pos] in "-+0123456789.eE":
            pos += 1
        num_str = s[start:pos]
        try:
            if "." in num_str or "e" in num_str.lower():
                return float(num_str), pos
            else:
                return int(num_str), pos
        except ValueError:
            logger.error(f"Invalid number at position {start}: {num_str}")
            raise

    def skip_whitespace(s, pos):
        while pos < len(s) and s[pos] in " \t\n\r":
            pos += 1
        return pos

    def parse_string(s, pos):
        quote_char = s[pos]
        assert quote_char in ('"', "'")
        pos += 1
        result = ""
        while pos < len(s):
            c = s[pos]
            if c == "\\":
                pos += 1
                if pos >= len(s):
                    raise ValueError("Invalid escape sequence")
                c = s[pos]
                escape_sequences = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", quote_char: quote_char}
                result += escape_sequences.get(c, c)
            elif c == quote_char:
                pos += 1
                # Attempt to convert to a number if possible
                converted_value = convert_value(result)
                return converted_value, pos
            else:
                result += c
            pos += 1
        raise ValueError("Unterminated string")

    def parse_key(s, pos):
        pos = skip_whitespace(s, pos)
        if s[pos] in ('"', "'"):
            key, pos = parse_string(s, pos)
            return key, pos
        else:
            raise ValueError(f"Expected string for key at position {pos}")

    def parse_object(s, pos):
        obj = {}
        assert s[pos] == "{"
        pos += 1
        pos = skip_whitespace(s, pos)
        while pos < len(s) and s[pos] != "}":
            pos = skip_whitespace(s, pos)
            key, pos = parse_key(s, pos)
            pos = skip_whitespace(s, pos)
            if pos >= len(s) or s[pos] != ":":
                raise ValueError(f'Expected ":" at position {pos}')
            pos += 1
            pos = skip_whitespace(s, pos)
            value, pos = parse_value(s, pos)
            obj[key] = value
            pos = skip_whitespace(s, pos)
            if pos < len(s) and s[pos] == ",":
                pos += 1
                pos = skip_whitespace(s, pos)
            elif pos < len(s) and s[pos] == "}":
                break
            elif pos < len(s) and s[pos] != "}":
                raise ValueError(f'Expected "," or "}}" at position {pos}')
        if pos >= len(s) or s[pos] != "}":
            raise ValueError(f'Expected "}}" at position {pos}')
        pos += 1
        return obj, pos

    def parse_array(s, pos):
        lst = []
        assert s[pos] == "["
        pos += 1
        pos = skip_whitespace(s, pos)
        while pos < len(s) and s[pos] != "]":
            value, pos = parse_value(s, pos)
            lst.append(value)
            pos = skip_whitespace(s, pos)
            if pos < len(s) and s[pos] == ",":
                pos += 1
                pos = skip_whitespace(s, pos)
            elif pos < len(s) and s[pos] == "]":
                break
            elif pos < len(s) and s[pos] != "]":
                raise ValueError(f'Expected "," or "]" at position {pos}')
        if pos >= len(s) or s[pos] != "]":
            raise ValueError(f'Expected "]" at position {pos}')
        pos += 1
        return lst, pos

    def parse_triple_quoted_string(s, pos):
        if s[pos : pos + 3] == "'''":
            quote_str = "'''"
        elif s[pos : pos + 3] == '"""':
            quote_str = '"""'
        else:
            raise ValueError(f"Expected triple quotes at position {pos}")
        pos += 3
        result = ""
        while pos < len(s):
            if s[pos : pos + 3] == quote_str:
                pos += 3
                # Attempt to convert to a number if possible
                converted_value = convert_value(result)
                return converted_value, pos
            else:
                result += s[pos]
                pos += 1
        raise ValueError("Unterminated triple-quoted string")

    def parse_value(s, pos):
        pos = skip_whitespace(s, pos)
        if pos >= len(s):
            raise ValueError("Unexpected end of input")
        if s[pos] == "{":
            return parse_object(s, pos)
        elif s[pos] == "[":
            return parse_array(s, pos)
        elif s[pos : pos + 3] in ("'''", '"""'):
            return parse_triple_quoted_string(s, pos)
        elif s[pos] in ('"', "'"):
            return parse_string(s, pos)
        elif s[pos : pos + 4].lower() == "true":
            return True, pos + 4
        elif s[pos : pos + 5].lower() == "false":
            return False, pos + 5
        elif s[pos : pos + 4].lower() == "null":
            return None, pos + 4
        elif s[pos] in "-+0123456789.":
            return parse_number(s, pos)
        else:
            raise ValueError(f"Unexpected character at position {pos}: {s[pos]}")

    json_start = s.index("{")
    json_end = s.rfind("}")
    s = s[json_start : json_end + 1]

    s = s.strip()
    result, pos = parse_value(s, 0)
    pos = skip_whitespace(s, pos)
    if pos != len(s):
        raise ValueError(f"Unexpected content at position {pos}")
    return result
