# Copyright 2025 Individual Contributor: linxxx3 (linxxx3@gmail.com)
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

import time

import pytest

from recipe.agent_lightning_like.message_model import InternalChatCmplMessage, InternalChatCompletion, InternalChoice
from recipe.agent_lightning_like.parser import parse_model_response


def test_message_model():
    tool_call = {
        "id": "call_1",
        "type": "function",
        "function": {
            "name": "func1",
            "arguments": '{"arg1": "value1"}',
        },
    }
    msg = InternalChatCmplMessage(
        role="assistant",
        content="answer here",
        tool_calls=[tool_call],
        reasoning_content="thinking here",
    )
    resp = InternalChatCompletion(
        id="chatcmpl-123",
        object="chat.completion",
        created=int(time.time()),
        model="Default",
        choices=[
            InternalChoice(index=0, message=msg, finish_reason="stop"),
        ],
        usage=None,
    )
    serialized = resp.model_dump()
    assert "choices" in serialized and len(serialized["choices"]) == 1
    choice = serialized["choices"][0]
    assert "message" in choice and choice["message"]["role"] == "assistant"
    assert choice["message"]["content"] == "answer here"
    assert choice["message"]["reasoning_content"] == "thinking here"
    assert "tool_calls" in choice["message"] and len(choice["message"]["tool_calls"]) == 1
    tool_call = choice["message"]["tool_calls"][0]
    assert tool_call["type"] == "function"
    assert tool_call["function"]["name"] == "func1"
    assert tool_call["function"]["arguments"] == '{"arg1": "value1"}'


@pytest.mark.asyncio
async def test_parser_without_reasoning():
    class DummyTokenizer:
        def decode(self, ids, **kwargs):
            return (
                "model response here.\n"
                '<tool_call>{"name": "func1", "arguments": {"arg1": "value1"}}</tool_call>'
                '<tool_call>{"name": "func2", "arguments": {"arg2": "value2"}}</tool_call>'
            )

    tokenizer = DummyTokenizer()
    response_ids = [1, 2, 3]
    message = await parse_model_response(response_ids, tokenizer, tool_call_parser="hermes")
    assert isinstance(message, InternalChatCmplMessage)
    assert message.role == "assistant"
    assert message.content == "model response here.\n"
    assert message.reasoning_content is None
    assert message.tool_calls is not None and len(message.tool_calls) == 2
    assert message.tool_calls[0].function.name == "func1"
    assert message.tool_calls[0].function.arguments == '{"arg1": "value1"}'
    assert message.tool_calls[1].function.name == "func2"
    assert message.tool_calls[1].function.arguments == '{"arg2": "value2"}'
