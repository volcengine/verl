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

from uuid import uuid4

from verl.experimental.agent_loop.tool_parser import ToolParser

from .message_model import InternalChatCmplMessage


async def parse_model_response(
    response_ids: list[int], tokenizer, tool_call_parser: str = "hermes", reasoning_parser: str | None = None
) -> InternalChatCmplMessage:
    """Parse the model response to extract the content, tool calls, and reasoning (if any).
    TODO: implement reasoning_parser
    """
    tool_parser = ToolParser.get_tool_parser(tool_call_parser, tokenizer)
    content, tool_calls = await tool_parser.extract_tool_calls(response_ids)
    reasoning = ""

    tool_calls = [
        {
            "id": f"call_{uuid4().hex}",
            "type": "function",
            "function": tc.model_dump(),
        }
        for tc in tool_calls
    ]
    return InternalChatCmplMessage(
        role="assistant",
        content=content,
        tool_calls=tool_calls if tool_calls else None,
        reasoning_content=reasoning if reasoning else None,
    )
