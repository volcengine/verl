# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import logging
from typing import List

from mcp import Tool
from verl.tools.utils.mcp_clients.McpClientManager import ClientManager


logger = logging.getLogger(__file__)

async def add_mcp_tools(tool_selected_list: List[str]) -> List[dict]:
    tool_schemas = []

    for _, session in ClientManager.get_clients():
        # if session is None, then the client is not running
        if session.session is None:
            logger.error(f"session is `None` for {session.name}")
            continue

        tools = await session.session.list_tools()
        for tool in tools.tools:
            if tool.name in tool_selected_list:
                tool_schemas.append(mcp2openai(tool))

    return tool_schemas


def mcp2openai(mcp_tool: Tool) -> dict:
    """Convert a MCP Tool to an OpenAI ChatCompletionTool."""
    openai_format = {
        "type": "function",
        "function": {
            "name": mcp_tool.name,
            "description": mcp_tool.description,
            "parameters": mcp_tool.inputSchema,
            "strict": False,
        }
    }
    if not openai_format['function']['parameters'].get('required', None):
        openai_format['function']['parameters']['required'] = []
    return openai_format