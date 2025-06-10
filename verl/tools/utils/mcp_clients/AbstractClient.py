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

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Optional
from mcp import McpError
from mcp.types import (
    CallToolResult,
    ListToolsResult,
    TextContent
)

from verl.tools.utils.mcp_clients.mcpServerStatus import McpServerStatus
from verl.tools.utils.mcp_clients.session import McpClientSession
import logging
logger = logging.getLogger(__name__)

class GenericMcpClient(ABC):
    name: str
    config: Any
    client: Any
    session: McpClientSession | None = None

    def __init__(self, name: str) -> None:
        super().__init__()
        self.session = None
        self.name = name

        logger.debug(f"initializing client class for {name}")

    @abstractmethod
    async def _maintain_session(self):
        pass

    async def _session_maintainer(self):
        while True:
            try:
                await self._maintain_session()
            except FileNotFoundError as e:
                logger.error(f"failed to maintain session for {self.name}: file {e.filename} not found.")
            except Exception as e:
                logger.error(f"failed to maintain session for {self.name}: {type(e)} {e.args}")

            logger.debug(f"restarting session for {self.name}")
            await asyncio.sleep(0.5)

    async def start(self):
        asyncio.create_task(self._session_maintainer())

    async def call_tool(
        self, name: str, arguments: dict, timeout: Optional[int] = None
    ) -> CallToolResult:
        await self._wait_for_session()

        try:
            return await self.session.call_tool(
                name=name,
                arguments=arguments,
            )

        except asyncio.TimeoutError:
            logger.error(f"timed out calling tool: {name}")
            return CallToolResult(
                content=[
                    TextContent(type="text", text=f"Timeout Error calling {name}")
                ],
                isError=True,
            )

        except McpError as e:
            logger.error(f"error calling {name}: {e}")
            return CallToolResult(
                content=[TextContent(type="text", text=f"Error calling {name}: {e}")],
                isError=True,
            )

    async def list_tools(self) -> ListToolsResult:
        # if session is None, then the client is not running
        # wait to see if it restarts
        await self._wait_for_session()

        try:
            return await self.session.list_tools()
        except Exception as e:
            logger.error(f"error listing tools: {e}")
            return ListToolsResult(tools=[])

    async def _wait_for_session(self, timeout: int = 5, http_error: bool = True):
        try:
            # FIXME a way to solve timeout
            while self.session is None:
                await asyncio.sleep(1)
                logger.debug(f"waiting for session for {self.name}")

        except asyncio.TimeoutError:
            if http_error:
                raise McpError(
                    message=f"Could not connect to MCP server \"{self.name}\".",
                    code=500
                )

            raise TimeoutError(f"Could not connect to MCP server \"{self.name}\"." )

        assert self.session is not None, "Session is None"

    async def status(self) -> McpServerStatus:
        """Get the status of the MCP server"""
        return McpServerStatus(
            name=self.name, online=self.session is not None, enabled=True
        )
