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

import json
from typing import Any, Union

from mcp import McpError, StdioServerParameters
from verl.tools.utils.mcp_clients.config import SSEMCPServer, Settings
from pydantic import ValidationError

from .SseClient import SseClient
from .StdioClient import StdioClient
import logging
logger = logging.getLogger(__name__)


client_types = Union[StdioClient, SseClient]


class MCPClientManager:
    clients: dict[str, client_types] = {}
    initialized = False

    def load_config(self, file: str) -> dict[str, Any]:
        try:
            with open(file, "r") as f:
                return json.load(f)

        except FileNotFoundError:
            logger.warning(f'the "{file}" file was not found')

        except Exception:
            logger.error(f'there was an error reading the "{file}" file')

        return {}

    async def initialize(self, config_path):
        if self.initialized:
            return
        """Initialize the MCP Client Manager and start all clients"""
        result = self.load_config(config_path)
        try:
            config = Settings(**result)
        except ValidationError as e:
            logger.error("unable to load a valid configuration")
            for error in e.errors():
                logger.error(f"{error['loc'][0]}: {error['msg']}")
            exit(1)

        for server_name, server_config in config.mcp_servers.items():
            self.clients[server_name] = await self.construct_client(
                server_name, server_config
            )
        self.initialized = True

    async def construct_client(self, name, server_config) -> client_types:

        if isinstance(server_config, StdioServerParameters):
            client = StdioClient(name, server_config)
            await client.start()
            return client

        if isinstance(server_config, SSEMCPServer):
            # TODO: implement sse client
            client = SseClient(name, server_config)  # type: ignore
            await client.start()
            return client

        raise NotImplementedError("Client Type not supported")

    def get_client(self, server_name: str):
        return self.clients[server_name]

    def get_clients(self):
        return list(self.clients.items())

    async def get_client_from_tool(self, tool: str):
        for _, client in self.get_clients():
            
            # client cannot have tools if it is not connected
            if not client.session:
                continue

            try:
                list_tools = await client.session.list_tools()
                for client_tool in list_tools.tools:
                    if client_tool.name == tool:
                        return client
            except McpError:
                continue


ClientManager = MCPClientManager()
