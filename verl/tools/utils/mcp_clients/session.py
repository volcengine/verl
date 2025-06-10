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

from datetime import timedelta
from typing import Awaitable, Callable

import mcp.types as types
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp.shared.session import BaseSession, RequestResponder
from mcp.shared.version import SUPPORTED_PROTOCOL_VERSIONS

import logging
logger = logging.getLogger(__name__)

sampling_function_signature = Callable[
    [types.CreateMessageRequestParams], Awaitable[types.CreateMessageResult]
]


class McpClientSession(
    BaseSession[
        types.ClientRequest,
        types.ClientNotification,
        types.ClientResult,
        types.ServerRequest,
        types.ServerNotification,
    ]
):

    def __init__(
        self,
        read_stream: MemoryObjectReceiveStream[types.JSONRPCMessage | Exception],
        write_stream: MemoryObjectSendStream[types.JSONRPCMessage],
        read_timeout_seconds: timedelta | None = None,
    ) -> None:
        super().__init__(
            read_stream,
            write_stream,
            types.ServerRequest,
            types.ServerNotification,
            read_timeout_seconds=read_timeout_seconds,
        )

    async def __aenter__(self):
        session = await super().__aenter__()
        self._task_group.start_soon(self._consume_messages)
        return session

    async def _consume_messages(self):
        try:
            async for message in self.incoming_messages:
                try:
                    if isinstance(message, Exception):
                        logger.error(f"Received exception in message stream: {message}")
                    elif isinstance(message, RequestResponder):                        
                        logger.debug(f"Received request: {message.request}")                        
                    elif isinstance(message, types.ServerNotification):
                        if isinstance(message.root, types.LoggingMessageNotification):
                            logger.debug(f"Received notification from server: {message.root.params}")                        
                        else:
                            logger.debug(f"Received notification from server: {message}")                        
                    else:
                        logger.debug(f"Received notification: {message}")
                except Exception as e:
                    logger.exception(f"Error processing message: {e}")
        except Exception as e:
            logger.exception(f"Message consumer task failed: {e}")

    async def initialize(self) -> types.InitializeResult:
        result = await self.send_request(
            types.ClientRequest(
                types.InitializeRequest(
                    method="initialize",
                    params=types.InitializeRequestParams(
                        protocolVersion=types.LATEST_PROTOCOL_VERSION,
                        capabilities=types.ClientCapabilities(
                            sampling=types.SamplingCapability(),
                            experimental=None,
                            roots=types.RootsCapability(
                                listChanged=True
                            ),
                        ),
                        clientInfo=types.Implementation(name="client", version="0.0.1"),
                    ),
                )
            ),
            types.InitializeResult,
        )

        if result.protocolVersion not in SUPPORTED_PROTOCOL_VERSIONS:
            raise RuntimeError(
                "Unsupported protocol version from the server: "
                f"{result.protocolVersion}"
            )

        await self.send_notification(
            types.ClientNotification(
                types.InitializedNotification(method="notifications/initialized")
            )
        )

        return result

    async def send_ping(self) -> types.EmptyResult:
        """Send a ping request."""
        return await self.send_request(
            types.ClientRequest(
                types.PingRequest(
                    method="ping",
                )
            ),
            types.EmptyResult,
        )

    async def send_progress_notification(
        self, progress_token: str | int, progress: float, total: float | None = None
    ) -> None:
        """Send a progress notification."""
        await self.send_notification(
            types.ClientNotification(
                types.ProgressNotification(
                    method="notifications/progress",
                    params=types.ProgressNotificationParams(
                        progressToken=progress_token,
                        progress=progress,
                        total=total,
                    ),
                ),
            )
        )

    async def call_tool(
        self, name: str, arguments: dict | None = None
    ) -> types.CallToolResult:
        """Send a tools/call request."""
        return await self.send_request(
            types.ClientRequest(
                types.CallToolRequest(
                    method="tools/call",
                    params=types.CallToolRequestParams(name=name, arguments=arguments),
                )
            ),
            types.CallToolResult,
        )

    async def complete(
        self, ref: types.ResourceReference | types.PromptReference, argument: dict
    ) -> types.CompleteResult:
        """Send a completion/complete request."""
        return await self.send_request(
            types.ClientRequest(
                types.CompleteRequest(
                    method="completion/complete",
                    params=types.CompleteRequestParams(
                        ref=ref,
                        argument=types.CompletionArgument(**argument),
                    ),
                )
            ),
            types.CompleteResult,
        )

    async def list_tools(self) -> types.ListToolsResult:
        """Send a tools/list request."""
        return await self.send_request(
            types.ClientRequest(
                types.ListToolsRequest(
                    method="tools/list",
                )
            ),
            types.ListToolsResult,
        )
    