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
from mcp import StdioServerParameters, stdio_client

from verl.tools.utils.mcp_clients.session import McpClientSession
from .AbstractClient import GenericMcpClient
import shutil
import os
import logging
logger = logging.getLogger(__name__)


# Keywords to identify virtual environment variables
venv_keywords = ["CONDA", "VIRTUAL", "PYTHON"]

class StdioClient(GenericMcpClient):
    config: StdioServerParameters

    def __init__(self, name: str, config: StdioServerParameters) -> None:
        super().__init__(name=name)

        own_config = config.model_copy(deep=True)

        env = dict(os.environ.copy())

        env = {
            key: value for key, value in env.items()
            if not any(key.startswith(keyword) for keyword in venv_keywords)
        }

        if config.env is not None:
            env.update(config.env)

        own_config.env = env

        command = shutil.which(config.command)
        if command is None:
            logger.error(f"could not find command {config.command}")
            exit(1)

        own_config.command = command

        # this changes the default to ignore
        if "encoding_error_handler" not in config.model_fields_set:
            own_config.encoding_error_handler = "ignore"

        self.config = own_config

    async def _maintain_session(self):
        logger.debug(f"starting maintain session for {self.name}")
        async with stdio_client(self.config) as client:
            logger.debug(f"entered stdio_client context manager for {self.name}")
            assert client[0] is not None, f"missing read stream for {self.name}"
            assert client[1] is not None, f"missing write stream for {self.name}"
            async with McpClientSession(*client) as session:
                logger.debug(f"entered client session context manager for {self.name}")
                await session.initialize()
                logger.debug(f"finished initialise session for {self.name}")
                self.session = session

                try:
                    while True:
                        await asyncio.sleep(10)
                        await session.send_ping()

                except Exception as exc:
                    logger.error(f"ping failed for {self.name}: {exc}")
                    self.session = None

        logger.debug(f"exiting session for {self.name}")
