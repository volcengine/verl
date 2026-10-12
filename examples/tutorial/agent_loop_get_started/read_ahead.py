# Copyright 2026 Individual Contributor: Jiahui Cao
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
"""ToolAgentLoop example for linked records served from an immutable snapshot."""

import asyncio
import json
from contextvars import ContextVar
from typing import Any

import aiohttp

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
from verl.tools.base_tool import BaseTool
from verl.tools.read_ahead import ReadAhead
from verl.tools.schemas import ToolResponse

_record_reader: ContextVar[tuple[ReadAhead[int, dict], asyncio.Lock]] = ContextVar("record_reader")


class ReadRecordTool(BaseTool):
    """Read a record using the current trajectory's reader."""

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs):
        """Return only the record explicitly requested by the model."""
        record_id = parameters["record_id"]
        if type(record_id) is not int:
            raise ValueError("record_id must be an integer")
        reader, lock = _record_reader.get()
        async with lock:
            result = await reader.read(record_id)
        return ToolResponse(text=json.dumps(result)), 0.0, {}


class ReadAheadAgentLoop(ToolAgentLoop):
    """Scope HTTP reads to one trajectory, including exceptional termination."""

    def __init__(self, *args, records_url: str, read_ahead: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.records_url = records_url.rstrip("/")
        self.read_ahead = read_ahead

    async def run(self, sampling_params: dict[str, Any], priority: int = 0, **kwargs) -> AgentLoopOutput:
        """Keep at most one successor read alive between model turns."""
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as client:

            async def fetch(record_id: int) -> dict:
                async with client.get(f"{self.records_url}/{record_id}.json") as response:
                    response.raise_for_status()
                    return await response.json()

            def successor(record: dict) -> int | None:
                next_id = record.get("next_record_id")
                return next_id if self.read_ahead and type(next_id) is int else None

            reader = ReadAhead(fetch, successor)
            token = _record_reader.set((reader, asyncio.Lock()))
            try:
                return await super().run(sampling_params, priority=priority, **kwargs)
            finally:
                _record_reader.reset(token)
                await reader.aclose()
