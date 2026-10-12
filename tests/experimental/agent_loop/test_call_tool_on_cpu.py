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

"""Unit tests for ToolAgentLoop._call_tool error handling (no GPU required).

Tests that malformed tool calls return specific, actionable error messages
instead of generic exception strings.
"""

import asyncio
import json
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from verl.tools.schemas import ToolResponse


@dataclass
class FakeFunctionCall:
    """Minimal FunctionCall for testing."""

    name: str
    arguments: str


@dataclass
class FakeAgentData:
    """Minimal AgentData for testing."""

    tools_kwargs: dict = field(default_factory=dict)


class FakeTool:
    """A fake tool that succeeds."""

    def __init__(self, name: str):
        self.name = name

    async def create(self, create_kwargs=None):
        return "instance_1", ToolResponse()

    async def execute(self, instance_id, parameters, **kwargs):
        return ToolResponse(text=f"OK: {parameters}"), 1.0, {}

    async def release(self, instance_id):
        pass


class FakeFailingTool(FakeTool):
    """A fake tool that raises during execute."""

    async def execute(self, instance_id, parameters, **kwargs):
        raise RuntimeError("database connection failed")


class FakeLongResponseTool(FakeTool):
    """A fake tool that returns a long response."""

    def __init__(self, name: str, text: str):
        super().__init__(name)
        self.text = text

    async def execute(self, instance_id, parameters, **kwargs):
        return ToolResponse(text=self.text), 1.0, {}


def _make_tool_agent_loop(
    tools: dict[str, Any],
    max_tool_response_length: int = 10000,
    tool_response_truncate_side: str = "left",
):
    """Create a minimal ToolAgentLoop instance with only the fields _call_tool needs."""
    from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop

    mock = MagicMock(spec=ToolAgentLoop)
    mock.tools = tools
    mock.max_tool_response_length = max_tool_response_length
    mock.tool_response_truncate_side = tool_response_truncate_side
    # Bind the real _call_tool method to our mock
    mock._call_tool = ToolAgentLoop._call_tool.__get__(mock, ToolAgentLoop)
    return mock


class TestCallToolErrorHandling(unittest.IsolatedAsyncioTestCase):
    """Test ToolAgentLoop._call_tool error handling for malformed tool calls."""

    def setUp(self):
        self.tools = {
            "calculator": FakeTool("calculator"),
            "search": FakeTool("search"),
        }
        self.loop = _make_tool_agent_loop(self.tools)
        self.agent_data = FakeAgentData()

    async def test_valid_tool_call(self):
        """Valid tool call should succeed."""
        tool_call = FakeFunctionCall(name="calculator", arguments='{"a": 3, "b": 5}')
        response, reward, _ = await self.loop._call_tool(tool_call, {}, self.agent_data)
        assert reward == 1.0
        assert "OK" in response.text

    async def test_unknown_function_name(self):
        """Unknown function name should list available tools."""
        tool_call = FakeFunctionCall(name="calculater", arguments='{"a": 3}')
        response, reward, _ = await self.loop._call_tool(tool_call, {}, self.agent_data)
        assert reward == 0.0
        assert "Unknown function" in response.text
        assert "calculater" in response.text
        assert "calculator" in response.text
        assert "search" in response.text

    async def test_invalid_json_arguments(self):
        """Invalid JSON arguments should report parse error."""
        tool_call = FakeFunctionCall(name="calculator", arguments="{a: 3}")
        response, reward, _ = await self.loop._call_tool(tool_call, {}, self.agent_data)
        assert reward == 0.0
        assert "Invalid JSON" in response.text
        assert "calculator" in response.text

    async def test_empty_arguments(self):
        """Empty string arguments should report parse error."""
        tool_call = FakeFunctionCall(name="calculator", arguments="")
        response, reward, _ = await self.loop._call_tool(tool_call, {}, self.agent_data)
        assert reward == 0.0
        assert "Invalid JSON" in response.text

    async def test_none_arguments(self):
        """None arguments should report error."""
        tool_call = FakeFunctionCall(name="calculator", arguments=None)
        response, reward, _ = await self.loop._call_tool(tool_call, {}, self.agent_data)
        assert reward == 0.0
        assert "Invalid JSON" in response.text

    async def test_tool_execution_error(self):
        """Tool execution failure should include tool name in error."""
        tools = {"failing_tool": FakeFailingTool("failing_tool")}
        loop = _make_tool_agent_loop(tools)
        tool_call = FakeFunctionCall(name="failing_tool", arguments='{"query": "test"}')
        response, reward, _ = await loop._call_tool(tool_call, {}, self.agent_data)
        assert reward == 0.0
        assert "failing_tool" in response.text
        assert "database connection failed" in response.text

    async def test_left_truncation_keeps_response_tail(self):
        """Left truncation should drop the left side and preserve the response tail."""
        tool_response = (
            "Search results for capital of France:\n"
            "1. Lyon is a major city with a long Roman history.\n"
            "2. Marseille is a large port city in southern France.\n"
            "3. The final retrieved snippet says the capital is Paris.\n"
            "Final answer: Paris"
        )
        tools = {"search": FakeLongResponseTool("search", tool_response)}
        loop = _make_tool_agent_loop(tools, max_tool_response_length=19, tool_response_truncate_side="left")
        tool_call = FakeFunctionCall(name="search", arguments="{}")
        response, reward, _ = await loop._call_tool(tool_call, {}, self.agent_data)
        assert reward == 1.0
        assert response.text.startswith("(truncated)...")
        assert response.text.endswith("Final answer: Paris")

    async def test_right_truncation_keeps_response_head(self):
        """Right truncation should drop the right side and preserve the response head."""
        tool_response = (
            "Search results for capital of France:\n"
            "1. Lyon is a major city with a long Roman history.\n"
            "2. Marseille is a large port city in southern France.\n"
            "3. The final retrieved snippet says the capital is Paris.\n"
            "Final answer: Paris"
        )
        tools = {"search": FakeLongResponseTool("search", tool_response)}
        loop = _make_tool_agent_loop(tools, max_tool_response_length=19, tool_response_truncate_side="right")
        tool_call = FakeFunctionCall(name="search", arguments="{}")
        response, reward, _ = await loop._call_tool(tool_call, {}, self.agent_data)
        assert reward == 1.0
        assert response.text.startswith("Search results")
        assert response.text.endswith("...(truncated)")
        assert "Final answer: Paris" not in response.text


def _make_read_ahead_loop(records_url, enabled):
    from examples.tutorial.agent_loop_get_started.read_ahead import ReadAheadAgentLoop
    from verl.tools.tool_registry import initialize_tools_from_config

    config = Path(__file__).parents[3] / "examples/tutorial/agent_loop_get_started/read_ahead_tools.yaml"
    tools = initialize_tools_from_config(config)
    loop = object.__new__(ReadAheadAgentLoop)
    loop.tools = {tool.name: tool for tool in tools}
    loop.records_url = records_url
    loop.read_ahead = enabled
    loop.max_tool_response_length = 10000
    loop.tool_response_truncate_side = "left"
    return loop


async def _read_record(loop, record_id):
    call = FakeFunctionCall("read_record", json.dumps({"record_id": record_id}))
    response, _, _ = await loop._call_tool(call, {}, FakeAgentData())
    return json.loads(response.text)


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_read_ahead_example_isolates_concurrent_trajectories(monkeypatch, enabled):
    """Matching keys in two trajectories must not share speculative results."""
    from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop

    calls = []
    both_prefetched = asyncio.Event()

    async def fetch(request):
        record_id = int(request.match_info["record_id"])
        calls.append(record_id)
        if calls.count(2) == 2:
            both_prefetched.set()
        return web.json_response({"record_id": record_id, "next_record_id": 2 if record_id == 1 else None})

    async def model_turns(loop, *args, **kwargs):
        first = await _read_record(loop, 1)
        if enabled:
            await asyncio.wait_for(both_prefetched.wait(), timeout=5)
        second = await _read_record(loop, first["next_record_id"])
        return [first, second]

    monkeypatch.setattr(ToolAgentLoop, "run", model_turns)
    app = web.Application()
    app.router.add_get("/{record_id}.json", fetch)
    async with TestServer(app) as server:
        loop = _make_read_ahead_loop(str(server.make_url("")).rstrip("/"), enabled)
        outputs = await asyncio.gather(loop.run({}), loop.run({}))
    assert outputs == [[{"record_id": 1, "next_record_id": 2}, {"record_id": 2, "next_record_id": None}]] * 2
    assert sorted(calls) == [1, 1, 2, 2]


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["return", "error", "cancel"])
async def test_read_ahead_example_closes_unused_http_read(monkeypatch, ending):
    """The trajectory owns prefetch cleanup, not the per-call tool release."""
    from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop

    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def fetch(request):
        if request.match_info["record_id"] == "2":
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        return web.json_response({"record_id": 1, "next_record_id": 2})

    async def model_turns(loop, *args, **kwargs):
        await _read_record(loop, 1)
        await asyncio.wait_for(started.wait(), timeout=5)
        if ending == "error":
            raise RuntimeError("model failed")
        if ending == "cancel":
            await asyncio.Event().wait()
        return "finished"

    monkeypatch.setattr(ToolAgentLoop, "run", model_turns)
    app = web.Application()
    app.router.add_get("/{record_id}.json", fetch)
    server = TestServer(app)
    await server.start_server()
    try:
        loop = _make_read_ahead_loop(str(server.make_url("")).rstrip("/"), True)
        task = asyncio.create_task(loop.run({}))
        if ending == "cancel":
            await asyncio.wait_for(started.wait(), timeout=5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        elif ending == "error":
            with pytest.raises(RuntimeError, match="model failed"):
                await task
        else:
            assert await task == "finished"
        await asyncio.wait_for(cancelled.wait(), timeout=5)
    finally:
        await server.close()


if __name__ == "__main__":
    unittest.main()
