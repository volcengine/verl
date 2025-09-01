# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from functools import partial
from typing import Any, Optional
from uuid import uuid4

import aiohttp
import json

from verl.utils.rollout_trace import rollout_trace_op
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

from .request_processor import RequestProcessor
from .code_judge_utils import run_tool_calls_on_server_async, generate_tool_call_code, generate_tool_call_input


class CodeJudgeTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        host_addr = self.config.get("host_addr", "localhost")
        host_port = self.config.get("host_port", "8088")
        run_jupyter_tool_calls_on_server_async = partial(
            run_tool_calls_on_server_async,
            generate_tool_call_code=generate_tool_call_code,
            generate_tool_call_input=generate_tool_call_input,
            host_addr=host_addr,
            host_port=host_port,
        )
        request_processor_batch_size = self.config.get("request_processor_batch_size", 1)
        request_processor_concurrency = self.config.get("request_processor_concurrency", 1)
        request_processor_batch_timeout_seconds = self.config.get("request_processor_batch_timeout_seconds", 30)
        tool_connector = aiohttp.TCPConnector(limit=request_processor_concurrency, force_close=True, enable_cleanup_closed=True)
        tool_timeout = aiohttp.ClientTimeout(total=60)
        tool_session = aiohttp.ClientSession(connector=tool_connector, timeout=tool_timeout)
        self.request_processor = RequestProcessor(
            batch_size=request_processor_batch_size,
            batch_timeout_seconds=request_processor_batch_timeout_seconds,
            session=tool_session,
            concurrency=request_processor_concurrency,
            batch_submit_func=run_jupyter_tool_calls_on_server_async,
        )

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def _start_request_processor(self):
        if not self.request_processor._running:
            await self.request_processor.start()

    async def calc_reward(self, instance_id: str, **kwargs) -> str:
        return self._instance_dict[instance_id]["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]


class SimJupyterTool(CodeJudgeTool):
    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        assert "history_tool_calls" in kwargs, "history_tool_calls must be provided in kwargs"
        await self._start_request_processor()
        history_tool_calls = []
        for history_tool_call in kwargs["history_tool_calls"]:
            if history_tool_call.name == "jupyter_code":
                try:
                    arguments = json.loads(history_tool_call.arguments)
                    assert len(arguments) == 1 and "code" in arguments
                    history_tool_calls.append({
                        "name": "jupyter_code",
                        "arguments": {
                            "code": arguments["code"],
                        }
                    })
                except Exception as e:
                    pass

        self._instance_dict[instance_id] = {
            "response": "",
            "reward": [],
            "history_tool_calls": history_tool_calls,
        }
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        code = parameters.get("code", "")
        tool_call = {
            "name": "jupyter_code",
            "arguments": {
                "code": code,
            },
            "history_tool_calls": self._instance_dict[instance_id]["history_tool_calls"]
        }
        result_text = await self.request_processor.send_request(tool_call)
        return ToolResponse(text=result_text), 0.0, {}


class PythonTool(CodeJudgeTool):
    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        await self._start_request_processor()

        self._instance_dict[instance_id] = {
            "response": "",
            "reward": [],
        }
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        code = parameters.get("code", "")
        input = parameters.get("input", "")
        tool_call = {
            "name": "python_code_with_standard_io",
            "arguments": {
                "code": code,
                "input": input,
            },
        }
        result_text = await self.request_processor.send_request(tool_call)
        return ToolResponse(text=result_text), 0.0, {}
