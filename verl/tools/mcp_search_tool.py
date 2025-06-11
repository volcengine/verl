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
import logging
import os
from typing import TypeVar

from verl.tools.mcp_base_tool import MCPBaseTool

from .schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

T = TypeVar("T")


class MCPSearchTool(MCPBaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)

    def post_process(self, content: list):
        res = ""
        res_cnt = 0
        query_list = []
        metadata = {
            "api_request_error": "",
            "status": "unknown",
            "total_results": 0,
        }
        try:
            for part in content:
                if part.type != "text":
                    continue
                # FIXME(hechanghao): use json load for parsing result string
                res += part.text
                res_cnt += 1
        except json.JSONDecodeError:
            err_msg = "json parse error."
            logger.error(err_msg)
            metadata["api_request_error"] = err_msg
            metadata["status"] = "error"

        # update metadata
        metadata["status"] = "success"
        metadata["queries"] = query_list
        metadata["query_count"] = len(query_list)
        metadata["total_results"] = res_cnt
        return res, metadata
