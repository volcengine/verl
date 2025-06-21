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
import re
from typing import Tuple

from verl.tools.mcp_base_tool import MCPBaseTool

from .schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class MCPSandboxTool(MCPBaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)

    def _parse_tool_result(self, content: list) -> Tuple[str, dict]:
        metadata = {
            "status": "unknown",
            "total_results": 1,
            "run_success": False,  # A boolean flag indicating whether the code in the sandbox executed successfully
            "stdout": "",
            "stderr": "",
            "error_type": None,
            "api_request_error": "",
            "query_count": 1,
        }
        result_text = ""

        try:
            if not content or not isinstance(content, list) or len(content) == 0:
                raise ValueError("Tool result content is empty or not a list.")

            if content[0].type != "text":
                raise ValueError(f"Unsupported tool output type: '{content[0].type}'")

            outer_json_str = content[0].text
            outer_data = json.loads(outer_json_str)
            body_data = json.loads(outer_data.get("body"))

            status_code = outer_data.get("statusCode")

            if status_code == 200:
                metadata["run_success"] = True
                metadata["stdout"] = body_data.get("run_result", "")
                result_text = metadata["stdout"]
            elif status_code == 500:
                run_result = json.loads(body_data.get("run_result"))
                execution_details = run_result.get("run_result", {})

                metadata["stderr"] = execution_details["stderr"]
                metadata["stdout"] = execution_details["stdout"]

                result_text = metadata["stderr"].strip().replace("\\n", "\n")
                match = re.search(r"^([a-zA-Z_]\w*Error):", result_text, re.MULTILINE)
                if match:
                    metadata["error_type"] = match.group(1)
            else:
                raise ValueError(f"Unhandled statusCode: {metadata['status_code']}")
        except Exception as e:
            err_msg = f"Failed to parse tool result: {type(e).__name__} - {e}"
            logger.error(err_msg)
            metadata["api_request_error"] = err_msg
            metadata["status"] = "error"

        metadata["status"] = "success"

        return result_text, metadata
