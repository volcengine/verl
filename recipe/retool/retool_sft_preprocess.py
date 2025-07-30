# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Convert JoeYing/ReTool-SFT to standard multi-turn tool calling messages.
"""

import json
import os
import re
from typing import Any

import datasets
from omegaconf import OmegaConf

code_pattern = re.compile(r"```python(.*?)```", re.DOTALL)


def extract_code_message(content: str) -> tuple[dict[str, Any], str]:
    start, stop = "<code>", "</code>"
    i = content.find(start)
    if i == -1:
        return None, content
    j = content.find(stop)
    assert j > i

    code = content[i + len(start) : j]
    matches = code_pattern.findall(code)
    if matches:
        code = matches[0].strip()

    message = {
        "role": "assistant",
        "content": content[:i].strip(),
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "code_interpreter",
                    "arguments": {"code": code},
                },
            },
        ],
    }
    return message, content[j + len(stop) :]


def extract_answer_message(content: str) -> tuple[dict[str, Any], str]:
    start, stop = "<answer>", "</answer>"
    i = content.find(start)
    if i == -1:
        return None, content
    j = content.find(stop)
    assert j > i

    answer = content[:i] + content[i + len(start) : j]
    message = {
        "role": "assistant",
        "content": answer.strip(),
    }
    return message, content[j + len(stop) :]


def extract_interpreter_message(content: str) -> tuple[dict[str, Any], str]:
    start, stop = "<interpreter>", "</interpreter>"
    i = content.find(start)
    if i == -1:
        return None, content
    j = content.find(stop)
    assert j > i

    interpreter = content[i + len(start) : j]
    message = {
        "role": "tool",
        "content": interpreter.strip(),
    }
    return message, content[j + len(stop) :]


def process(row: dict, *, tools: str):
    messages = []

    # extract problem
    content = row["messages"][0]["content"]
    start = "*user question:*"
    i = content.find(start)
    assert i != -1
    prompt = content[i + len(start) :].replace("<answer>", "").replace("</answer>", "").strip()
    messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    # extract multi turns
    content = row["messages"][1]["content"]
    role = "assistant"
    while len(content) > 0:
        if role == "assistant":
            message, content = extract_code_message(content)
            if message is None:
                message, content = extract_answer_message(content)
            assert message is not None
            messages.append(message)
            role = "tool"
        else:
            message, content = extract_interpreter_message(content)
            assert message is not None
            messages.append(message)
            role = "assistant"

    tools = json.loads(tools)
    return {"messages": messages, "tools": tools}


if __name__ == "__main__":
    tools_config_file = "recipe/retool/sandbox_fusion_tool_config.yaml"
    tools_config = OmegaConf.load(tools_config_file)
    tool_schema = OmegaConf.to_container(tools_config["tools"][0]["tool_schema"])
    tools = json.dumps([tool_schema])

    data = datasets.load_dataset("JoeYing/ReTool-SFT")["train"]
    data = data.map(process, fn_kwargs={"tools": tools})
    save_path = os.path.expanduser("~/ReTool-SFT/data/train-00000-of-00001.parquet")
    data.to_parquet(save_path)
