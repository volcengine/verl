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
# Unit Tests for `initialize_tools_from_config`
import json
import os
from typing import Any, Tuple

import pytest
from transformers.utils import get_json_schema

from verl.tools.base_tool import BaseTool, OpenAIFunctionToolSchema
from verl.tools.utils.tool_registry import initialize_tools_from_config


class WeatherToolForTest(BaseTool):
    def get_current_temperature(self, location: str, unit: str = "celsius"):
        """Get current temperature at a location.

        Args:
            location: The location to get the temperature for, in the format "City, State, Country".
            unit: The unit to return the temperature in. Defaults to "celsius". (choices: ["celsius", "fahrenheit"])

        Returns:
            the temperature, the location, and the unit in a dict
        """
        return {
            "temperature": 26.1,
            "location": location,
            "unit": unit,
        }

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        schema = get_json_schema(self.get_current_temperature)
        return OpenAIFunctionToolSchema(**schema)

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        try:
            result = self.get_current_temperature(**parameters)
            return json.dumps(result), 0, {}
        except Exception as e:
            return str(e), 0, {}


class WeatherToolWithDataForTest(BaseTool):
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        schema = get_json_schema(self.get_temperature_date)
        return OpenAIFunctionToolSchema(**schema)

    def get_temperature_date(self, location: str, date: str, unit: str = "celsius"):
        """Get temperature at a location and date.

        Args:
            location: The location to get the temperature for, in the format "City, State, Country".
            date: The date to get the temperature for, in the format "Year-Month-Day".
            unit: The unit to return the temperature in. Defaults to "celsius". (choices: ["celsius", "fahrenheit"])

        Returns:
            the temperature, the location, the date and the unit in a dict
        """
        return {
            "temperature": 25.9,
            "location": location,
            "date": date,
            "unit": unit,
        }

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        try:
            result = self.get_temperature_date(**parameters)
            return json.dumps(result), 0, {}
        except Exception as e:
            return str(e), 0, {}


@pytest.fixture
def create_local_tool_config():
    tool_config = {
        "tools": [
            {
                "class_name": "tests.tools.test_base_tool_on_cpu.WeatherToolForTest",
                "config": {"type": "native"},
            },
            {
                "class_name": "tests.tools.test_base_tool_on_cpu.WeatherToolWithDataForTest",
                "config": {"type": "native"},
            },
        ]
    }
    tool_config_path = "/tmp/tool_config.json"
    with open(tool_config_path, "w") as f:
        json.dump(tool_config, f)
    yield tool_config_path
    if os.path.exists(tool_config_path):
        os.remove(tool_config_path)


@pytest.fixture
def create_fake_tool_config():
    tool_config = {
        "tools": [
            {
                "class_name": "tests.workers.rollout.fake_path.test_vllm_chat_scheduler.WeatherTool",
                "config": {"type": "native"},
            },
            {
                "class_name": "tests.workers.rollout.fake_path.test_vllm_chat_scheduler.WeatherToolWithData",
                "config": {"type": "native"},
            },
        ]
    }
    tool_config_path = "/tmp/tool_config.json"
    with open(tool_config_path, "w") as f:
        json.dump(tool_config, f)
    yield tool_config_path
    if os.path.exists(tool_config_path):
        os.remove(tool_config_path)


def test_initialize_tools_from_fake_config(create_fake_tool_config):
    tool_config_path = create_fake_tool_config

    # Use pytest.raises to check if an exception is raised when calling initialize_tools_from_config.
    # Since the tool configuration uses fake paths, an exception is expected during the tool initialization process.
    with pytest.raises(ModuleNotFoundError):
        _ = initialize_tools_from_config(tool_config_path)


def test_initialize_tools_from_local_config(create_local_tool_config):
    """
    Test the `initialize_tools_from_config` function using a local tool configuration.
    This test verifies that the function can correctly initialize tools based on a local configuration file.

    Args:
        create_local_tool_config: A pytest fixture that creates a local tool configuration file
                                  and returns its path. After the test is completed, the fixture
                                  will clean up the configuration file.
    """
    # Retrieve the path of the local tool configuration file generated by the fixture
    tool_config_path = create_local_tool_config

    tools = initialize_tools_from_config(tool_config_path)

    assert len(tools) == 2
    from tests.tools.test_base_tool_on_cpu import WeatherToolForTest, WeatherToolWithDataForTest

    assert isinstance(tools[0], WeatherToolForTest)
    assert isinstance(tools[1], WeatherToolWithDataForTest)
    assert tools[0].config == {"type": "native"}
    assert tools[1].config == {"type": "native"}
