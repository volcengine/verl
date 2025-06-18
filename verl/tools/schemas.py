# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class OpenAIFunctionPropertySchema(BaseModel):
    """The schema of a parameter in OpenAI format."""

    type: str
    description: str | None = None
    enum: list[str] | None = None


class OpenAIFunctionParametersSchema(BaseModel):
    """The schema of parameters in OpenAI format."""

    type: str
    properties: dict[str, OpenAIFunctionPropertySchema]
    required: list[str]


class OpenAIFunctionSchema(BaseModel):
    """The schema of a function in OpenAI format."""

    name: str
    description: str
    parameters: OpenAIFunctionParametersSchema
    strict: bool = False


class OpenAIFunctionToolSchema(BaseModel):
    """The schema of a tool in OpenAI format."""

    type: str
    function: OpenAIFunctionSchema


class OpenAIFunctionParsedSchema(BaseModel):
    """The parsed schema of a tool in OpenAI format."""

    name: str
    arguments: str  # JSON string


class OpenAIFunctionCallSchema(BaseModel):
    """The parsed schema of a tool in OpenAI format."""

    name: str
    arguments: dict[str, Any]
    has_decode_error: bool = Field(default=False, init=False, exclude=True)

    @model_validator(mode="wrap")
    @classmethod
    def parse_arguments(cls, values, handler):
        """Parse arguments if it is a JSON string and set decode error flag."""
        has_decode_error = False

        if isinstance(values, dict) and "arguments" in values:
            arguments = values["arguments"]
            if isinstance(arguments, str):
                try:
                    parsed = json.loads(arguments)
                    if not isinstance(parsed, dict):
                        parsed = {}
                        has_decode_error = True
                    values["arguments"] = parsed
                except json.JSONDecodeError:
                    values["arguments"] = {}
                    has_decode_error = True

        instance = handler(values)
        object.__setattr__(instance, "has_decode_error", has_decode_error)
        return instance


class OpenAIFunctionToolCall(BaseModel):
    """The tool call in OpenAI format."""

    id: str
    type: Literal["function"] = "function"
    function: OpenAIFunctionCallSchema
