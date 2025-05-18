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
"""Abstractions for implementing external tools used during RL training."""

from typing import Any, Optional, Tuple
from uuid import uuid4

from .schemas import OpenAIFunctionToolSchema


class BaseTool:
    """Base class for RL tools.

    A tool encapsulates any external functionality or environment interaction
    that can be invoked during training. Subclasses are expected to override the
    lifecycle methods below to implement concrete behavior.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """Initialize a new tool.

        Args:
            config: Arbitrary configuration for the tool.
            tool_schema: Schema describing the tool in OpenAI format.
        """

        self.config = config
        self.name = tool_schema.function.name
        self.tool_schema = tool_schema

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return the OpenAI schema for this tool."""

        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        """Create a new tool instance.

        A unique identifier is returned and can be used in subsequent calls to
        :meth:`execute`, :meth:`calc_reward` and :meth:`release`.

        Args:
            instance_id: Optional identifier to reuse. When ``None`` a random
                UUID is generated.

        Returns:
            The identifier associated with the created instance.
        """
        if instance_id is None:
            return str(uuid4())
        else:
            return instance_id

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        """Execute the tool with the given parameters.

        Args:
            instance_id: Identifier returned by :meth:`create`.
            parameters: Parameters for the tool. The mapping must conform to the
                schema returned by :meth:`get_openai_tool_schema`.

        Returns:
            A tuple ``(response, reward, metrics)`` where ``response`` is the
            textual output of the tool, ``reward`` is an immediate reward score
            and ``metrics`` is an arbitrary dictionary containing monitoring
            information.
        """
        return "Updated the tool state.", 0.0, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Compute the accumulated reward for the given instance."""
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        """Free resources associated with a tool instance."""
        pass
