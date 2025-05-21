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
from typing import Any, Optional, Tuple
from uuid import uuid4

from .schemas import OpenAIFunctionToolSchema


class BaseFeedback:
    """Base class for feedbacks.

    A feedback should support the following methods:

    - `to_openai_function_tool_schema`: return the tool schema in OpenAI format.
    - `create`: create a feedback instance for a trajectory.
    - `get_feedback`: get the feedback of the user.
    - `release`: release the feedback instance.
    """

    def __init__(self, config: dict):
        self.config = config
        self.name = config.get("name", "feedback")

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        """Create a feedback instance.

        Args:
            instance_id: The instance id of the feedback.

        Returns:
            The instance id of the feedback.
        """
        if instance_id is None:
            return str(uuid4())
        else:
            return instance_id

    async def get_feedback(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        """Get the feedback of the user.

        Args:
            instance_id: The instance id of the feedback.
            parameters: The json string of the parameters of the feedback.

        Returns: feedback_response, feedback_reward_score, feedback_metrics
            feedback_response: The response str of the feedback.
            feedback_reward_score: The step reward score of the feedback.
            feedback_metrics: The metrics of the feedback.
        """
        return "Updated the feedback state.", 0.0, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate the reward of the feedback.

        Args:
            instance_id: The instance id of the feedback.
        """
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the feedback.

        Args:
            instance_id: The instance id of the feedback.
        """
        pass
