# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
from typing import Any, Optional
from uuid import uuid4


class BaseInteraction:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.name: str = config.get("name", "interaction_agent")  # More general agent default role name

    async def start_interaction(self, instance_id: Optional[str] = None, **kwargs) -> str:
        """Create a tool instance.

        Args:
            instance_id: The instance id of the tool.

        Returns:
            The instance id of the tool.
        """
        if instance_id is None:
            return str(uuid4())
        else:
            return instance_id

    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], **kwargs
    ) -> tuple[bool, str, float, dict[str, Any]]: # More clear response generation method
        """
        Generates a response for the current turn of interaction.
        Returns a tuple containing:
        - should_terminate_sequence (bool): True if the interaction sequence should end.
        - response_content (str): The textual content of the response.
        - current_turn_score (float): The score for this specific turn/response.
        - additional_data (dict): Any extra information or metadata.
        """
        should_terminate_sequence: bool = False  # if True, end rollout
        response_content: str = "Your current result seems acceptable."
        current_turn_score: float = 0.8
        additional_data: dict[str, Any] = {}
        return should_terminate_sequence, response_content, current_turn_score, additional_data

    async def calculate_score(self) -> float:  # More clear score calculation method
        """
        Calculates a score for the interaction,
        potentially considering aspects like partial exposure & in-context task switching.
        should be invoke at turn-level
        """
        # ...implement the logic to calculate turn-level score...
        score = 0.0
        return score

    async def finalize_interaction(self) -> None:  # More clear interaction end and resource release method
        """
        Finalizes the interaction session and releases any associated state or resources.
        Simulates: release state
        """
        # ...implement the logic to release state...
        pass
