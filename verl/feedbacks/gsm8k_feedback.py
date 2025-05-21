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

import logging
import os
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from verl.utils.reward_score import gsm8k

from .base import BaseFeedback

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class Gsm8kFeedback(BaseFeedback):
    """A demo feedback for calculating the reward of gsm8k.

    - `create`: create a feedback instance for a trajectory.
    - `get_feedback`: get the feedback of the user.
    - `release`: release the feedback instance.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._instance_dict = {}

    async def create(self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": ground_truth,
            "reward": 0.0,
        }
        return instance_id

    async def get_feedback(self, instance_id: str, messages: List[Dict[str, Any]], **kwargs) -> Tuple[str, float, dict]:
        content = ''
        for i in range(len(messages) - 1, -1, -1):
            item = messages[i]
            if item.get('role') == 'user':
                content = item.get('content')
                break

        if content.startswith("#### "):
            self._instance_dict[instance_id]["response"] = content
        else:
            self._instance_dict[instance_id]["response"] = "#### " + content

        reward = await self.calc_reward(instance_id)
        if reward == 1.0:
            feedback = "Your response is correct!"
            go_on = False
        else:
            feedback = "Your response is incorrect! You need to reflect on your answer and try again."
            go_on = True

        return f"{feedback=} {reward=}", go_on, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return gsm8k.compute_score(
            self._instance_dict[instance_id]["response"],
            self._instance_dict[instance_id]["ground_truth"],
            method="flexible",
            format_score=0.0,
            score=1.0,
        )

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
