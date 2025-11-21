# Copyright 2025 Individual Contributor: linxxx3 (linxxx3@gmail.com)
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

from abc import ABC, abstractmethod
from typing import Any


class AgentClientBase(ABC):
    """Agent client base class."""

    def __init__(self, server_address: str, **kwargs):
        if server_address.startswith("http"):
            self.server_address_full = server_address
        else:
            self.server_address_full = f"http://{server_address}"

    @abstractmethod
    async def chat(self, trace_id: str, sampling_params: dict[str, Any], **kwargs) -> Any:
        """Custom chat function.
        Note: use async http client like aiohttp in this function, to avoid blocking the event loop.
        Args:
            trace_id: trace id for collecting the trajectory
            sampling_params: sampling parameters, e.g., temperature, top_p, max_tokens, etc.
            **kwargs: non-tensor fields of a data sample from RLHFDataset
        """
        raise NotImplementedError
