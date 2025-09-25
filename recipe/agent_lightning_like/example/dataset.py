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

from verl.utils.dataset import RLHFDataset


class CustomDataset(RLHFDataset):
    """A custom dataset for the agent-lightning-like example."""

    def __getitem__(self, item):
        row_dict = super().__getitem__(item)
        row_dict["agent_name"] = "lightning_demo"  # must match the name in agent_loop.yaml
        row_dict.pop("tools_kwargs", None)  # remove tools_kwargs if exists, tools defined in agent server side
        return row_dict
