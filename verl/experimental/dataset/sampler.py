# Copyright 2025 Amazon.com Inc and/or its affiliates
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
from abc import abstractmethod
from collections.abc import Sized

from omegaconf import DictConfig
from torch.utils.data import Sampler

from verl import DataProto


class AbstractSampler(Sampler[int]):
    """Abstract interface for custom samplers."""

    @abstractmethod
    def __init__(
        self,
        data_source: Sized,
        data_config: DictConfig,
    ):
        pass


class AbstractCurriculumSampler(AbstractSampler):
    """Experimental interface for curriculum learning samplers."""

    @abstractmethod
    def update(self, batch: DataProto) -> None:
        pass
