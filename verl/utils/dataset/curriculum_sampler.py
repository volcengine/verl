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

import torch
from omegaconf import DictConfig
from torch.utils.data import RandomSampler, Sampler


class AbstractCurriculumSampler(Sampler[int]):
    @abstractmethod
    def __init__(
        self,
        data_source: Sized,
        config: DictConfig,
    ):
        pass


class RandomCurriculumSampler(AbstractCurriculumSampler):
    def __init__(
        self,
        data_source: Sized,
        data_config: DictConfig,
    ):
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(1)
        sampler = RandomSampler(data_source=data_source)
        self.sampler = sampler

    def __iter__(self):
        return self.sampler.__iter__()

    def __len__(self) -> int:
        return len(self.sampler)
