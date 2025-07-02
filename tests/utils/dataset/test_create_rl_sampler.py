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
"""
test create_rl_sampler
"""

import pytest
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from verl.trainer.main_ppo import create_rl_sampler


class FakeIncorrectSampler:
    def __init__(self, data_source, data_config):
        pass


class FakeChatDataset(Dataset):
    def __init__(self):
        self.data = [
            {"prompt": "What's your name?", "response": "My name is Assistant."},
            {"prompt": "How are you?", "response": "I'm doing well, thank you."},
            {"prompt": "What is the capital of France?", "response": "Paris."},
            {
                "prompt": "Tell me a joke.",
                "response": "Why did the chicken cross the road? To get to the other side!",
            },
            {"prompt": "What is 2+2?", "response": "4"},
        ]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def test_create_custom_curriculum_samper():
    data_config = OmegaConf.create(
        {
            "curriculum": {
                "curriculum_class_path": "verl.utils.dataset.curriculum_sampler",
                "curriculum_class": "RandomCurriculumSampler",
            }
        }
    )

    dataset = FakeChatDataset()

    # doesn't raise
    create_rl_sampler(data_config, dataset)


def test_create_custom_curriculum_samper_wrong_class():
    data_config = OmegaConf.create(
        {
            "curriculum": {
                "curriculum_class_path": "tests.utils.dataset.test_create_rl_sampler",
                "curriculum_class": "FakeIncorrectSampler",
            }
        }
    )

    dataset = FakeChatDataset()

    # FakeIncorrectSampler is not an instance of AbstractCurriculumSampler, so raises
    with pytest.raises(AssertionError):
        create_rl_sampler(data_config, dataset)
