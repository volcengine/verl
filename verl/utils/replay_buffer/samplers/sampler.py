# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

from typing import Iterator

"""
plugin based sampler design
"""


class Sampler:
    # TODO: have sample_keys as arguments
    def __init__(self, sample_keys=None) -> None:
        # each sample key has its own index

        # sample_keys is a set of attributes where sampler might slice and dice on
        # if we sample on a non specified sampling attribute, sampler will become extremely slow.
        # TODO: dedup the index if multiple samplers share the same attributes to be indexed
        sample_keys = sample_keys or {}
        self._index = {attribute: {} for attribute in sample_keys}

    def build_index(self, key, value):
        # build an index table which maps:
        # indexed key to key.
        pass

    def bind_task_processor(self, task_processor):
        # To merge with pending tasks in p0 queue during sampling
        self._task_processor = task_processor

    def get_index(self, index_name):
        return self._index[index_name]

    def remove_index(self, key):
        for val in self._index.values():
            val.pop(key, None)

    def sample(self) -> Iterator:
        raise NotImplementedError
