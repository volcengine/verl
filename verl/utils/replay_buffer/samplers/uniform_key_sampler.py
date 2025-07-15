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

import random
from typing import Iterator

from verl.utils.replay_buffer.persistable_replay_buffer_client import get_key_index_name
from verl.utils.replay_buffer.samplers.sampler import Sampler


class UniformKeySampler(Sampler):
    def __init__(self) -> None:
        super().__init__({get_key_index_name()})

    def build_index(self, key, value):
        # build an index table which maps:
        # indexed key to key.
        self._index[get_key_index_name()][key] = key

    def sample(self) -> Iterator:
        self._task_processor.acquire_q_lock()

        key_index = self._index[get_key_index_name()]
        keys = set(key_index.keys())
        # merge with pending tasks in p0 queue to get updates for all keys
        keys = list(self._task_processor.merge_pending_keys(keys))
        self._task_processor.release_q_lock()
        while keys:  # raises StopIteration on every next() if keys is empty
            random_key = random.choice(keys)
            yield random_key
