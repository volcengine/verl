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

from verl.experimental.replay_buffer.replay_buffer_client import ReplayBufferClient

"""
The replay buffer backed by an in-mem dict.
It's NOT thread safe.
"""


class VanillaReplayBufferClient(ReplayBufferClient):
    def __init__(self):
        self.__pool = dict()

    def push(self, key, batches):
        if not isinstance(batches, list):
            batches = [batches]
        if key not in self.__pool:
            self.__pool[key] = []
        for batch in batches:
            self.__pool[key].append(batch)

    def get(self, key: str):
        return self.__pool.get(key, None)

    def sample(self):
        keys = list(self.__pool.keys())
        while keys:  # raises StopIteration on every next() if keys is empty
            random_key = random.choice(keys)
            yield random_key

    def delete(self, key: str):
        self.__pool.pop(key)
