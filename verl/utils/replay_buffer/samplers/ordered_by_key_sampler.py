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


class OrderedByKeySampler(Sampler):
    from enum import Enum

    class OrderType(Enum):
        Descending = "descending"
        Ascending = "ascending"

    def __init__(self, p: float, order_type: OrderType) -> None:
        super().__init__({get_key_index_name()})
        self._order_type = order_type
        self._p = p
        self._sampled_indices = set()

    def build_index(self, key, value):
        # build an index table which maps:
        # indexed key to key.
        self._index[get_key_index_name()][key] = key

    def sample(self, num_samples=None) -> Iterator:
        """
        p % 采用时间戳最新的样本，(1 - p) % 从剩余样本中均匀采样
        """
        self._task_processor.acquire_q_lock()
        index_map = self.get_index(get_key_index_name())
        indexed_keys = set(index_map.keys())
        # merge with pending tasks in p0 queue to get updates for all keys
        indexed_keys = list(self._task_processor.merge_pending_keys(indexed_keys))
        self._task_processor.release_q_lock()

        if num_samples is not None:
            ordered_count = int(self._p * num_samples)
        else:
            ordered_count = int(self._p * len(indexed_keys))

        # Ordered sampling
        # Assume the keys are either int or string with format "rollout_id"
        sorted_keys = sorted(
            indexed_keys,
            key=lambda idx: idx if isinstance(idx, int) else int(idx.split("_")[-1]),
            reverse=self._order_type == self.OrderType.Descending,
        )
        for index in sorted_keys[:ordered_count]:
            yield index
            self._sampled_indices.add(index)

        # Remaining count for uniform sampling
        remaining_count = len(indexed_keys) - ordered_count
        remaining_indices = list(set(indexed_keys) - self._sampled_indices)

        if remaining_count > 0 and remaining_indices:
            uniform_samples = random.sample(remaining_indices, min(remaining_count, len(remaining_indices)))
            for index in uniform_samples:
                yield index
                self._sampled_indices.add(index)
