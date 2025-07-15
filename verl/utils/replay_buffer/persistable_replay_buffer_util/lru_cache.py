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

import pickle
import threading
from collections import OrderedDict
from queue import Queue

from pympler import asizeof

from verl import DataProto
from verl.utils.replay_buffer.task_processor import Task, TaskType


def get_batch_bytes(batch):
    """
    Return the bytes of a single DataProto. Referenced DataProto's print_size method.
    Doesn’t take into account batch.meta_info.
    """
    size_of_tensordict = 0
    if batch.batch is not None:
        for key, tensor in batch.batch.items():
            size_of_tensordict += tensor.element_size() * tensor.numel()
    size_of_numpy_array = 0
    for key, numpy_array in batch.non_tensor_batch.items():
        size_of_numpy_array += numpy_array.nbytes
    return size_of_tensordict + size_of_numpy_array


def get_batch_list_bytes(batch_list):
    """
    Sum up the bytes of the DataProtos or other objects in the batch list.
    """
    return sum(
        get_batch_bytes(batch) if isinstance(batch, DataProto) else asizeof.asizeof(batch) for batch in batch_list
    )


class LRUCacheEvictionManager:
    """
    Calculates the size in bytes of the LRUCache and append eviction task if it exceeds limit. This process happens
    in a background thread to not block the other threads.
    """

    def __init__(self, cache_size_limit_in_mb, data, db):
        self._cache_size_in_bytes = cache_size_limit_in_mb * 1024 * 1024  # NOTE: Occasionally may exceed
        self._size_in_bytes = 0
        self._work_queue = Queue()  # A queue of new values inserted/deleted for the eviction thread
        # to process. Stores {new value, whether inserted/deleted}
        self._eviction_thread = threading.Thread(target=self._eviction_worker, daemon=True)
        self._eviction_thread.start()
        self._data = data
        self._db = db

    def bind_task_processor(self, task_processor):
        self._task_processor = task_processor

    def update_work_queue(self, value, inserted):
        self._work_queue.put((value, inserted))

    def get_size_in_bytes(self):
        return self._size_in_bytes

    def get_size_limit(self):
        return self._cache_size_in_bytes

    def _eviction_worker(self):
        while True:
            curr = self._work_queue.get()  # python queue: blocks until an item arrives
            value = curr[0]
            was_inserted = curr[1]  # True if inserted, false if deleted

            value_bytes = get_batch_list_bytes(value)
            if not was_inserted:
                value_bytes *= -1

            self._size_in_bytes += value_bytes
            if was_inserted and self._size_in_bytes > self._cache_size_in_bytes:
                eviction_task = Task(TaskType.EVICT)
                # TODO: If every push that exceeds limit just initiates one eviction task, then if this
                # push is very large and the value_lru at the execution time is very small, then after eviction, the
                # cache still exceeds limit by a lot. So need to decide how many items to evict. Maybe every time after
                # eviction task is executed and after size calculation is done, still check whether the size exceeds
                # limit. If so, append another eviction task. During eviction execution, get the current size (since
                # delete could have already occurred, making eviction unnecessary), if still exceeds limit, execute.
                # For now, assume that the batch lists are about average in size.
                self._task_processor.add_task(eviction_task)

    @staticmethod
    def load_extend_dump(existing_list, new_value):
        if existing_list is None:
            existing_value_deser = []
        else:
            existing_value_deser = pickle.loads(existing_list)
        existing_value_deser.extend(new_value)
        return pickle.dumps(existing_value_deser)


class LRUCache:
    # TODO: an adaptive eviction policy based on actual memory usage
    def __init__(self, cache_size_limit_in_mb, db):
        # LRU Cache -> Most recently accessed on the rightmost, least recently accessed on the leftmost
        # It computes the number of bytes and conducts eviction in an async thread, not blocking push
        self._data = OrderedDict()
        # self._dict_lock = threading.Lock()
        self._eviction_manager = LRUCacheEvictionManager(cache_size_limit_in_mb, self._data, db)

    def bind_task_processor(self, task_processor):
        self._eviction_manager.bind_task_processor(task_processor)

    # restore the dict and size_in_bytes from hdfs. Called during initialization
    def restore_state(self, cache_state):
        self._data = cache_state["dict"]
        self._eviction_manager._size_in_bytes = cache_state["size_in_bytes"]

    def get(self, key):
        # with self._dict_lock:
        to_return = self._data.get(key, None)
        if to_return is not None:
            self._data.move_to_end(key, last=True)  # move the newly added key, value to the rightmost
        return to_return

    def size_exceeds_limit(self):
        return self._eviction_manager.get_size_in_bytes() > self._eviction_manager.get_size_limit()

    def push(self, key, new_value):
        self._eviction_manager.update_work_queue(new_value, True)
        # with self._dict_lock:
        existing_value = self._data.get(key, [])
        existing_value.extend(new_value)
        self._data.update({key: existing_value})
        self._data.move_to_end(key, last=True)  # move the newly added key, value to the rightmost

    def get_keys(self):
        # with self._dict_lock:
        return set(self._data.keys())  # self._dict.keys() is a view object

    def delete(self, key):
        # with self._dict_lock:
        popped = self._data.pop(key, None)
        if popped is not None:
            self._eviction_manager.update_work_queue(popped, False)
