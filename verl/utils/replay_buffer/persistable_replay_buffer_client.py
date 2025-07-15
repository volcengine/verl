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

import logging
import os
import pickle

import rocksdbpy

from verl.utils.replay_buffer.persistable_replay_buffer_util.lru_cache import LRUCache
from verl.utils.replay_buffer.persistable_replay_buffer_util.util import to_bytes
from verl.utils.replay_buffer.replay_buffer_client import ReplayBufferClient
from verl.utils.replay_buffer.task_processor import Task, TaskProcessor, TaskType

logger = logging.getLogger(__name__)

# TODO:
# 1. The underlying db is rocksdb,
# where there's no hard limit on the value size per se, but large value brings memory pressure
# thus limit the number of parallism of push or get. So we have to solve it in the long run.
# given each push would load the entire list of value into mem.
#   Some approaches:
#     1.1. if the list of value is too large, let's parition it.
#     1.2. if multiple pushs on the same keys are happening, let's do a batch push, instead of
#          pusing it one by one
# 2. push could be slow, so we should allow user to do push asynchronously.
"""
A thread safe LRU cache used to store replay buffer data.
"""


def get_key_index_name():
    return f"{PersistableReplayBufferClient.MAGIC_SUFFIX}_key"


class PersistableReplayBufferClient(ReplayBufferClient):
    MAGIC_SUFFIX = "replay_buffer_db_1735829"

    def __init__(self, replay_buffer_name, cache_size_limit_in_mb, restore_from_hdfs_path=None, samplers=None):
        """
        cache_size_limit_in_mb: the replay buffer in-mem cache size. The cache is used to serve hot data.
        restore_from_hdfs_path: where the replay buffer data will be backed up to
        """
        self._db_path = os.path.join(PersistableReplayBufferClient.MAGIC_SUFFIX, replay_buffer_name)
        self._samplers = samplers if samplers else []
        if restore_from_hdfs_path is not None:
            # Restore replay buffer from hdfs to self.dp_path if restore_from_hdfs_path is not None. Start backup thread
            from verl.utils.replay_buffer.persistable_replay_buffer_util.hdfs_backup_manager import HDFSBackupManager

            self._backup_manager = HDFSBackupManager(replay_buffer_name, self._db_path, restore_from_hdfs_path)
        else:
            self._backup_manager = None

        self._db = self._initialize_db()
        self._cache = LRUCache(cache_size_limit_in_mb, self._db)
        self._task_processor = TaskProcessor(self._db, self._cache, self._backup_manager, self._db_path, self._samplers)

        self._cache.bind_task_processor(self._task_processor)
        for sampler in self._samplers:
            sampler.bind_task_processor(self._task_processor)

        if self._backup_manager is not None:
            cache_state = self._backup_manager.get_restored_cache()
            if cache_state:  # initially none
                self._cache.restore_state(cache_state)
            # pass the initalized db and cache to _backup_manager
            self._backup_manager.bind_db_cache_processor(self._db, self._cache, self._task_processor)
            self._backup_manager.run()

    def _build_index(self, key, value):
        # build an index table which maps:
        # indexed key to key.
        self._index[get_key_index_name()][key] = key

    def _remove_index(self, key):
        # build an index table which maps:
        # indexed key to key.
        if key in self._index[get_key_index_name()]:
            self._index[get_key_index_name()].pop(key)

    def get_index(self, index_name):
        return self._index[index_name]

    def _initialize_db(self):
        """Initialize RocksDB with thread-safe options."""
        opts = rocksdbpy.Option()
        opts.create_if_missing(True)
        # opts.max_background_jobs = 16  # Optimize for concurrent operations
        return rocksdbpy.open(self._db_path, opts)

    def push(self, key, new_value):
        """Append new elements to the existing list value for a key."""
        if not isinstance(new_value, list):
            new_value = [new_value]

        push_task = Task(TaskType.PUSH, key, new_value)
        self._task_processor.add_task(push_task)

    def get(self, key):
        """Retrieve a value from replay buffer with locking."""
        self._task_processor.acquire_q_lock()
        base_value = self._cache.get(key)
        if not base_value:
            # not in cache, check in rocksdb
            key_encoded = to_bytes(key)
            existing_value = self._db.get(key_encoded)
            if existing_value:
                base_value = pickle.loads(existing_value)
                # Add the populate task to the task_processor
                populate_task = Task(TaskType.POPULATE, key)
                self._task_processor.add_task(populate_task)
        # Merge existing value with pending tasks in p0_q. value can be None here.
        merged_value = self._task_processor.merge_pending_tasks(key, base_value)
        self._task_processor.release_q_lock()
        return merged_value

    def delete(self, key):
        delete_task = Task(TaskType.DELETE, key)
        self._task_processor.add_task(delete_task)

    def close(self):
        if hasattr(self, "_task_processor"):
            self._task_processor.shutdown()

    def sample(self):
        pass
