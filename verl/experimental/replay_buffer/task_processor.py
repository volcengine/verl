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

import concurrent.futures
import logging
import os
import pickle
import queue
import random
import shutil
import threading
from enum import Enum

from verl.experimental.replay_buffer.persistable_replay_buffer_util.util import to_bytes

logger = logging.getLogger(__name__)


class TaskType(Enum):
    PUSH = "push"
    DELETE = "delete"
    POPULATE = "populate"
    EVICT = "evict"
    SNAPSHOT = "snapshot"


class Task:
    def __init__(self, type: TaskType, key=None, value=None):
        self.type = type
        self.key = key
        self.value = value

    def set_timestamp(self, timestamp=None):
        self.timestamp = timestamp


class TaskProcessor:
    """
    Executes tasks in replay buffer's p0 and p1 task queue. Determine which task queue to execute based on
    random number generator, with p0 having the priority (80% choose p0, 20% choose p1).
    """

    def __init__(self, db, cache, backup_manager, db_path, samplers):
        self._p0_q = queue.Queue()  # for PUSH / DELETE / SNAPSHOT
        self._p1_q = queue.Queue()  # for POPULATE / EVICT
        self._p0_index = dict()  # stores {key -> pending list of PUSH/DELTE TASK in p0_q}
        self._p0_lock = threading.Lock()  # locks the processing of p0 queue during get/sample.
        self._db = db
        self._cache = cache
        self._backup_manager = backup_manager  # can be None
        self._db_path = db_path
        self._samplers = samplers

        self._task_available = threading.Event()  # whether p0 or p1 queue has tasks
        self._shutdown = threading.Event()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._executor.submit(self._process_tasks)

    def shutdown(self):
        """Shut down the executor"""
        self._shutdown.set()
        self._task_available.set()
        self._executor.shutdown(wait=True, cancel_futures=False)

    def acquire_q_lock(self):
        self._p0_lock.acquire()

    def release_q_lock(self):
        self._p0_lock.release()

    def _process_tasks(self):
        while not self._shutdown.is_set():
            if self._p0_q.empty() and self._p1_q.empty():
                self._task_available.clear()
                self._task_available.wait(timeout=0.5)  # wait for a task to be added
                continue
            if not self._p0_q.empty() and self._p1_q.empty():
                task_batch = self._get_task_batch(self._p0_q)
                self._execute_p0_q(task_batch)
            elif not self._p1_q.empty() and self._p0_q.empty():
                task_batch = self._get_task_batch(self._p1_q)
                self._execute_p1_q(task_batch)
            elif not self._p0_q.empty() and not self._p1_q.empty():  # both not empty
                if random.random() > 0.2:
                    task_batch = self._get_task_batch(self._p0_q)
                    self._execute_p0_q(task_batch)
                else:
                    task_batch = self._get_task_batch(self._p1_q)
                    self._execute_p1_q(task_batch)

    def merge_pending_tasks(self, key, base_value):
        """
        Called in the get method. Merge the already existing value with unprocessed p0 queue tasks.
        """
        merged_value = base_value  # can be None
        if merged_value is None:
            merged_value = []

        pending_tasks = self._p0_index.get(key, [])
        for pending_task in pending_tasks:
            if pending_task.type == TaskType.PUSH:
                merged_value.extend(pending_task.value)
            elif pending_task.type == TaskType.DELETE:
                merged_value.clear()
        return merged_value if merged_value else None

    def merge_pending_keys(self, base_keys):
        """
        Called in the sample method. Merge the already existing keys with keys in unprocessed p0 queue tasks.
        """
        merged_keys = base_keys
        if merged_keys is None:
            merged_keys = set()
        # Get the final operation for each key -> whether push or delete.
        for key, pending_tasks in self._p0_index.items():
            final_operation = pending_tasks[-1]
            if final_operation.type == TaskType.PUSH:
                merged_keys.add(final_operation.key)
            elif final_operation.type == TaskType.DELETE and final_operation.key in merged_keys:
                merged_keys.remove(final_operation.key)
        return merged_keys

    def add_task(self, task):
        """
        Add the task to p0 or p1 queue based on the task type
        """
        if task.type == TaskType.PUSH or task.type == TaskType.DELETE or task.type == TaskType.SNAPSHOT:
            self._p0_q.put(task)
            # for push/delete, record the task in p0_index.
            if task.type != TaskType.SNAPSHOT:
                if task.key not in self._p0_index:
                    self._p0_index[task.key] = []
                self._p0_index[task.key].append(task)
        else:
            self._p1_q.put(task)
        self._task_available.set()

    def _execute_p0_q(self, task_batch):
        """
        Execute a batch of tasks in p0_q at a time
        """
        net_operations = {}  # maps each key to a list of its "net" push/delete operations after cancellation
        last_snapshot_idx = None  # the index of last snapshot in task_batch list
        for idx, task in enumerate(task_batch):
            if task.type == TaskType.SNAPSHOT:
                last_snapshot_idx = idx
                continue
            # for push/delete tasks
            task.set_timestamp(idx)
            if task.type == TaskType.DELETE:
                net_operations[task.key] = [task]  # Cancel all previous delete/push operations
            elif task.type == TaskType.PUSH:
                if task.key not in net_operations:
                    net_operations[task.key] = []
                net_operations[task.key].append(task)

        # edge case: if snapshot task is the first or there is only snapshot task, just execute it.
        if last_snapshot_idx == 0:
            self._take_snapshot()

        for key, net_ops in net_operations.items():
            for task in net_ops:
                if (
                    last_snapshot_idx and task.timestamp + 1 >= last_snapshot_idx
                ):  # every task is either push/delete updated above, so must have timestamp
                    self._take_snapshot()
                self.acquire_q_lock()
                if task.type == TaskType.PUSH:
                    self._execute_push_operation(key, task.value)
                elif task.type == TaskType.DELETE:
                    self._execute_delete_operation(key)
                self._p0_index[key].remove(task)
                if not self._p0_index[key]:  # empty
                    self._p0_index.pop(key)
                self.release_q_lock()

    def _execute_p1_q(self, task_batch):
        """
        Execute a batch of tasks in p1_q at a time
        """
        for task in task_batch:
            if task.type == TaskType.EVICT:
                self._execute_evict_operation()
            elif task.type == TaskType.POPULATE:
                self._execute_populate_operation(task.key)

    def _execute_push_operation(self, key, value):
        self._cache.push(key, value)  # lru eviction manager does size calculation and triggers evict if necessary
        if self._backup_manager is not None:
            self._backup_manager.mark_updated()
        for sampler in self._samplers:
            sampler.build_index(key, value)

    def _execute_delete_operation(self, key):
        self._cache.delete(key)
        self._db.delete(to_bytes(key))
        for sampler in self._samplers:
            sampler.remove_index(key)

    def _execute_evict_operation(self):
        if not self._cache.size_exceeds_limit():  # maybe several delete has occurred, so not exceeding anymore
            return
        # still exceeds limit. evict
        key_lru, value_lru = next(iter(self._cache._data.items()))
        self._add_to_db(key_lru, value_lru)
        self._cache.delete(key_lru)  # size calculation in lru_cache eviction

    def _add_to_db(self, key, value):
        def load_extend_dump(existing_list, new_value):
            if existing_list is None:
                existing_value_deser = []
            else:
                existing_value_deser = pickle.loads(existing_list)
            existing_value_deser.extend(new_value)
            return pickle.dumps(existing_value_deser)

        key_encoded = to_bytes(key)
        existing_value = self._db.get(key_encoded)
        serialized_value = load_extend_dump(existing_value, value)
        self._db.set(key_encoded, serialized_value)

    def _execute_populate_operation(self, key):
        key_encoded = key.encode()
        value = self._db.get(key_encoded)
        if value is None:  # maybe already deleted
            return
        self._cache.push(key, value)
        self._db.delete(key_encoded)

    def _get_task_batch(self, queue):
        """
        Return a batch of tasks from the queue, to be executed.
        """
        tash_batch = []
        BATCH_LIMIT = 30  # TODO: TBD
        while len(tash_batch) < BATCH_LIMIT and not queue.empty():
            task = queue.get()
            tash_batch.append(task)
        return tash_batch

    def _take_snapshot(self):
        # pickle cache and zip rocksdb data to their file paths
        cache_state = {
            "dict": self._cache._data,
            "size_in_bytes": self._cache._eviction_manager._size_in_bytes,
        }
        from verl.experimental.replay_buffer.persistable_replay_buffer_client import PersistableReplayBufferClient
        from verl.experimental.replay_buffer.persistable_replay_buffer_util.util import delete_files

        local_rocksdb_path = (
            f"{self._db_path}.zip"  # TODO: Maybe don't put as local variables, because shared with backup_manager
        )
        local_cache_path = os.path.join(PersistableReplayBufferClient.MAGIC_SUFFIX, "lru_cache.pickle")

        delete_files(local_rocksdb_path, local_cache_path)
        shutil.make_archive(f"{self._db_path}", "zip", root_dir=self._db_path)
        with open(local_cache_path, "wb") as f:
            pickle.dump(cache_state, f)
        logger.info("snapshot of the replay buffer taken.")
        self._backup_manager.trigger_upload()
