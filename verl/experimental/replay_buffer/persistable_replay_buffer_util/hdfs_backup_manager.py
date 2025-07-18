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
import shutil
import threading
import time

logger = logging.getLogger(__name__)


class HDFSBackupManager:
    """
    HDFS backup manager for the replay buffer. Periodically backup rocksdb data to HDFS.
    """

    BACKUP_INTERVAL_IN_SEC = 180
    BACKUP_FILE_NAME = "replay_buffer_data"  # for rocksdb data
    CACHE_DUMP_FILE_NAME = "lru_cache"

    def __init__(self, replay_buffer_name, dp_path, restore_from_hdfs_path=None):
        self._db_path = dp_path
        self._is_updated = False
        self._restored_cache_state = None
        if restore_from_hdfs_path is not None:
            # hdfs_path unique as long as replay_buffer_name are unique
            restore_from_hdfs_path = os.path.join(restore_from_hdfs_path, replay_buffer_name)
            self._mount_path = HDFSBackupManager.hdfs_path_map2_local_path(restore_from_hdfs_path)
            if self._mount_path:
                self._init_from_hdfs()
                self._is_updated = False  # whether push had been called since the last backup
                self._backup_thread = threading.Thread(target=self._backup_loop, daemon=True)

                self._upload_thread = threading.Thread(target=self._upload_to_hdfs_loop, daemon=True)
                self._upload_trigger = threading.Event()  # To let the upload loop sleep until snapshot is ready
            else:
                raise RuntimeError("When HDFS backup is enabled, HDFS fusing must be set up correctly!")

    @staticmethod
    def hdfs_path_map2_local_path(hdfs_path: str) -> str:
        HDFS_PREFIX = "hdfs://"
        FUSE_READER_DISABLE_ENV = "DISABLE_HDFS_FUSE_READER"
        FUSE_VOLUME_ENV = "HDFSFUSE_VOLUMES"

        if not hdfs_path.startswith(HDFS_PREFIX):
            return ""
        if os.getenv(FUSE_READER_DISABLE_ENV, "false") == "true":
            return ""
        fuse_mount_maps = os.getenv(FUSE_VOLUME_ENV, None)
        if not fuse_mount_maps:
            return ""
        fuse_mount_maps = eval(fuse_mount_maps)
        hdfs_path = hdfs_path if hdfs_path.endswith("/") else hdfs_path + "/"
        for record in fuse_mount_maps:
            record_hdfs_path = record.get("hdfs_path", "NONE")
            record_hdfs_path = record_hdfs_path if record_hdfs_path.endswith("/") else record_hdfs_path + "/"
            if hdfs_path.startswith(record_hdfs_path):
                sub_path = hdfs_path[len(record_hdfs_path) :].lstrip("/")
                return os.path.join(record.get("mount_path"), sub_path)
        return ""

    def bind_db_cache_processor(self, db, cache, task_processor):
        self._db = db
        self._cache = cache
        self._task_processor = task_processor

    def run(self):
        self._backup_thread.start()
        self._upload_thread.start()

    def mark_updated(self):
        self._is_updated = True

    def reset_updated(self):
        self._is_updated = False

    def _init_from_hdfs(self):
        """Check whether there is a .zip backup in HDFS. If so, unzip into self.db_path"""
        remote_zip_path = os.path.join(self._mount_path, f"{HDFSBackupManager.BACKUP_FILE_NAME}.zip")
        if os.path.exists(remote_zip_path):
            logger.info("Found existing replay buffer rocksdb backup in HDFS! Restoring to the local.")
            # Unzip to self.db_path and remove zip file
            if os.path.exists(self._db_path):
                shutil.rmtree(self._db_path)
            os.makedirs(self._db_path, exist_ok=True)
            shutil.unpack_archive(remote_zip_path, extract_dir=self._db_path)

        # load lrucache state
        remote_cache_path = os.path.join(self._mount_path, f"{HDFSBackupManager.CACHE_DUMP_FILE_NAME}.pickle")
        if os.path.exists(remote_cache_path):
            logger.info("Found existing replay buffer lrucache backup in HDFS! Restoring to the local.")
            with open(remote_cache_path, "rb") as f:
                self._restored_cache_state = pickle.load(f)

    def get_restored_cache(self):
        return self._restored_cache_state

    def _backup_loop(self):
        """
        Initiates a snapshot task every 3 minutes.
        """
        while True:
            time.sleep(HDFSBackupManager.BACKUP_INTERVAL_IN_SEC)
            if self._is_updated:
                self.reset_updated()
                # Initiate a snapshot task
                from verl.experimental.replay_buffer.task_processor import Task, TaskType

                snapshot_task = Task(TaskType.SNAPSHOT)
                self._task_processor.add_task(snapshot_task)

    def trigger_upload(self):
        self._upload_trigger.set()

    def _upload_to_hdfs_loop(self):
        # Check whether the paths exist (snapshot task executed). If so, upload to hdfs
        from verl.experimental.replay_buffer.persistable_replay_buffer_client import PersistableReplayBufferClient
        from verl.experimental.replay_buffer.persistable_replay_buffer_util.util import delete_files

        local_rocksdb_path = f"{self._db_path}.zip"  # TODO: Maybe don't put them as local variables
        local_cache_path = os.path.join(PersistableReplayBufferClient.MAGIC_SUFFIX, "lru_cache.pickle")
        while True:
            self._upload_trigger.wait()
            self._upload_trigger.clear()

            if os.path.exists(local_rocksdb_path) and os.path.exists(local_cache_path):
                remote_rocksdb_temp_path = os.path.join(self._mount_path, "replay_buffer_data_temp.zip")
                remote_rocksdb_final_path = os.path.join(self._mount_path, f"{HDFSBackupManager.BACKUP_FILE_NAME}.zip")

                remote_cache_temp_path = os.path.join(self._mount_path, "replay_buffer_lrucache_temp.pickle")
                remote_cache_final_path = os.path.join(
                    self._mount_path, f"{HDFSBackupManager.CACHE_DUMP_FILE_NAME}.pickle"
                )
                logger.info("start to back up replay buffer to HDFS.")
                start_time = time.time()
                try:
                    os.makedirs(
                        os.path.dirname(remote_rocksdb_temp_path), exist_ok=True
                    )  # create the parent directory if doesn't exist
                    shutil.copy(local_rocksdb_path, remote_rocksdb_temp_path)
                    os.rename(remote_rocksdb_temp_path, remote_rocksdb_final_path)

                    os.makedirs(os.path.dirname(remote_cache_temp_path), exist_ok=True)
                    shutil.copy(local_cache_path, remote_cache_temp_path)
                    os.rename(remote_cache_temp_path, remote_cache_final_path)
                finally:
                    delete_files(local_rocksdb_path, local_cache_path, remote_rocksdb_temp_path, remote_cache_temp_path)
                logger.info(f"finished backing up replay buffer to HDFS in {time.time() - start_time}s")
