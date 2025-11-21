# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import asyncio
from asyncio import Queue
from typing import Any

import ray


@ray.remote
class QueueMonitor:
    """
    Queue for early exit. When the queue reaches its threshold, all worker tasks will be canceled.
    """

    def __init__(self, threshold: int = 10):
        self.queue = Queue()
        self.threshold = threshold
        self.workers = []

    def get_threshold(self):
        return self.threshold

    def set_threshold(self, threshold: int):
        self.threshold = threshold

    def check_threshold(self) -> bool:
        current_size = self.queue.qsize()
        return current_size >= self.threshold

    def get_current_size(self) -> int:
        return self.queue.qsize()

    async def put(self, item: Any):
        if self.check_threshold():
            await self._cancel_all_workers()
            return

        await self.queue.put(item)

    def append_worker(self, worker):
        self.workers.append(worker)

    async def _cancel_all_workers(self):
        tasks = [worker.cancel_tasks.remote() for worker in self.workers]
        await asyncio.gather(*tasks)

    def clear(self):
        items = []
        while not self.queue.empty():
            try:
                item = self.queue.get_nowait()
                items.append(item)
                self.queue.task_done()
            except asyncio.QueueEmpty:
                break
        return items
