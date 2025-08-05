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

import asyncio
import os


async def fn():
    try:
        print(10)
        await asyncio.sleep(100)
    except asyncio.CancelledError:
        print("cancelled from fn")
    print(11)
    return 1


def test_queue_group_pop_from_longest():
    os.environ["VERL_QUEUE_LOGGING_LEVEL"] = "DEBUG"
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def run_test():
        task = fn()
        a = asyncio.ensure_future(task)
        # print(a)
        await asyncio.wait([a], timeout=1)
        a.cancel()
        try:
            await a
        except asyncio.CancelledError:
            print("ccc cancelled")

    loop.run_until_complete(run_test())
    loop.close()
