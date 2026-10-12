# Copyright 2026 Individual Contributor: Jiahui Cao
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

import pytest

from verl.tools.read_ahead import ReadAhead


@pytest.mark.asyncio
async def test_next_read_overlaps_model_turn_without_recursive_lookahead():
    started = asyncio.Event()
    calls = []

    async def fetch(key):
        calls.append(key)
        if key == 2:
            started.set()
        return {"key": key, "next": key + 1 if key < 3 else None}

    reader = ReadAhead(fetch, lambda value: value["next"])
    try:
        assert await reader.read(1) == {"key": 1, "next": 2}
        await asyncio.wait_for(started.wait(), timeout=1)
        assert calls == [1, 2]
        assert await reader.read(2) == {"key": 2, "next": 3}
        assert await reader.read(3) == {"key": 3, "next": None}
        assert calls == [1, 2, 3]
    finally:
        await reader.aclose()


@pytest.mark.asyncio
async def test_different_model_argument_discards_prediction():
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def fetch(key):
        if key == 2:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        return {"key": key, "next": 2 if key == 1 else None}

    reader = ReadAhead(fetch, lambda value: value["next"])
    try:
        await reader.read(1)
        await asyncio.wait_for(started.wait(), timeout=1)
        assert await reader.read(9) == {"key": 9, "next": None}
        assert cancelled.is_set()
    finally:
        await reader.aclose()


@pytest.mark.asyncio
async def test_failed_prediction_retries_the_actual_read():
    failed = asyncio.Event()
    attempts = 0

    async def fetch(key):
        nonlocal attempts
        if key == 2:
            attempts += 1
            if attempts == 1:
                failed.set()
                raise OSError("temporary failure")
        return {"key": key, "next": 2 if key == 1 else None}

    reader = ReadAhead(fetch, lambda value: value["next"])
    try:
        await reader.read(1)
        await asyncio.wait_for(failed.wait(), timeout=1)
        assert await reader.read(2) == {"key": 2, "next": None}
        assert attempts == 2
    finally:
        await reader.aclose()


@pytest.mark.asyncio
async def test_actual_read_failure_is_not_hidden():
    async def fetch(key):
        raise OSError("read failed")

    reader = ReadAhead(fetch, lambda value: None)
    try:
        with pytest.raises(OSError, match="read failed"):
            await reader.read(1)
    finally:
        await reader.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("consume", [False, True])
async def test_cancellation_leaves_no_background_read(consume):
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def fetch(key):
        if key == 2:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        return {"next": 2}

    reader = ReadAhead(fetch, lambda value: value["next"])
    await reader.read(1)
    await asyncio.wait_for(started.wait(), timeout=1)
    if consume:
        demand = asyncio.create_task(reader.read(2))
        await asyncio.sleep(0)
        demand.cancel()
        with pytest.raises(asyncio.CancelledError):
            await demand
    await reader.aclose()
    await reader.aclose()
    assert cancelled.is_set()
    with pytest.raises(RuntimeError, match="closed"):
        await reader.read(3)
