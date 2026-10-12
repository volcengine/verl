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
"""Bounded, opt-in read-ahead for immutable tool data."""

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any, Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class ReadAhead(Generic[K, V]):
    """Overlap one predicted read with the next model turn.

    Use one instance per trajectory, with serial calls to ``read`` and a
    read-only, immutable data source. Keys must identify the entire read,
    including any arguments or snapshot that can affect its result.
    ``successor`` must use only the returned value. Results become visible
    only when ``read`` receives the matching key. ``fetch`` must support
    cancellation; call ``aclose`` after reads stop, including on cancellation.

    Args:
        fetch: Async read of one key, safe to run before the model asks for it.
        successor: Pure callback returning the next key, or None to stop.
    """

    def __init__(
        self,
        fetch: Callable[[K], Coroutine[Any, Any, V]],
        successor: Callable[[V], K | None],
    ) -> None:
        self._fetch = fetch
        self._successor = successor
        self._key: K | None = None
        self._pending: asyncio.Task[V] | None = None
        self._closed = False

    async def read(self, key: K) -> V:
        """Read the requested key, then prepare at most its next successor."""
        if self._closed:
            raise RuntimeError("ReadAhead is closed")
        pending = self._pending
        if pending is not None and key == self._key:
            self._pending = None
            try:
                value = await pending
            except Exception:
                # A failed speculative read must not replace the demanded read.
                value = await self._fetch(key)
        else:
            await self._discard()
            value = await self._fetch(key)
        self._key = self._successor(value)
        if self._key is not None:
            self._pending = asyncio.create_task(self._fetch(self._key))
        return value

    async def _discard(self) -> None:
        if self._pending is not None:
            self._pending.cancel()
            await asyncio.gather(self._pending, return_exceptions=True)
            self._pending = None
        self._key = None

    async def aclose(self) -> None:
        """Cancel unused work and retrieve any speculative exception."""
        self._closed = True
        await self._discard()
