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
"""
Shared Memory (SHM) Cache Implementation for Multi-Modal Features in vLLM

This module implements a zero-copy IPC mechanism for transferring large multi-modal
features (e.g., image/video tensors ~10MB each) between vLLM's P0 (Engine) and P1 (Worker)
processes in multi-turn multi-modal interactions, avoiding the overhead of ZeroMQ serialization.

Architecture:
    P0 (Engine) --[SHM Write]--> Shared Memory <--[SHM Read]-- P1 (Worker)

    - P0 uses SingleWriterShmObjectStorage to write mm_features
    - P1 uses SingleWriterShmObjectStorage to read mm_features
    - FileLock provides cross-process synchronization (Ray Actor safe)

Performance Impact:
    - Without SHM: ~50ms per 10MB image (ZMQ serialization)
    - With SHM: ~5ms per image (memory pointer + small metadata)
    - ~10x speedup for multi-turn multi-modal workloads with large images/videos

Configuration:
    Enable via RolloutConfig.mm_shm_cache_gb:
    - Set to 0 to disable (default)
    - Set to 2-4 GB for typical multi-turn multi-modal workloads
    - Adjust mm_shm_cache_max_object_size_mb for very large images/videos

See Also:
    - vLLM's native SHM cache: vllm/distributed/device_communicators/shm_object_storage.py
    - Usage: verl/workers/rollout/vllm_rollout/vllm_async_server.py
"""

import fcntl
import logging
import os
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from vllm.distributed.device_communicators.shm_object_storage import SingleWriterShmObjectStorage

logger = logging.getLogger(__name__)

# magic numbers that ensure we are using the same LoRA adapter during the rollout and training process
VLLM_LORA_INT_ID = 123
VLLM_LORA_NAME = "123"
VLLM_LORA_PATH = "simon_lora_path"

GiB_BYTES = 1024 * 1024 * 1024
MiB_BYTES = 1024 * 1024


def get_vllm_max_lora_rank(lora_rank: int):
    """
    For vLLM, the smallest `max_lora_rank` is 8, and allowed values are (8, 16, 32, 64, 128, 256, 320, 512)
    This function automatically adjusts the `max_lora_rank` to the nearest allowed value.

    Reference: https://github.com/vllm-project/vllm/blob/8a297115e2367d463b781adb86b55ac740594cf6/vllm/config/lora.py#L27
    """
    assert lora_rank > 0, f"lora_rank must be greater than 0 to invoke this function, get {lora_rank}"
    vllm_max_lora_ranks = [8, 16, 32, 64, 128, 256, 320, 512]
    for rank in vllm_max_lora_ranks:
        if lora_rank <= rank:
            return rank

    raise ValueError(f"lora_rank must be less than or equal to {vllm_max_lora_ranks[-1]}, but got {lora_rank}")


class FileLock:
    """
    A cross-process file-based lock that can be used across Ray Actors.
    This is needed because multiprocessing.Lock() cannot be shared across Ray Actors.

    The lock file is created in /dev/shm for fast access (same as shared memory).
    """

    def __init__(self, lock_file: str):
        self.lock_file = lock_file
        self._fd = None

    def __enter__(self):
        if self._fd is None:
            self._fd = open(self.lock_file, "w")
        fcntl.flock(self._fd, fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._fd:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            self._fd.close()
            self._fd = None
        return False

    def acquire(self, blocking: bool = True) -> bool:
        if self._fd is None:
            self._fd = open(self.lock_file, "w")
        try:
            if blocking:
                fcntl.flock(self._fd, fcntl.LOCK_EX)
            else:
                fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            self._fd.close()
            self._fd = None
            return False

    def release(self):
        if self._fd:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            self._fd.close()
            self._fd = None


def generate_shm_names(dp_rank_local: int, shm_name_prefix: str, lock_file_prefix: str) -> tuple[str, str]:
    """Generate unique SHM name and lock file path for a given DP rank.

    Using PID ensures different verl runs on the same machine don't conflict.

    Args:
        dp_rank_local: Local data parallel rank
        shm_name_prefix: Prefix for shared memory segment name
        lock_file_prefix: Prefix for lock file path

    Returns:
        tuple: (shm_name, lock_file_path)
    """
    pid = os.getpid()
    shm_name = f"{shm_name_prefix}_{dp_rank_local}_{pid}"
    lock_file = f"{lock_file_prefix}_{dp_rank_local}_{pid}.lock"
    return shm_name, lock_file


def create_shm_sender_cache(
    vllm_config, shm_name: str, lock_file: str, max_object_size_mb: int = 100
) -> Optional["SingleWriterShmObjectStorage"]:
    """
    Create SHM-based sender cache for P0 (Engine side).

    This uses vLLM's SingleWriterShmObjectStorage which:
    1. Creates a shared memory ring buffer
    2. Stores serialized mm_features in shared memory
    3. Returns (address, monotonic_id) instead of actual data

    P1 Workers will read from the same shared memory.

    Args:
        vllm_config: vLLM configuration object
        shm_name: Name for the shared memory segment
        lock_file: Path to the lock file for synchronization
        max_object_size_mb: Maximum size of a single cached object in MB

    Returns:
        SHM cache object or None if disabled/failed
    """
    try:
        model_config = vllm_config.model_config
        mm_config = model_config.get_multimodal_config()

        if mm_config is None or getattr(mm_config, "mm_shm_cache_gb", 0) <= 0:
            logger.info("[P0 SHM Cache] DISABLED: mm_shm_cache_gb=0 or mm_config=None")
            return None

        from vllm.distributed.device_communicators.shm_object_storage import (
            MsgpackSerde,
            SingleWriterShmObjectStorage,
            SingleWriterShmRingBuffer,
        )

        cache_gb = mm_config.mm_shm_cache_gb
        # Use provided max_object_size_mb, or get from mm_config, or default to 100
        max_obj_size_mb = getattr(mm_config, "mm_shm_cache_max_object_size_mb", max_object_size_mb)
        tp_size = vllm_config.parallel_config.tensor_parallel_size

        # Create the lock file
        try:
            with open(lock_file, "w"):
                pass
        except OSError as e:
            logger.error(f"[P0 SHM Cache] Failed to create lock file {lock_file}: {e}")
            return None

        # Clean up stale SHM from previous runs
        _cleanup_stale_shm(shm_name)

        # Create the shared memory ring buffer (P0 is the writer)
        ring_buffer = SingleWriterShmRingBuffer(
            data_buffer_size=int(cache_gb * GiB_BYTES),
            name=shm_name,
            create=True,
        )

        # Create the object storage on top of the ring buffer
        shm_cache = SingleWriterShmObjectStorage(
            max_object_size=max_obj_size_mb * MiB_BYTES,
            n_readers=tp_size,
            ring_buffer=ring_buffer,
            serde_class=MsgpackSerde,
            reader_lock=None,  # P0 (writer) doesn't need reader lock
        )

        logger.info(
            f"[P0 SHM Cache] ENABLED: shm_name={shm_name}, cache_gb={cache_gb}, "
            f"n_readers={tp_size}, max_object_size_mb={max_obj_size_mb}"
        )
        return shm_cache

    except ImportError as e:
        logger.warning(f"[P0 SHM Cache] DISABLED: vLLM SHM support not available: {e}")
        return None
    except Exception as e:
        logger.exception(f"[P0 SHM Cache] DISABLED: Failed to create SHM cache: {e}")
        return None


def create_shm_receiver_cache(
    vllm_config, shm_name: str, lock_file: str, max_object_size_mb: int = 100
) -> "SingleWriterShmObjectStorage":
    """
    Create SHM-based receiver cache for P1 (Worker side).

    This connects to the shared memory created by P0 and reads mm_features from it.

    Args:
        vllm_config: vLLM configuration object
        shm_name: Name for the shared memory segment (must match P0)
        lock_file: Path to the lock file for synchronization
        max_object_size_mb: Maximum size of a single cached object in MB

    Returns:
        SHM cache object

    Raises:
        RuntimeError: If SHM cache creation fails
    """
    from vllm.distributed.device_communicators.shm_object_storage import (
        MsgpackSerde,
        SingleWriterShmObjectStorage,
        SingleWriterShmRingBuffer,
    )

    model_config = vllm_config.model_config
    mm_config = model_config.get_multimodal_config()

    if mm_config is None:
        raise RuntimeError("[P1 SHM Cache] mm_config is None but SHM cache is enabled")

    if getattr(mm_config, "mm_shm_cache_gb", 0) <= 0:
        raise RuntimeError(
            f"[P1 SHM Cache] mm_shm_cache_gb={getattr(mm_config, 'mm_shm_cache_gb', 0)} but SHM cache is enabled"
        )

    cache_gb = mm_config.mm_shm_cache_gb
    # Use provided max_object_size_mb, or get from mm_config, or default to 100
    max_obj_size_mb = getattr(mm_config, "mm_shm_cache_max_object_size_mb", max_object_size_mb)
    tp_size = vllm_config.parallel_config.tensor_parallel_size

    reader_lock = FileLock(lock_file)

    # Connect to the shared memory ring buffer (P1 is a reader, not creator)
    ring_buffer = SingleWriterShmRingBuffer(
        data_buffer_size=int(cache_gb * GiB_BYTES),
        name=shm_name,
        create=False,
    )

    # Create the object storage on top of the ring buffer
    shm_cache = SingleWriterShmObjectStorage(
        max_object_size=max_obj_size_mb * MiB_BYTES,
        n_readers=tp_size,
        ring_buffer=ring_buffer,
        serde_class=MsgpackSerde,
        reader_lock=reader_lock,
    )

    logger.info(
        f"[P1 SHM Cache] ENABLED: shm_name={shm_name}, cache_gb={cache_gb}, max_object_size_mb={max_obj_size_mb}"
    )
    return shm_cache


def _cleanup_stale_shm(shm_name: str) -> None:
    """Try to clean up stale SHM from previous runs."""
    try:
        from multiprocessing import shared_memory

        stale_shm = shared_memory.SharedMemory(name=shm_name)
        stale_shm.close()
        stale_shm.unlink()
        logger.debug(f"[P0 SHM Cache] Cleaned up stale SHM: {shm_name}")
    except FileNotFoundError:
        pass  # No stale SHM, this is expected
    except Exception as e:
        logger.warning(f"[P0 SHM Cache] Failed to clean stale SHM: {e}")


def apply_shm_sender_cache(scheduler_output, mm_sender_cache) -> None:
    """
    Apply SHM-based caching for mm_features before sending to P1.

    This follows vLLM's native SHM cache pattern:
    - For each mm_feature, store the data in shared memory
    - Replace feature.data with an address item containing (address, monotonic_id)
    - P1 will check if data has 'address' field and read from SHM

    Args:
        scheduler_output: vLLM SchedulerOutput object
        mm_sender_cache: SHM cache object from create_shm_sender_cache
    """
    if mm_sender_cache is None:
        return

    try:
        from vllm.multimodal.inputs import (
            MultiModalBatchedField,
            MultiModalFieldElem,
            MultiModalKwargsItem,
        )
    except ImportError:
        return

    if not hasattr(scheduler_output, "scheduled_new_reqs"):
        return

    # Step 1: Pre-touch all cached items to prevent eviction during processing
    for req in scheduler_output.scheduled_new_reqs:
        if not hasattr(req, "mm_features") or not req.mm_features:
            continue
        for feature in req.mm_features:
            identifier = feature.identifier
            if identifier and mm_sender_cache.is_cached(identifier):
                mm_sender_cache.touch(identifier)

    # Step 2: Process each feature (cache hit or miss)
    for req in scheduler_output.scheduled_new_reqs:
        if not hasattr(req, "mm_features") or not req.mm_features:
            continue

        for feature in req.mm_features:
            identifier = feature.identifier
            original_data = feature.data

            if original_data is None:
                continue

            if mm_sender_cache.is_cached(identifier):
                # Cache hit: get address from cache
                address, monotonic_id = mm_sender_cache.get_cached(identifier)
            else:
                # Cache miss: store in SHM
                try:
                    address, monotonic_id = mm_sender_cache.put(identifier, original_data)
                except (ValueError, MemoryError) as e:
                    logger.warning(f"[P0 SHM Cache] Failed to cache {identifier}: {e}")
                    continue

            # Replace data with address item (following vLLM's pattern)
            modality = feature.modality
            addr_elem = MultiModalFieldElem(
                modality=modality,
                key="address",
                data=address,
                field=MultiModalBatchedField(),
            )
            id_elem = MultiModalFieldElem(
                modality=modality,
                key="monotonic_id",
                data=monotonic_id,
                field=MultiModalBatchedField(),
            )
            feature.data = MultiModalKwargsItem.from_elems([addr_elem, id_elem])


def create_shm_receiver_apply_fn(shm_cache, stats_holder):
    """
    Create a function to apply SHM receiver cache when processing scheduler_output.

    This is used to monkey-patch the _apply_mm_cache method on the Worker side.

    Args:
        shm_cache: SHM cache object from create_shm_receiver_cache
        stats_holder: Object with _p1_shm_read_count, _p1_shm_touch_count, _p1_shm_read_errors attrs

    Returns:
        A function that can be used as _apply_mm_cache
    """

    def apply_shm_receiver_cache(scheduler_output):
        """Read mm_features from SHM cache."""
        if shm_cache is None:
            return

        # Collect all address items and pre-touch them
        address_items = []
        for req_data in scheduler_output.scheduled_new_reqs:
            if not hasattr(req_data, "mm_features") or not req_data.mm_features:
                continue

            for feature in req_data.mm_features:
                if feature.data is None:
                    continue

                # Check if this is an address item (from P0 SHM cache)
                if hasattr(feature.data, "__contains__") and "address" in feature.data:
                    address_elem = feature.data["address"]
                    monotonic_id_elem = feature.data["monotonic_id"]

                    address = address_elem.data if hasattr(address_elem, "data") else address_elem
                    monotonic_id = monotonic_id_elem.data if hasattr(monotonic_id_elem, "data") else monotonic_id_elem

                    # Pre-touch: increment reader_count to prevent eviction
                    try:
                        shm_cache.touch("", address=address, monotonic_id=monotonic_id)
                        stats_holder._p1_shm_touch_count += 1
                    except Exception as e:
                        logger.debug(f"[P1 SHM Cache] Failed to touch cache: {e}")
                    address_items.append((feature, address, monotonic_id))

        # Read actual data from SHM
        for feature, address, monotonic_id in address_items:
            try:
                feature.data = shm_cache.get(address, monotonic_id)
                stats_holder._p1_shm_read_count += 1
            except Exception as e:
                logger.warning(f"[P1 SHM Cache] Failed to read from cache: {e}")
                stats_holder._p1_shm_read_errors += 1

    return apply_shm_receiver_cache
