import os
import logging
import collections
import uuid
from dataclasses import dataclass, field
from itertools import cycle
import copy
from typing import Any, List, Dict, Set, Optional, Tuple, Union
import ray
import torch
import numpy as np
from tensordict import TensorDict
from verl.protocol import (
    DataProto, DataProtoFuture
)


# A handle to wait for asynchronous operations to complete
class WaitHandle:
    def __init__(self, object_refs: List[ray.ObjectRef]):
        self.object_refs = object_refs

    def wait(self):
        ray.get(self.object_refs)

    
class PrefetchingDistIterator:
    def __init__(
        self,
        dist_data_proto: "DistDataProto",
        batch_plan: List[List[uuid.UUID]],
        epochs: int,
        prefetch_k: int = 2,
        device = "cpu",
        select_keys = None,
        non_tensor_select_keys = None
    ):
        self.ddp = dist_data_proto
        self.batch_plan = batch_plan  # 预先计算好的批次列表
        self.epochs = epochs
        self.prefetch_k = max(1, prefetch_k)
        self.device = device
        self.select_keys = select_keys
        self.non_tensor_select_keys = non_tensor_select_keys

        self._total_batches = len(self.batch_plan)
        self._full_plan_iter_idx = 0  # 指向下一个要fetch的batch在plan中的索引
        self._current_epoch = 0

        self._pending_futures: collections.deque[DataProtoFuture] = collections.deque()

    def __iter__(self):
        self._full_plan_iter_idx = 0
        self._current_epoch = 0
        self._pending_futures.clear()

        self._start_prefetching()
        
        return self

    def _get_next_batch_info(self) -> Optional[List[uuid.UUID]]:
        if self._current_epoch >= self.epochs:
            return None # 所有epoch完成

        batch_idx_in_epoch = self._full_plan_iter_idx % self._total_batches
        
        if self._full_plan_iter_idx > 0 and batch_idx_in_epoch == 0:
            self._current_epoch += 1
        
        if self._current_epoch >= self.epochs:
            return None

        self._full_plan_iter_idx += 1
        return self.batch_plan[batch_idx_in_epoch]

    def _fetch_one_batch(self):
        row_ids_for_next_batch = self._get_next_batch_info()

        if row_ids_for_next_batch is not None:
            micro_batch_ddp = DistDataProto(
                row_ids=row_ids_for_next_batch,
                row_to_worker=self.ddp.row_to_worker,
                dataworker_handles=self.ddp.dataworker_handles,
            )
            
            future = micro_batch_ddp.select(
                 batch_keys=self.select_keys,
                 non_tensor_batch_keys=self.non_tensor_select_keys,
                 async_mode=True,
                 device=self.device
            )
            self._pending_futures.append(future)

    def _start_prefetching(self):
        for _ in range(self.prefetch_k):
            self._fetch_one_batch()

    def __next__(self) -> DataProto:
        if not self._pending_futures:
            raise StopIteration
        
        future_to_get = self._pending_futures.popleft()
        
        self._fetch_one_batch()
        
        data_proto = future_to_get.get()
        return data_proto

    def __len__(self):
        return self._total_batches * self.epochs
    

@ray.remote
class DataWorker:
    def __init__(self):
        self.data: Dict[uuid.UUID, DataProto] = {}

    def get_data_len(self):
        return len(self.data)

    def put(self, row_id: uuid.UUID, item: DataProto) -> None:
        self.data[row_id] = item

    def update(self, row_id: uuid.UUID, item: DataProto) -> None:
        if row_id not in self.data:
            raise ValueError(f"Row {row_id} not found")
        if item.batch is not None:
                self.data[row_id].batch.update(item.batch)
        if item.non_tensor_batch is not None:
                self.data[row_id].non_tensor_batch.update(item.non_tensor_batch)
        if item.meta_info is not None:
            self.data[row_id].meta_info.update(item.meta_info)

    def get(self, row_id: uuid.UUID, tensor_keys: List[str] | None = None, non_tensor_keys: List[str]  | None = None) -> DataProto:
        if row_id not in self.data:
            raise ValueError(f"Row {row_id} not found")
        return self.data[row_id].select(tensor_keys, non_tensor_keys)
    
    def get_metadata(self, row_ids: List[uuid.UUID]) -> Dict[uuid.UUID, Dict]:
        metadata = {}
        for row_id in row_ids:
            if row_id not in self.data:
                continue
            
            # if row_id in self._metadata_cache:
            #     metadata[row_id] = self._metadata_cache[row_id]
            #     continue
            
            data_item = self.data[row_id]
            meta = {"shape": {}}
            
            if "attention_mask" in data_item.batch:
                attention_mask = data_item.batch["attention_mask"]
                meta["token_num"] = int(attention_mask.sum().item())
            #     meta["max_seq_len"] = attention_mask.shape[-1]

            for key in data_item.batch.keys():
                meta["shape"][key] = data_item.batch[key].shape
            
            # self._metadata_cache[row_id] = meta
            metadata[row_id] = meta
            
        return metadata

    def delete(self, row_ids: List[uuid.UUID]) -> None:
        for row_id in row_ids:
            if row_id in self.data:
                del self.data[row_id]

    def copy_local(self, source_id: uuid.UUID, target_id: uuid.UUID) -> None:
        if source_id not in self.data:
            raise ValueError(f"Source row {source_id} not found")
        if target_id in self.data:
            raise ValueError(f"Target row {target_id} already exists")
        self.data[target_id] = copy.deepcopy(self.data[source_id])

    def copy_remote(self, source_ref: ray.ObjectRef, target_id: uuid.UUID) -> None:
        if target_id in self.data:
            raise ValueError(f"Target row {target_id} already exists")
        self.data[target_id] = copy.deepcopy(ray.get(source_ref))

    def union_local(self, row_id: str, new_data_id: uuid.UUID) -> None:
        new_data_item = self.data[new_data_id]
        self.data[row_id].union(new_data_item)

    def union_remote(self, row_id: str, new_data_id: uuid.UUID, other_worker) -> None:
        new_data_item = other_worker.get.remote(
            new_data_id, 
        )
        self.data[row_id].union(new_data_item)

class DistDataProtoConfig(type):
    _config = {}

    auto_padding_key = "_verl_auto_padding"

    @property
    def auto_padding(cls):
        enabled_by_env = os.getenv("VERL_AUTO_PADDING", "FALSE").upper() in ["TRUE", "1"]
        return enabled_by_env or cls._config.get(cls.auto_padding_key, False)

    @auto_padding.setter
    def auto_padding(cls, enabled: bool):
        assert isinstance(enabled, bool), f"enabled must be a boolean, got {enabled} as {type(enabled)}"
        cls._config[cls.auto_padding_key] = enabled


def pad_distdataproto_to_divisor(data: "DistDataProto", size_divisor: int):
    """
    Pad the DistDataProto to size divisible by size_divisor (distributed version).
    
    Args:
        size_divisor (int): Size divisor. Must be positive.
    
    Returns:
        Tuple[DistDataProto, int]: Padded DistDataProto and pad size.
    """
    assert isinstance(size_divisor, int) and size_divisor > 0, \
        f"size_divisor must be a positive integer, got {size_divisor}"
    
    current_len = len(data.row_ids)
    if current_len == 0:
        logging.warning("Padding a DistDataProto with no items, no changes made")
        return data, 0
    
    remainder = current_len % size_divisor
    if remainder == 0:
        return data, 0
    
    pad_size = size_divisor - remainder

    pad_source_rows = [data.row_ids[i % len(data.row_ids)] for i in range(pad_size)]
    
    pad_row_ids = []
    pad_row_to_worker = {}
    copy_tasks = []
    
    for r_src in pad_source_rows:
        r_pad = uuid.uuid4()
        pad_row_ids.append(r_pad)
        src_worker_idx = data.row_to_worker[r_src]
        pad_row_to_worker[r_pad] = src_worker_idx
        worker = data.dataworker_handles[src_worker_idx]
        copy_tasks.append(worker.copy_local.remote(r_src, r_pad))
    
    if copy_tasks:
        ray.get(copy_tasks)
    
    new_row_ids = data.row_ids + pad_row_ids
    new_row_to_worker = data.row_to_worker.copy()
    new_row_to_worker.update(pad_row_to_worker)
    
    return DistDataProto(
        row_ids=new_row_ids,
        row_to_worker=new_row_to_worker,
        dataworker_handles=data.dataworker_handles,
        meta_info=data.meta_info.copy() if data.meta_info else None
    ), pad_size

def unpad_distdataproto(data, pad_size: int) -> "DistDataProto":
    """
    Unpad the DistDataProto by removing the last `pad_size` rows (distributed version).
    
    Args:
        pad_size (int): Number of rows to remove from the end.
    
    Returns:
        DistDataProto: Unpadded DistDataProto.
    """
    if pad_size <= 0:
        return data
    
    new_row_ids = data.row_ids[:-pad_size]
    new_row_to_worker = {rid: data.row_to_worker[rid] for rid in new_row_ids}

    # unpad_row_ids = data.row_ids[pad_size:]
    # delete_tasks = []
    # for row_id in unpad_row_ids:
    #     worker_idx = data.row_to_worker[row_id]
    #     worker = data.dataworker_handles[worker_idx]
    #     delete_tasks.append(worker.delete.remote([row_id]))

    # ray.get(delete_tasks)
    
    return DistDataProto(
        row_ids=new_row_ids,
        row_to_worker=new_row_to_worker,
        dataworker_handles=data.dataworker_handles,
        meta_info=data.meta_info.copy() if data.meta_info else None
    )

class DistDataProto:
    def __init__(
        self,
        row_ids: List[uuid.UUID],
        row_to_worker: Dict[uuid.UUID, int],
        dataworker_handles: List[ray.actor.ActorHandle],
        meta_info: dict = None
    ):
        self.row_ids = row_ids
        self.row_to_worker = row_to_worker  
        self.dataworker_handles = dataworker_handles  
        self._num_workers = len(dataworker_handles) if dataworker_handles else 0
        self.meta_info = meta_info

    @staticmethod
    def _shard(data: DataProto, dataworker_handles: List[ray.actor.ActorHandle]) -> Tuple[List[uuid.UUID], Dict[uuid.UUID, int]]:
        """Shards the DataProto and stores the partitions across DataWorkers.
        
        Partitions the input data into shards, distributes them to DataWorker actors,
        and returns metadata for accessing the stored shards.
        Args:
            data: Data protocol buffer to be partitioned and stored.
            dataworker_handles: Ray actor handles to DataWorker instances where shards will be stored.
        
        Returns:
            A tuple containing:
            - List of UUIDs: Unique identifiers assigned to each data shard.
            - Dictionary mapping each shard UUID to the index of its corresponding DataWorker 
            in `dataworker_handles` (indicating which DataWorker stores the shard).
        
        """
        if not isinstance(data, DataProto) or len(data) == 0:
            raise ValueError("Invalid DataProto input")
        num_workers = len(dataworker_handles)
        if num_workers == 0:
            raise ValueError("No DataWorkers available")

        row_ids = []
        row_to_worker = {}
        put_tasks = []

        for idx in range(len(data)):
            row_id = uuid.uuid4()
            row_ids.append(row_id)
            worker_idx = idx % num_workers  # scatter evenly
            row_to_worker[row_id] = worker_idx

            row_data = data[idx:idx+1]
            worker = dataworker_handles[worker_idx]
            put_tasks.append(worker.put.remote(row_id, row_data))

        ray.get(put_tasks)
        return row_ids, row_to_worker
    
    @classmethod
    def from_dict(
        cls,         
        tensors: Optional[dict[str, torch.Tensor]] = None,
        non_tensors=None,
        meta_info=None,
        num_batch_dims=1,
        auto_padding=False,
        dataworker_handles=None) -> "DistDataProto":

        data = DataProto.from_dict(
            tensors,
            non_tensors,
            meta_info,
            num_batch_dims,
            auto_padding
        )
        if not dataworker_handles:
            raise RuntimeError("DataWorkers not initialized")
        row_ids, row_to_worker = cls._shard(data, dataworker_handles)
        if meta_info is None:
            meta_info = {} 
        if auto_padding:
            meta_info[DistDataProtoConfig.auto_padding_key] = True
        return cls(row_ids, row_to_worker, dataworker_handles, meta_info=meta_info)

    @classmethod
    def from_single_dict(cls, data: dict, meta_info=None, auto_padding=False, dataworker_handles=None) -> "DistDataProto":
        data = DataProto.from_single_dict(data, meta_info, auto_padding)
        if not dataworker_handles:
            raise RuntimeError("DataWorkers not initialized")
        row_ids, row_to_worker = cls._shard(data, dataworker_handles)
        if meta_info is None:
            meta_info = {}
        if auto_padding:
            meta_info[DistDataProtoConfig.auto_padding_key] = True
        return cls(row_ids, row_to_worker, dataworker_handles, meta_info=meta_info)
    
    @classmethod
    def from_dataproto(cls, data: DataProto, dataworker_handles=None) -> "DistDataProto":
        if not dataworker_handles:
            raise RuntimeError("DataWorkers not initialized")
        row_ids, row_to_worker = cls._shard(data, dataworker_handles)
        if data.meta_info is None:
            meta_info = {}
        else:
            meta_info = data.meta_info.copy()
        return cls(row_ids, row_to_worker, dataworker_handles, meta_info=meta_info)
    
    @staticmethod
    def concat(data: List["DistDataProto"]) -> "DistDataProto":
        """Concat a list of DistDataProto. The rows are concatenated in order.
        The meta_info is assumed to be identical and will use the first one.

        Args:
            data (List[DistDataProto]): list of DistDataProto

        Returns:
            DistDataProto: concatenated DistDataProto
        """
        if not data:
            raise ValueError("Cannot concat empty list of DistDataProto")
        
        first_handles = data[0].dataworker_handles
        for i, item in enumerate(data[1:], 1):
            if item.dataworker_handles != first_handles:
                raise ValueError(f"DistDataProto at index {i} has different dataworker_handles")
        
        concatenated_row_ids = []
        concatenated_row_to_worker = {}
        
        for dist_data in data:
            concatenated_row_ids.extend(dist_data.row_ids)
            
            for row_id in dist_data.row_ids:
                if row_id in concatenated_row_to_worker and concatenated_row_to_worker[row_id] != dist_data.row_to_worker[row_id]:
                    raise ValueError(f"Conflicted row_id {row_id} found during concat")
                concatenated_row_to_worker[row_id] = dist_data.row_to_worker[row_id]
        
        meta_info = data[0].meta_info if data else None
        
        return DistDataProto(
            row_ids=concatenated_row_ids,
            row_to_worker=concatenated_row_to_worker,
            dataworker_handles=first_handles,
            meta_info=meta_info
        )

    def repeat(self, repeat_times=2, interleave=True):
        """
        Repeat the distributed data a specified number of times.

        Args:
            repeat_times (int): Number of times to repeat the data.
            interleave (bool): Whether to interleave the repeated data.

        Returns:
            DistDataProto: A new DistDataProto with repeated data.
        """
        if repeat_times <= 0:
            raise ValueError("repeat_times must be positive")
        
        if repeat_times == 1:
            # No need to repeat, return a copy
            return DistDataProto(
                row_ids=self.row_ids.copy(),
                row_to_worker=self.row_to_worker.copy(),
                dataworker_handles=self.dataworker_handles,
                meta_info=self.meta_info.copy()
            )

        new_row_ids = []
        new_row_to_worker = {}
        copy_tasks = []

        if interleave:
            # Interleave mode: [A, B, C] -> [A, A, B, B, C, C] (repeat_times=2)
            for original_row_id in self.row_ids:
                worker_idx = self.row_to_worker[original_row_id]
                worker = self.dataworker_handles[worker_idx]
                
                # First copy is the original
                new_row_ids.append(original_row_id)
                new_row_to_worker[original_row_id] = worker_idx
                
                # Create additional copies
                for _ in range(repeat_times - 1):
                    new_row_id = uuid.uuid4()
                    new_row_ids.append(new_row_id)
                    new_row_to_worker[new_row_id] = worker_idx
                    
                    # Schedule copy operation
                    copy_tasks.append(
                        worker.copy_local.remote(original_row_id, new_row_id)
                    )
        else:
            # Non-interleave mode: [A, B, C] -> [A, B, C, A, B, C] (repeat_times=2)
            for _ in range(repeat_times):
                for original_row_id in self.row_ids:
                    worker_idx = self.row_to_worker[original_row_id]
                    worker = self.dataworker_handles[worker_idx]
                    
                    if _ == 0:
                        # First iteration: use original row_ids
                        new_row_ids.append(original_row_id)
                        new_row_to_worker[original_row_id] = worker_idx
                    else:
                        # Subsequent iterations: create new copies
                        new_row_id = uuid.uuid4()
                        new_row_ids.append(new_row_id)
                        new_row_to_worker[new_row_id] = worker_idx
                        
                        # Schedule copy operation
                        copy_tasks.append(
                            worker.copy_local.remote(original_row_id, new_row_id)
                        )

        # Wait for all copy operations to complete
        if copy_tasks:
            ray.get(copy_tasks)

        return DistDataProto(
            row_ids=new_row_ids,
            row_to_worker=new_row_to_worker,
            dataworker_handles=self.dataworker_handles,
            meta_info=self.meta_info.copy()
            )

    def copy(self):
        return DistDataProto(
            row_ids=self.row_ids.copy(),
            row_to_worker=self.row_to_worker.copy(),
            dataworker_handles=self.dataworker_handles,
            meta_info=self.meta_info.copy()
        )

    def reorder(self, indices: List[int]):
        if len(indices) != len(self.row_ids):
            raise ValueError("Indices length mismatch")
        max_idx = len(self.row_ids) - 1
        for idx in indices:
            if not (0 <= idx <= max_idx):
                raise IndexError("Index out of range")

        new_row_ids = [self.row_ids[i] for i in indices]
        self.row_ids = new_row_ids

    def split(self, split_size: int) -> List["DistDataProto"]:
        if split_size <= 0:
            raise ValueError("split_size must be positive")

        chunks = []
        for i in range(0, len(self.row_ids), split_size):
            chunk_row_ids = self.row_ids[i:i+split_size]
            chunks.append(DistDataProto(
                row_ids=chunk_row_ids,
                row_to_worker=self.row_to_worker.copy(),
                dataworker_handles=self.dataworker_handles,
                meta_info=self.meta_info.copy()
            ))
        return chunks

    def is_padding_enabled(self):
        """
        Check if padding is enabled for the DataProto.
        Returns:
            bool: True if padding is enabled, False otherwise.
        """
        dataproto_specific_padding = self.meta_info.get(DistDataProtoConfig.auto_padding_key, False)
        return dataproto_specific_padding or DistDataProtoConfig.auto_padding

    def padding(self, padding_size, padding_candidate=""):
        """Pad the DistDataProto by concatenating with padding_candidate.repeat(padding_size)
        Args:
            padding_size (int): the number of repeated padding_candidate
            padding_candidate: the item to be repeated and appended to the DistDataProto, only supporting ["first", "last"]
        """
        if padding_size == 0:
            return
        
        if padding_candidate == "first":
            source_row_id = self.row_ids[0]
        elif padding_candidate == "last":
            source_row_id = self.row_ids[-1]
        else:
            raise ValueError(f"padding_candidate must be 'first' or 'last', got {padding_candidate}")
        
        source_worker_idx = self.row_to_worker[source_row_id]
        copy_tasks = []
        
        for i in range(padding_size):
            new_row_id = uuid.uuid4()
            target_worker_idx = i % self._num_workers
            target_worker = self.dataworker_handles[target_worker_idx]
            
            if target_worker_idx == source_worker_idx:
                # Same worker, use local copy
                copy_tasks.append(
                    target_worker.copy_local.remote(source_row_id, new_row_id)
                )
            else:
                # Different worker, need to get data first then copy
                source_worker = self.dataworker_handles[source_worker_idx]
                data_ref = source_worker.get.remote(
                    source_row_id, 
                )
                copy_tasks.append(
                    target_worker.copy_remote.remote(data_ref, new_row_id)
                )
            
            # Update metadata
            self.row_ids.append(new_row_id)
            self.row_to_worker[new_row_id] = target_worker_idx
        
        # Wait for all copy operations to complete
        if copy_tasks:
            ray.get(copy_tasks)

    def chunk(self, chunks: int) -> List["DistDataProto"]:
        """Split the batch among dim=0 into chunks. The meta_info is passed to each DistDataProto after split.
        Args:
            chunks (int): the number of chunks to split on dim=0
        Returns:
            List[DistDataProto]: a list of DistDataProto after splitting
        """
        total_rows = len(self.row_ids)
        
        padding_enabled = self.meta_info.get('auto_padding', False) if self.meta_info else False
        
        if not padding_enabled:
            assert total_rows % chunks == 0, (
                f"only support equal chunk. Got size of DistDataProto {total_rows} and chunk {chunks}."
            )
        
        if padding_enabled:
            base_size = total_rows // chunks
            remainder = total_rows % chunks
            chunk_sizes = [base_size + (1 if i < remainder else 0) for i in range(chunks)]
        else:
            chunk_size = total_rows // chunks
            chunk_sizes = [chunk_size] * chunks
        
        chunked_row_ids = []
        start_idx = 0
        for chunk_size in chunk_sizes:
            end_idx = start_idx + chunk_size
            chunked_row_ids.append(self.row_ids[start_idx:end_idx])
            start_idx = end_idx

        output = []
        for chunk_row_ids in chunked_row_ids:
            chunk_row_to_worker = {}
            for row_id in chunk_row_ids:
                chunk_row_to_worker[row_id] = self.row_to_worker[row_id]
            
            chunk_ddp = DistDataProto(
                row_ids=chunk_row_ids,
                row_to_worker=chunk_row_to_worker,
                dataworker_handles=self.dataworker_handles,
                meta_info=self.meta_info.copy() if self.meta_info else None
            )
            output.append(chunk_ddp)
        
        return output


    @staticmethod
    def select_collect_fn(data: list["DataProto"], meta_info, device) -> "DataProto":
        data_out = DataProto.concat(data)
        if meta_info:
            data_out.meta_info = meta_info.copy()
        return data_out.to(device)

    def select(self, batch_keys = None, non_tensor_batch_keys = None, meta_info_keys=None, async_mode: bool = False, device="cpu") -> Union[DataProto, DataProtoFuture]:
        """Selectively fetch and materialize specified data entries from distributed workers.
        
        Unlike the `select` method in DataProto which performs filtering, this method triggers 
        materialization (immediate loading) of data for explicitly specified keys.
        Args:
            batch_keys (Optional[Iterable[str]]): a list of strings indicating the keys in batch to selectively fetch. All tensor data will be fetched if None.
            non_tensor_batch_keys (Optional[Iterable[str]]): a list of strings indicating the keys in no tensor batch to selectively fetch. All non tensor data will be fetched if None.
            meta_info_keys (Optional[Iterable[str]]): Placeholder, no function for now.
            async_mode (bool): When True, returns immediately with a DataProtoFuture allowing 
                asynchronous retrieval. Otherwise blocks until data is ready. Defaults to False.
            device (str): Target device for output DataProto. Defaults to "cpu".
        Returns:
            Union[DataProto, DataProtoFuture]: Materialized data container. Returns a future 
            object if `async_mode` is True; otherwise returns the populated DataProto.
        """
        if not self.row_ids:
            return DataProto() if not async_mode else DataProtoFuture([])
        if not self.dataworker_handles:
            raise ValueError("No DataWorkers available")

        get_tasks = []
        for row_id in self.row_ids:
            worker_idx = self.row_to_worker[row_id]
            worker = self.dataworker_handles[worker_idx]
            get_tasks.append(worker.get.remote(row_id, batch_keys, non_tensor_batch_keys))

        if async_mode:
            data_out = DataProtoFuture(collect_fn=lambda x:self.select_collect_fn(x, self.meta_info, device), futures=get_tasks)
        else:
            fetched_rows = ray.get(get_tasks)
            data_out = self.select_collect_fn(fetched_rows, self.meta_info, device)
        return data_out
        
    def get_metadata(self) -> Dict[uuid.UUID, Dict]:
        if not self.row_ids:
            return {}
        
        worker_to_rows = {}
        for row_id in self.row_ids:
            worker_idx = self.row_to_worker[row_id]
            if worker_idx not in worker_to_rows:
                worker_to_rows[worker_idx] = []
            worker_to_rows[worker_idx].append(row_id)
        
        tasks = []
        for worker_idx, row_list in worker_to_rows.items():
            worker = self.dataworker_handles[worker_idx]
            tasks.append(worker.get_metadata.remote(row_list))
        
        results = ray.get(tasks)
        metadata = {}
        for result in results:
            metadata.update(result)
        
        return metadata
    
    def update(self, data: DataProto):
        if not self.row_ids:
            return None
        assert len(self.row_ids) == len(data)

        update_tasks = []
        for idx, row_id in enumerate(self.row_ids):
            row_data = data[idx:idx+1]
            worker_idx = self.row_to_worker[row_id]
            worker = self.dataworker_handles[worker_idx]
            update_tasks.append(worker.update.remote(row_id, row_data))
        ray.get(update_tasks)
        if data.meta_info is not None:
            if self.meta_info is None:
                self.meta_info = data.meta_info.copy()
            else:
                self.meta_info.update(data.meta_info)

    def destroy(self, async_mode: bool = False) -> Optional[WaitHandle]:
        if not self.row_ids:
            return None

        delete_tasks = []
        for row_id in self.row_ids:
            worker_idx = self.row_to_worker[row_id]
            worker = self.dataworker_handles[worker_idx]
            delete_tasks.append(worker.delete.remote([row_id]))

        if async_mode:
            return WaitHandle(delete_tasks)
        else:
            ray.get(delete_tasks)
            self.row_ids = []
            self.row_to_worker = {}
            return None
        
    def union(self, other: "DistDataProto") -> "DistDataProto":
        """Union with another DistDataProto. Union each row separately on DataWorkers.
        
        Args:
            other (DistDataProto): another DistDataProto to union
            
        Returns:
            DistDataProto: the DistDataProto after union (returns self)
        """
        if len(self.row_ids) != len(other.row_ids):
            raise ValueError(f"Cannot union DistDataProto with different lengths: {len(self.row_ids)} vs {len(other.row_ids)}")
        
        if len(self.row_ids) == 0:
            return self
        
        union_tasks = []
        
        for i, (self_row_id, other_row_id) in enumerate(zip(self.row_ids, other.row_ids)):
            self_worker_idx = self.row_to_worker[self_row_id]
            other_worker_idx = other.row_to_worker[other_row_id]
            
            self_worker = self.dataworker_handles[self_worker_idx]
            other_worker = other.dataworker_handles[other_worker_idx]
            
            if self_worker_idx == other_worker_idx:
                task = self_worker.union_local.remote(self_row_id, other_row_id)
            else:
                task = self_worker.union_remote.remote(self_row_id, other_row_id, other_worker)
            
            union_tasks.append(task)
        
        ray.get(union_tasks)
        
        return self
    
    def make_iterator(
        self,
        mini_batch_size: Optional[int] = 8,
        epochs: int = 1,
        seed=None,
        dataloader_kwargs={},
        read_order: Optional[List[List[uuid.UUID]]] = None,
        prefetch_k: int = 2,
        device = "cpu",
        select_keys = None,
        non_tensor_select_keys = None
    ) -> PrefetchingDistIterator:
        """
        创建一个支持预取的分布式迭代器。
        Args:
            mini_batch_size (int): 当 read_order 未提供时，用于顺序切分的批次大小。
            epochs (int): 迭代的总轮数。
            read_order (Optional[List[List[uuid.UUID]]]): 
                一个预定义的读取顺序。每个内部列表代表一个微批次，
                包含该批次所有样本的 row_id。如果提供此参数，
                将忽略 mini_batch_size。这对于动态批处理等高级
                用法至关重要。
            prefetch_k (int): 预取窗口的大小。例如，k=2 表示在处理
                批次 N 时，系统会异步拉取批次 N+1 和 N+2。
        Returns:
            PrefetchingDistIterator: 一个可以按指定顺序或默认顺序进行
                                    高效迭代的迭代器。
        """
        if dataloader_kwargs and dataloader_kwargs.get("shuffle", False) and seed:
            assert read_order is None, "read_order must be None when shuffle is enabled"
            read_order = []
            generator = torch.Generator()
            generator.manual_seed(seed)
            idx_dataloader = torch.utils.data.DataLoader(self.row_ids, batch_size=mini_batch_size, generator=generator, **dataloader_kwargs, collate_fn=lambda x:x)
            for _ in range(epochs):
                for batch_idxs in idx_dataloader:
                    read_order.append(batch_idxs)
        if read_order:
            batch_plan = read_order
            # total_items_in_plan = sum(len(batch) for batch in batch_plan)
            # if total_items_in_plan != len(self.row_ids):
            #      print(f"Warning: Number of items in read_order ({total_items_in_plan}) "
            #            f"does not match data size ({len(self.row_ids)}).")
        else:
            if mini_batch_size <= 0:
                raise ValueError("mini_batch_size must be positive for default ordering.")
            assert len(self.row_ids) % mini_batch_size == 0, \
                f"Data size {len(self.row_ids)} is not divisible by mini_batch_size {mini_batch_size}."
            
            batch_plan = []
            for _ in range(epochs):
                for i in range(0, len(self.row_ids), mini_batch_size):
                    batch_plan.append(self.row_ids[i:i + mini_batch_size])
        return PrefetchingDistIterator(
            dist_data_proto=self,
            batch_plan=batch_plan,
            epochs=1,
            prefetch_k=prefetch_k,
            device=device,
            select_keys=select_keys,
            non_tensor_select_keys=non_tensor_select_keys
        )

    def __len__(self) -> int:
        return len(self.row_ids)

    def __repr__(self) -> str:
        return (f"DistDataProto(row_count={len(self)}, "
                f"num_workers={self._num_workers})")
