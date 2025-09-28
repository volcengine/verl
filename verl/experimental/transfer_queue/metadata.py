# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright 2025 Huawei Ltd. and/or its affiliates
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

import dataclasses
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from tensordict import TensorDict

from verl.experimental.transfer_queue.utils.utils import ProductionStatus


@dataclass
class FieldMeta:
    """
    Records the metadata of a single data field. (name, dtype, shape, etc.)
    """

    # field name (e.g., 'prompt', 'response', etc.)
    name: str

    # data schema info
    dtype: Optional[Any]
    shape: Optional[Any]

    # data status info
    production_status: ProductionStatus = ProductionStatus.NOT_PRODUCED

    def __str__(self) -> str:
        return (
            f"FieldMeta(name='{self.name}', dtype={self.dtype}, "
            f"shape={self.shape}, production_status={self.production_status})"
        )

    @property
    def is_ready(self) -> bool:
        """Check if this field is ready for consumption"""
        return self.production_status == ProductionStatus.READY_FOR_CONSUME


@dataclass
class SampleMeta:
    """
    Records the metadata of a single data sample (stored as a row in the data system).
    """

    # algorithm related info
    global_step: int  # global step, used for data versioning

    # data retrival info
    global_index: int  # global row index, uniquely identifies a data sample
    storage_id: str  # storage unit id
    local_index: int  # local row index in the storage unit

    # data fields info
    # this fields may not contain all the fields of the sample, but only fields-of-interest
    fields: dict[str, FieldMeta]

    def __post_init__(self):
        """Initialize is_ready property based on field readiness"""
        # Check if all fields are ready and update is_ready property
        object.__setattr__(self, "_is_ready", all(field.is_ready for field in self.fields.values()))

    def __str__(self) -> str:
        return (
            f"SampleMeta(global_step={self.global_step}, "
            f"global_index={self.global_index}, storage_id='{self.storage_id}', "
            f"local_index={self.local_index}, fields={self.fields})"
        )

    @property
    def field_names(self) -> list[str]:
        """Get list of field names for this sample"""
        return list(self.fields.keys())

    @property
    def batch_index(self) -> int:
        """Get the batch index of this sample (to be set by BatchMeta)"""
        return getattr(self, "_batch_index", -1)

    def get_field_by_name(self, name: str) -> Optional[FieldMeta]:
        """Get FieldMeta by field name"""
        return self.fields.get(name)

    def has_field(self, name: str) -> bool:
        """Check if this sample has a specific field"""
        return name in self.fields

    def is_field_ready(self, field_name: str) -> bool:
        """Check if a specific field is ready for consumption"""
        field = self.fields.get(field_name)
        return field.is_ready if field else False

    def add_fields(self, fields: dict[str, FieldMeta]) -> "SampleMeta":
        """
        Add new fields to this sample. New fields will be initialized with given dtype, shape
        and production_status (if provided). If not provided, default values (None, None, READY_FOR_CONSUME)
        will be used.
        This modifies the sample in-place to include the new fields.
        """
        self.fields = _union_fields(self.fields, fields)
        # Update is_ready property
        object.__setattr__(self, "_is_ready", all(field.is_ready for field in self.fields.values()))
        return self

    def union(self, other: "SampleMeta", validate: bool = True) -> "SampleMeta":
        """
        Create a union of this sample's fields with another sample's fields.
        Assume both samples have the same global index. If fields overlap, the
        fields in this sample will be replaced by the other sample's fields.

        Args:
            other: Another SampleMeta to union with
            validate: Whether to validate union conditions

        Returns:
            New SampleMeta with unioned fields (None if validation fails)
        """
        if validate:
            if self.global_index != other.global_index:
                raise ValueError(
                    f"Error: Global indexes ({self.global_index} and {other.global_index}) do not match for union."
                )

        # Merge fields
        self.fields = _union_fields(self.fields, other.fields)

        # Update is_ready property
        object.__setattr__(self, "_is_ready", all(field.is_ready for field in self.fields.values()))
        return self

    @property
    def is_ready(self) -> bool:
        """Check if all fields in this sample are ready for consumption"""
        return getattr(self, "_is_ready", False)

    @property
    def production_status(self) -> dict[str, ProductionStatus]:
        """Get production status for all fields (backward compatibility)"""
        return {name: field.production_status for name, field in self.fields.items()}


@dataclass
class StorageMetaGroup:
    """
    Represents a group of samples stored in the same storage unit.
    Used to organize samples by their storage_id for efficient client operations.
    """

    storage_id: str
    sample_metas: list[SampleMeta] = dataclasses.field(default_factory=list)

    def add_sample_meta(self, sample_meta: SampleMeta) -> None:
        """Add a SampleMeta object to this storage group"""
        self.sample_metas.append(sample_meta)

    def get_batch_indexes(self) -> list[int]:
        """Get all internal indexes from stored SampleMeta objects"""
        return [meta.batch_index for meta in self.sample_metas]

    def get_global_indexes(self) -> list[int]:
        """Get all global indexes from stored SampleMeta objects"""
        return [meta.global_index for meta in self.sample_metas]

    def get_local_indexes(self) -> list[int]:
        """Get all local indexes from stored SampleMeta objects"""
        return [meta.local_index for meta in self.sample_metas]

    def get_field_names(self) -> list[str]:
        """Get all unique field names from stored SampleMeta objects"""
        all_fields: set[str] = set()
        for meta in self.sample_metas:
            all_fields.update(meta.fields.keys())
        return list(all_fields)

    def get_transfer_info(self, field_names: Optional[list[str]] = None) -> dict[str, list | dict]:
        """Convert to dictionary format for backward compatibility"""
        if field_names is None:
            field_names = self.get_field_names()
        return {
            "batch_indexes": self.get_batch_indexes(),
            "global_indexes": self.get_global_indexes(),
            "local_indexes": self.get_local_indexes(),
            "fields": field_names,
            "field_data": {},  # Placeholder for field data to be filled later
        }

    @property
    def size(self) -> int:
        """Number of samples in this storage meta group"""
        return len(self.sample_metas)

    @property
    def is_empty(self) -> bool:
        """Check if this storage meta group is empty"""
        return len(self.sample_metas) == 0

    def __len__(self) -> int:
        """Number of samples in this storage meta group"""
        return self.size

    def __bool__(self) -> bool:
        """Truthiness based on whether group has samples"""
        return not self.is_empty

    def __str__(self) -> str:
        return f"StorageMetaGroup(storage_id='{self.storage_id}', size={self.size})"


@dataclass
class BatchMeta:
    """
    Records the metadata of a batch of data samples.
    """

    samples: list[SampleMeta]
    extra_info: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        """Initialize all computed properties during initialization"""
        # Basic properties
        object.__setattr__(self, "_size", len(self.samples))
        object.__setattr__(self, "_is_ready", all(sample.is_ready for sample in self.samples))

        # Pre-compute all list properties for better performance
        if self.samples:
            for idx, sample in enumerate(self.samples):
                object.__setattr__(sample, "_batch_index", idx)  # Ensure batch_index is set correctly

            object.__setattr__(self, "_global_indexes", [sample.global_index for sample in self.samples])
            object.__setattr__(self, "_local_indexes", [sample.local_index for sample in self.samples])
            object.__setattr__(self, "_storage_ids", [sample.storage_id for sample in self.samples])

            # assume all samples have the same fields.
            object.__setattr__(self, "_field_names", sorted(self.samples[0].field_names))

            # Initialize storage groups for efficient client operations
            storage_meta_groups = self._build_storage_meta_groups()
            object.__setattr__(self, "_storage_meta_groups", storage_meta_groups)
        else:
            object.__setattr__(self, "_global_indexes", [])
            object.__setattr__(self, "_local_indexes", [])
            object.__setattr__(self, "_storage_ids", [])
            object.__setattr__(self, "_field_names", [])
            object.__setattr__(self, "_storage_meta_groups", {})

    @property
    def size(self) -> int:
        """Return the number of samples in this batch"""
        return getattr(self, "_size", 0)

    @property
    def global_indexes(self) -> list[int]:
        """Get all global indexes in this batch"""
        return getattr(self, "_global_indexes", [])

    @property
    def field_names(self) -> list[str]:
        """Get all unique field names in this batch"""
        return getattr(self, "_field_names", [])

    @property
    def local_indexes(self) -> list[int]:
        """Get all local indexes in this batch"""
        return getattr(self, "_local_indexes", [])

    @property
    def storage_ids(self) -> list[str]:
        """Get all storage unit IDs in this batch"""
        return getattr(self, "_storage_ids", [])

    @property
    def is_ready(self) -> bool:
        """Check if all samples in this batch are ready for consumption"""
        # TODO: get ready status from controller realtime
        return getattr(self, "_is_ready", False)

    def _build_storage_meta_groups(self) -> dict[str, StorageMetaGroup]:
        """Build storage groups from samples during initialization"""
        storage_meta_groups: dict[str, StorageMetaGroup] = {}

        for sample in self.samples:
            storage_id = sample.storage_id
            if storage_id not in storage_meta_groups:
                storage_meta_groups[storage_id] = StorageMetaGroup(storage_id=storage_id)

            # Use add_sample_meta to store SampleMeta references directly
            storage_meta_groups[storage_id].add_sample_meta(sample)

        return storage_meta_groups

    @property
    def storage_meta_groups(self) -> dict[str, StorageMetaGroup]:
        """Get storage groups organized by storage_id"""
        return getattr(self, "_storage_meta_groups", {})

    @property
    def storage_unit_ids(self) -> list[str]:
        """Get list of all storage unit IDs"""
        return list(self.storage_meta_groups.keys())

    def get_storage_meta_groups(self, storage_id: str) -> Optional[StorageMetaGroup]:
        """Get storage group by storage ID"""
        return self.storage_meta_groups.get(storage_id)

    # Extra info interface methods
    def get_extra_info(self, key: str, default: Any = None) -> Any:
        """Get extra info by key"""
        return self.extra_info.get(key, default)

    def set_extra_info(self, key: str, value: Any) -> None:
        """Set extra info by key"""
        self.extra_info[key] = value

    def update_extra_info(self, info_dict: dict[str, Any]) -> None:
        """Update extra info with multiple key-value pairs"""
        self.extra_info.update(info_dict)

    def remove_extra_info(self, key: str) -> Any:
        """Remove extra info by key and return its value"""
        return self.extra_info.pop(key, None)

    def clear_extra_info(self) -> None:
        """Clear all extra info"""
        self.extra_info.clear()

    def has_extra_info(self, key: str) -> bool:
        """Check if extra info contains a specific key"""
        return key in self.extra_info

    def get_all_extra_info(self) -> dict[str, Any]:
        """Get all extra info as a dictionary"""
        return self.extra_info.copy()

    def add_fields(self, tensor_dict: TensorDict, set_all_ready: bool = True) -> "BatchMeta":
        """
        Add new fields from a TensorDict to all samples in this batch.
        This modifies each sample in-place to include the new fields.

        Args:
            tensor_dict (TensorDict): The input TensorDict containing new fields.
            set_all_ready (bool): If True, set all production_status to READY_FOR_CONSUME. Default is True.
        """
        fields = _extract_field_metas(tensor_dict, set_all_ready)
        for idx, sample in enumerate(self.samples):
            sample.add_fields(fields=fields[idx])

        # Update batch-level fields cache
        object.__setattr__(self, "_field_names", sorted(self.samples[0].field_names))
        object.__setattr__(self, "_is_ready", all(sample.is_ready for sample in self.samples))
        return self

    def __len__(self) -> int:
        """Return the number of samples in this batch."""
        return len(self.samples)

    def __getitem__(self, item):
        if isinstance(item, int | np.integer):
            sample_meta = self.samples[item] if self.samples else []
            return BatchMeta(samples=[sample_meta], extra_info=self.extra_info)
        else:
            raise TypeError(f"Indexing with {type(item)} is not supported now!")

    def chunk(self, chunks: int) -> list["BatchMeta"]:
        """
        Split this batch into smaller chunks.

        Args:
            chunks: number of chunks

        Return:
            List of smaller BatchMeta chunks
        """
        chunk_list = []
        n = len(self.samples)

        # Calculate the base size and remainder of each chunk
        base_size = n // chunks
        remainder = n % chunks

        start = 0
        for i in range(chunks):
            # Calculate the size of the current chunk(the first remainder chunk is 1 more than the base size)
            current_chunk_size = base_size + 1 if i < remainder else base_size
            end = start + current_chunk_size
            chunk_samples = self.samples[start:end]
            chunk = BatchMeta(samples=chunk_samples, extra_info=self.extra_info.copy())
            chunk_list.append(chunk)
            start = end
        return chunk_list

    @classmethod
    def concat(cls, data: list["BatchMeta"], validate: bool = True) -> Optional["BatchMeta"]:
        """
        Concatenate multiple BatchMeta chunks into one large batch.

        Args:
            data: List of BatchMeta chunks to concatenate
            validate: Whether to validate concatenation conditions

        Returns:
            Concatenated BatchMeta

        Raises:
            ValueError: If validation fails (e.g., field names do not match)
        """
        if not data:
            return None

        if validate:
            base_fields = data[0].field_names

            for chunk in data:
                if chunk.field_names != base_fields:
                    raise ValueError("Error: Field names do not match for concatenation.")

        # Combine all samples
        all_samples = []
        for chunk in data:
            all_samples.extend(chunk.samples)
        # Merge all extra_info dictionaries from the chunks
        merged_extra_info = {}
        for chunk in data:
            merged_extra_info.update(chunk.extra_info)
        return BatchMeta(samples=all_samples, extra_info=merged_extra_info)

    def union(self, other: "BatchMeta", validate: bool = True) -> Optional["BatchMeta"]:
        """
        Create a union of this batch's fields with another batch's fields.
        Assume both batches have the same global indices. If fields overlap, the
        fields in this batch will be replaced by the other batch's fields.

        Args:
            other: Another BatchMeta to union with
            validate: Whether to validate union conditions

        Returns:
            New BatchMeta with unioned fields

        Raises:
            ValueError: If validation fails (e.g., batch sizes or global indexes do not match)
        """
        if validate:
            if self.size != other.size:
                raise ValueError("Error: Batch sizes do not match for union.")

            self_global_indexes = sorted(self.global_indexes)
            other_global_indexes = sorted(other.global_indexes)
            if self_global_indexes != other_global_indexes:
                raise ValueError("Error: Global indexes do not match for union.")

        # Create a mapping from global_index to SampleMeta in the other batch
        other_sample_map = {sample.global_index: sample for sample in other.samples}

        # Merge samples
        merged_samples = []
        for sample in self.samples:
            if sample.global_index in other_sample_map:
                other_sample = other_sample_map[sample.global_index]
                merged_sample = sample.union(other_sample, validate=validate)
                merged_samples.append(merged_sample)
            else:
                merged_samples.append(sample)

        # Merge extra info dictionaries
        merged_extra_info = {**self.extra_info, **other.extra_info}

        return BatchMeta(samples=merged_samples, extra_info=merged_extra_info)

    def reorder(self, indices: list[int]):
        """
        Reorder the SampleMeta in the BatchMeta according to the given indices.

        The operation is performed in-place, modifying the current BatchMeta's SampleMeta order.

        Args:
            indices : list[int]
                A list of integers specifying the new order of SampleMeta. Each integer
                represents the current index of the SampleMeta in the BatchMeta.
        """
        # Reorder the samples
        reordered_samples = [self.samples[i] for i in indices]
        object.__setattr__(self, "samples", reordered_samples)

        # Update necessary attributes
        self._update_after_reorder()

    def _update_after_reorder(self) -> None:
        """Update related attributes specifically for the reorder operation"""
        # Update batch_index for each sample
        for idx, sample in enumerate(self.samples):
            object.__setattr__(sample, "_batch_index", idx)

        # Update cached index lists
        object.__setattr__(self, "_global_indexes", [sample.global_index for sample in self.samples])
        object.__setattr__(self, "_local_indexes", [sample.local_index for sample in self.samples])
        object.__setattr__(self, "_storage_ids", [sample.storage_id for sample in self.samples])

        # Rebuild storage groups
        storage_meta_groups = self._build_storage_meta_groups()
        object.__setattr__(self, "_storage_meta_groups", storage_meta_groups)

        # Note: No need to update _size, _field_names, _is_ready, etc., as these remain unchanged after reorder

    @classmethod
    def from_samples(
        cls, samples: SampleMeta | list[SampleMeta], extra_info: Optional[dict[str, Any]] = None
    ) -> "BatchMeta":
        """
        Create a BatchMeta from a single SampleMeta or a list of SampleMeta objects.

        Args:
            samples: A single SampleMeta or a list of SampleMeta objects
            extra_info: Optional additional information to store with the batch

        Returns:
            BatchMeta instance containing the provided sample(s)

        Example:
            >>> sample_meta = SampleMeta(...)
            >>> batch_meta = BatchMeta.from_samples(sample_meta)

            >>> sample_metas = [sample1, sample2, sample3]
            >>> batch_meta = BatchMeta.from_samples(sample_metas, extra_info={"source": "training"})
        """
        if extra_info is None:
            extra_info = {}

        if isinstance(samples, SampleMeta):
            samples = [samples]

        return cls(samples=samples, extra_info=extra_info)

    @classmethod
    def empty(cls, extra_info: Optional[dict[str, Any]] = None) -> "BatchMeta":
        """
        Create an empty BatchMeta with no samples.

        Args:
            extra_info: Optional additional information to store with the batch

        Returns:
            Empty BatchMeta instance

        Example:
            >>> empty_batch = BatchMeta.empty()
        """
        if extra_info is None:
            extra_info = {}
        return cls(samples=[], extra_info=extra_info)


def _union_fields(fields1: dict[str, FieldMeta], fields2: dict[str, FieldMeta]) -> dict[str, FieldMeta]:
    """Union two sample's fields. If fields overlap, the fields in fields1 will be replaced by fields2."""
    for name in fields2.keys():
        fields1[name] = fields2[name]
    return fields1


def _extract_field_metas(tensor_dict: TensorDict, set_all_ready: bool = True) -> list[dict[str, FieldMeta]]:
    """
    Extract field metas from a TensorDict. If data in tensor_dict does not have dtype or shape attribute,
    the corresponding dtype or shape will be set to None.

    Args:
        tensor_dict (TensorDict): The input TensorDict.
        set_all_ready (bool): If True, set all production_status to READY_FOR_CONSUME.
                              Otherwise, set to NOT_PRODUCED. Default is True.

    Returns:
        all_fields (list[dict[FieldMeta]]): A list of dictionaries containing field metadata.
    """
    all_fields = []
    batch_size = tensor_dict.batch_size[0]
    for idx in range(batch_size):
        fields = {}
        sample = tensor_dict[idx]
        for name, value in sample.items():
            fields[name] = FieldMeta(
                name=name,
                dtype=value.dtype if hasattr(value, "dtype") else None,
                shape=value.shape if hasattr(value, "shape") else None,
                production_status=ProductionStatus.READY_FOR_CONSUME
                if set_all_ready
                else ProductionStatus.NOT_PRODUCED,
            )
        all_fields.append(fields)

    return all_fields
