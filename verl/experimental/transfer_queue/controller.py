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

import logging
import math
import os
import threading
import time
from threading import Thread
from typing import Any, Optional
from uuid import uuid4

import numpy as np
import ray
import torch
import zmq
from ray.util import get_node_ip_address

from verl.experimental.transfer_queue.metadata import (
    BatchMeta,
    FieldMeta,
    SampleMeta,
)
from verl.experimental.transfer_queue.utils.utils import (
    ProductionStatus,
    TransferQueueRole,
    random_sampler,
)
from verl.experimental.transfer_queue.utils.zmq_utils import (
    ZMQMessage,
    ZMQRequestType,
    ZMQServerInfo,
    create_zmq_socket,
    get_free_port,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("TQ_LOGGING_LEVEL", logging.WARNING))

TQ_CONTROLLER_GET_METADATA_TIMEOUT = int(os.environ.get("TQ_CONTROLLER_GET_METADATA_TIMEOUT", 300))
TQ_CONTROLLER_GET_METADATA_CHECK_INTERVAL = int(os.environ.get("TQ_CONTROLLER_GET_METADATA_CHECK_INTERVAL", 1))
TQ_INIT_FIELD_NUM = int(os.environ.get("TQ_INIT_FIELD_NUM", 10))


@ray.remote(num_cpus=1)
class TransferQueueController:
    def __init__(
        self,
        num_storage_units: int,
        global_batch_size: int,
        num_global_batch: int = 1,
        num_n_samples: int = 1,
    ) -> None:
        """Initialize the TransferQueueController.

        Args:
            num_storage_units: Number of storage units in the system
            global_batch_size: Size of each global batch
            num_global_batch: Number of global batches to maintain in storage
            num_n_samples: For each prompt, sample n responses
        """
        self.controller_id = f"TQ_CONTROLLER_{uuid4()}"

        self._init_zmq_socket()  # Initialize ZMQ sockets for data communication

        self.num_storage_units = num_storage_units
        self.global_batch_size = (
            global_batch_size  # Used as offset for global index to identify corresponding global step
        )
        self.num_global_batch = num_global_batch
        self.num_n_samples = num_n_samples
        self.total_storage_size = self.global_batch_size * self.num_global_batch * self.num_n_samples

        self.data_production_status = torch.zeros(
            self.total_storage_size, TQ_INIT_FIELD_NUM, dtype=torch.int8
        )  # Initialize with default number of fields, dynamically extensible
        # task_name -> consumption_status mapping
        self.data_consumption_status: dict[str, torch.Tensor] = {}
        self.field_name_mapping: dict[
            str, int
        ] = {}  # Mapping table from field_name to the column indices in self.data_production_status tables
        # Per-sample dtype and shape storage: {global_index: {field_name: {'dtype': dtype, 'shape': shape}}}
        self.per_tensor_dtype_mapping: dict[int, dict[str, torch.dtype]] = {}
        self.per_tensor_shape_mapping: dict[int, dict[str, torch.Size]] = {}

        self._build_index_storage_mapping()

        self._start_process_handshake()
        self._start_process_update_data_status()
        self._start_process_request()

    def _get_consumption_status(self, task_name: str) -> torch.Tensor:
        """
        Get or create the consumption status tensor for a specific task.
        The consumption status is a binary, 1D tensor that records whether the corresponding sample has been consumed
        by the task.

        Args:
            task_name: Name of the consumer task

        Returns:
            Consumption status tensor for the specified task
        """
        # Retrieve or create the consumption state tensor for a specified consumer
        if task_name not in self.data_consumption_status:
            # Initialize state for a new consumer
            self.data_consumption_status[task_name] = torch.zeros(self.total_storage_size, dtype=torch.int8)
        return self.data_consumption_status[task_name]

    def _get_per_tensor_dtype(self, global_index: int, field_name: str) -> Optional[torch.dtype]:
        """Get dtype for a specific sample and field.

        Args:
            global_index: Global index of the sample
            field_name: Name of the field

        Returns:
            dtype of the specified field for the sample, or None if not found
        """
        return self.per_tensor_dtype_mapping.get(global_index, {}).get(field_name)

    def _get_per_tensor_shape(self, global_index: int, field_name: str) -> Optional[torch.Size]:
        """Get shape for a specific sample and field.

        Args:
            global_index: Global index of the sample
            field_name: Name of the field

        Returns:
            Shape of the specified field for the sample, or None if not found
        """
        return self.per_tensor_shape_mapping.get(global_index, {}).get(field_name)

    def _step_to_global_index_range(self, global_step: int) -> tuple[int, int]:
        """Convert global step to corresponding global index range.

        Args:
            global_step: The global step to convert

        Returns:
            Tuple of (start_index, end_index) for the given global step
        """
        start_idx = (global_step % self.num_global_batch) * self.global_batch_size * self.num_n_samples
        end_idx = start_idx + self.global_batch_size * self.num_n_samples

        return start_idx, end_idx

    def generate_data_status_mask(
        self, data_fields: list[str], global_step: int, task_name: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate mask matrix for filtering data based on field availability and consumption status.

        This function is called within _get_meta and generates a mask matrix based on
        user-specified fields and the current step. The mask matrix selects the required
        rows and columns from self.data_production_status while inversely selecting from
        self.data_consumption_status to support automated vectorization.

        Args:
            data_fields: List of field names to include in the mask
            global_step: Current global step for row selection
            task_name: Name of the consumer task for consumption status

        Returns:
            Tuple of (row_mask, col_mask) tensors for filtering data status matrices
        """

        # Check if all requested fields are registered
        for col in data_fields:
            if col not in self.field_name_mapping:
                # Return empty mask indicating no available data for unregistered columns
                empty_row_mask = torch.zeros(self.data_production_status.shape[0], dtype=torch.bool)
                empty_col_mask = torch.zeros(self.data_production_status.shape[1], dtype=torch.bool)
                return empty_row_mask, empty_col_mask

        # Map steps to global indices
        start_idx, end_idx = self._step_to_global_index_range(global_step)
        row_mask = torch.zeros(self.data_production_status.shape[0], dtype=torch.bool)
        row_mask[start_idx:end_idx] = True

        # Invert selection based on consumption status
        consumer_status = self._get_consumption_status(task_name)
        unconsumed_mask = consumer_status == 0
        row_mask &= unconsumed_mask

        # Select the specified fields
        col_mask = torch.zeros(self.data_production_status.shape[1], dtype=torch.bool)
        valid_fields = [self.field_name_mapping[col] for col in data_fields]
        if valid_fields:
            col_mask[valid_fields] = True

        return row_mask, col_mask

    def _build_index_storage_mapping(self):
        """
        Build mappings between global indices and storage locations.

        Distributes samples across storage units based on total storage space and
        maintains mappings between global index and local index within each storage.
        """
        # Assign each sample to a storage node. Here we scatter the samples in each GBS to different storage nodes
        # Samples are arranged sequentially, similar to generate_data_status_mask
        real_global_batch_size = self.global_batch_size * self.num_n_samples
        global_batch_per_storage_unit = math.ceil(real_global_batch_size / self.num_storage_units)

        # Build mapping between global index and storage unit for locating each data sample
        batch_storage_indices = np.repeat(np.arange(self.num_storage_units), global_batch_per_storage_unit)[
            :real_global_batch_size
        ]
        self._global_index_storage_rank_mapping = np.tile(batch_storage_indices, self.num_global_batch)

        # Build mapping between global index and local index within each storage unit
        indices = np.arange(self.total_storage_size)
        pos_in_batch = indices % real_global_batch_size
        g = indices // real_global_batch_size
        pos_in_block = pos_in_batch % global_batch_per_storage_unit
        self.global_index_local_index_mapping = g * global_batch_per_storage_unit + pos_in_block

    def get_data_production_status(self) -> torch.Tensor:
        """
        Get the current data production status matrix. The data production status is a 2D matrix that records whether
        the corresponding data is ready for each field of each sample.

        Returns:
            Tensor representing production status of all data fields
        """
        return self.data_production_status

    def get_field_name_mapping(self) -> dict[str, Any]:
        """Get the field name to column index mapping.

        Returns:
            Dictionary mapping field names to their column indices
        """
        return self.field_name_mapping

    def get_data_consumption_status(self) -> dict[str, torch.Tensor]:
        """Get consumption status for all tasks.

        Returns:
            Dictionary mapping task names to their consumption status tensors
        """
        return self.data_consumption_status

    def get_global_index_mapping(self):
        """Get global index to storage mapping information.

        Returns:
            Tuple containing storage rank mapping and local index mapping
        """
        return self._global_index_storage_rank_mapping, self.global_index_local_index_mapping

    def _get_metadata(
        self,
        data_fields: list[str],
        batch_size: int,
        global_step: int,
        mode: str = "fetch",
        task_name: str | None = None,
        get_n_samples=False,
        *args,
        **kwargs,
    ) -> BatchMeta:
        """
        Retrieve metadata with support for three modes.

        Args:
            data_fields: List of field names to include in metadata
            batch_size: Number of samples to retrieve
            global_step: Global step for which to retrieve metadata
            mode: Operation mode - 'insert', 'fetch', or 'force_fetch'
                - mode="insert": Insert metadata for new rows (without checking data status)
                - mode="fetch": Retrieve metadata for ready data (check data status and sample)
                - mode="force_fetch": Directly return metadata (without checking data status)
            task_name: Name of the consumer task (required for fetch modes)
            get_n_samples: Whether to retrieve n_samples as groups
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            BatchMeta object containing the requested metadata

        Raises:
            TimeoutError: If waiting for sufficient data times out in fetch mode
        """
        if mode == "insert":
            # TODO: Currently we only supports put the entire GBS data in one time
            assert batch_size == self.global_batch_size * self.num_n_samples, (
                f"batch_size {batch_size} must equal "
                f"global_batch_size * num_n_samples {self.global_batch_size * self.num_n_samples}"
            )
            start_idx, end_idx = self._step_to_global_index_range(global_step)
            batch_global_indexes = list(range(start_idx, end_idx))
            return self._generate_batch_meta(global_step, batch_global_indexes, data_fields, mode)

        assert task_name is not None
        if mode == "fetch":
            # Find consumable samples within current batch and package into BatchMeta when reading

            start_time = time.time()
            while True:
                ready_for_consume_idx = self._scan_data_status(data_fields, global_step, task_name, get_n_samples)

                if len(ready_for_consume_idx) >= batch_size:
                    break

                if time.time() - start_time > TQ_CONTROLLER_GET_METADATA_TIMEOUT:
                    raise TimeoutError(
                        f"Timeout while waiting for sufficient data. "
                        f"Required: {batch_size}, Available: {len(ready_for_consume_idx)}"
                    )

                logger.warning(
                    f"Insufficient data available. Required: {batch_size}, "
                    f"Available: {len(ready_for_consume_idx)}. Retrying in "
                    f"{TQ_CONTROLLER_GET_METADATA_CHECK_INTERVAL}s..."
                )
                time.sleep(TQ_CONTROLLER_GET_METADATA_CHECK_INTERVAL)
            logger.debug(f"ready for consume idx: {ready_for_consume_idx}")

            batch_global_indexes = random_sampler(ready_for_consume_idx, batch_size, get_n_samples, self.num_n_samples)
        elif mode == "force_fetch":
            start_idx, end_idx = self._step_to_global_index_range(global_step)
            consumer_status = self._get_consumption_status(task_name)
            not_consumed_idx = [i for i in range(start_idx, end_idx) if consumer_status[i] == 0]
            batch_global_indexes = random_sampler(not_consumed_idx, batch_size, get_n_samples, self.num_n_samples)

        # Mark this batch of data as consumed
        consumer_status = self._get_consumption_status(task_name)
        consumer_status[batch_global_indexes] = 1
        # Package into metadata
        metadata = self._generate_batch_meta(global_step, batch_global_indexes, data_fields, mode)
        logger.debug(f"_get_metadata: {metadata}")

        return metadata

    def _scan_data_status(
        self, data_fields: list[str], global_step: int, task_name: str, get_n_samples: bool
    ) -> list[int]:
        """
        Scan data status to find samples ready for consumption.

        Args:
            data_fields: List of field names to check
            global_step: Global step to scan
            task_name: Name of the consumer task
            get_n_samples: Whether to return n_samples as groups

        Returns:
            List of global indices that are ready for consumption
        """
        # Get row and column masks
        row_mask, col_mask = self.generate_data_status_mask(data_fields, global_step, task_name)
        logger.debug(f"row_mask, col_mask: {row_mask, col_mask}")

        if not row_mask.any() or not col_mask.any():
            return []

        # Extract subset of data status for relevant fields
        logger.debug(f"self.data_production_status: {self.data_production_status}")
        data_status_of_interest = self.data_production_status[:, col_mask]
        logger.debug(f"data_status_of_interest: {data_status_of_interest}")

        # Use torch.all for vectorized check instead of sum comparison
        all_fields_ready = torch.all(data_status_of_interest, dim=1)

        # Filter samples that meet criteria combined with row mask
        ready_mask = all_fields_ready & row_mask

        if get_n_samples and self.num_n_samples > 1:
            # Reshape to group view and check group completeness
            group_all_ready = torch.all(ready_mask.view(-1, self.num_n_samples), dim=1)

            # Get indices of fully ready groups
            ready_group_indices = group_all_ready.nonzero(as_tuple=False).flatten()

            # Calculate all sample indices
            sample_offset = torch.arange(self.num_n_samples)
            ready_for_consume_idx = (
                (ready_group_indices.unsqueeze(1) * self.num_n_samples + sample_offset).flatten().tolist()
            )

            return ready_for_consume_idx
        else:
            ready_for_consume_idx = torch.nonzero(ready_mask, as_tuple=False).flatten().tolist()
            logger.debug(f"ready_for_consume_idx: {ready_for_consume_idx}")

            return ready_for_consume_idx

    def _generate_batch_meta(
        self, global_step: int, global_indexes: list[int], data_fields: list[str], mode: str
    ) -> BatchMeta:
        """
        Generate BatchMeta by resolving storage locations for given global indexes.

        For each global index, looks up the corresponding storage node address using:
        - global_index_local_index_mapping: Maps to local index within storage
        - _global_index_storage_id_mapping: Maps to storage node identifier

        Args:
            global_step: Current global step
            global_indexes: List of global indexes to process
            data_fields: List of data field names
            mode: Operation mode ('fetch', 'insert', or 'force_fetch')

        Returns:
            BatchMeta object containing sample metadata with resolved storage locations
        """
        global_arr = np.array(global_indexes)
        storage_ids = self.global_index_storage_id_mapping[global_arr]
        local_indexes = self.global_index_local_index_mapping[global_arr]

        samples = []

        # Create samples from the flattened BatchMeta data
        # TODO: Optimize this
        for i, global_index in enumerate(global_indexes):
            local_index = local_indexes[i]
            storage_id = storage_ids[i]

            # Create FieldMeta objects for each field
            fields = []
            for field_name in data_fields:
                if mode == "fetch":
                    production_status = ProductionStatus.READY_FOR_CONSUME  # Since we filtered by ready status
                    # Get per-tensor dtype and shape for this specific global_index and field
                    dtype = self._get_per_tensor_dtype(global_index, field_name)
                    shape = self._get_per_tensor_shape(global_index, field_name)
                elif mode == "insert":
                    production_status = ProductionStatus.NOT_PRODUCED  # FIXME: not real-time
                    dtype = None
                    shape = None
                elif mode == "force_fetch":
                    col_index = self.field_name_mapping.get(field_name)
                    if col_index is not None and self.data_production_status[global_index, col_index] == 1:
                        production_status = ProductionStatus.READY_FOR_CONSUME
                        dtype = self._get_per_tensor_dtype(global_index, field_name)
                        shape = self._get_per_tensor_shape(global_index, field_name)
                    else:
                        production_status = ProductionStatus.NOT_PRODUCED
                        dtype = None
                        shape = None
                field_meta = FieldMeta(
                    name=field_name,
                    dtype=dtype,
                    shape=shape,
                    production_status=production_status,
                )
                fields.append(field_meta)

            sample = SampleMeta(
                global_step=global_step,
                global_index=global_index,
                storage_id=storage_id,
                local_index=local_index,
                fields={field.name: field for field in fields},
            )
            samples.append(sample)

        return BatchMeta(samples=samples)

    def _update_production_status(self, indexes: list[int], fields: list[str]) -> None:
        """
        Update production status for specified indexes and fields.

        Args:
            indexes: List of global indexes to update
            fields: List of field names to update
        """
        # TODO: Replace self.data_production_status == 0 or ==1 operations with ProductionStatus enum
        # Update data production status matrix
        new_fields = [field for field in fields if field not in self.field_name_mapping]
        if new_fields:
            needed_fields = len(new_fields)
            current_fields = self.data_production_status.shape[1]
            # Expand data status matrix if needed
            if len(self.field_name_mapping) + needed_fields > current_fields:
                add_fields = max(TQ_INIT_FIELD_NUM, needed_fields + 1)
                new_matrix = torch.zeros((self.total_storage_size, add_fields), dtype=torch.int8)
                self.data_production_status = torch.cat([self.data_production_status, new_matrix], dim=1)

        for field in fields:
            if field not in self.field_name_mapping.keys():
                self.field_name_mapping[field] = len(self.field_name_mapping)
        self.data_production_status[
            torch.tensor(indexes)[:, None], torch.tensor([self.field_name_mapping.get(field) for field in fields])
        ] = 1

    def _update_field_info(
        self,
        fields: list[str],
        per_tensor_dtypes: dict[int, dict[str, Any]],
        per_tensor_shapes: dict[int, dict[str, Any]],
        global_indexes: list[int],
    ) -> None:
        """
        Store per-tensor dtype and shape information.

        Args:
            fields: List of field names
            per_tensor_dtypes: Dict mapping global_index to field dtypes {global_index: {field: dtype}}
            per_tensor_shapes: Dict mapping global_index to field shapes {global_index: {field: shape}}
            global_indexes: List of global indexes corresponding to the samples
        """
        for global_idx in global_indexes:
            if global_idx not in self.per_tensor_dtype_mapping:
                self.per_tensor_dtype_mapping[global_idx] = {}
            if global_idx not in self.per_tensor_shape_mapping:
                self.per_tensor_shape_mapping[global_idx] = {}

            for field in fields:
                if global_idx in per_tensor_dtypes and field in per_tensor_dtypes[global_idx]:
                    self.per_tensor_dtype_mapping[global_idx][field] = per_tensor_dtypes[global_idx][field]
                if global_idx in per_tensor_shapes and field in per_tensor_shapes[global_idx]:
                    self.per_tensor_shape_mapping[global_idx][field] = per_tensor_shapes[global_idx][field]

    def _init_zmq_socket(self):
        """
        Initialize ZMQ sockets for communication.

        Sets up three ZMQ service ports for:
        1. Receiving handshake requests from storage
        2. Handling client data read/write requests
        3. Receiving status update signals from storage
        """
        self.zmq_context = zmq.Context()

        self._node_ip = get_node_ip_address()
        self._handshake_socket_port = get_free_port()
        self._request_handle_socket_port = get_free_port()
        self._data_status_update_socket_port = get_free_port()

        self.handshake_socket = create_zmq_socket(
            ctx=self.zmq_context,
            socket_type=zmq.ROUTER,
        )
        self.handshake_socket.bind(f"tcp://{self._node_ip}:{self._handshake_socket_port}")

        self.request_handle_socket = create_zmq_socket(
            ctx=self.zmq_context,
            socket_type=zmq.ROUTER,
        )
        self.request_handle_socket.bind(f"tcp://{self._node_ip}:{self._request_handle_socket_port}")

        self.data_status_update_socket = create_zmq_socket(
            ctx=self.zmq_context,
            socket_type=zmq.ROUTER,
        )
        self.data_status_update_socket.bind(f"tcp://{self._node_ip}:{self._data_status_update_socket_port}")

        self.zmq_server_info = ZMQServerInfo.create(
            role=TransferQueueRole.CONTROLLER,
            id=self.controller_id,
            ip=self._node_ip,
            ports={
                "handshake_socket": self._handshake_socket_port,
                "request_handle_socket": self._request_handle_socket_port,
                "data_status_update_socket": self._data_status_update_socket_port,
            },
        )

    def _wait_connection(self):
        """Wait for all storage instances to complete handshake.

        Clients don't need handshake to support dynamic scaling. Continuously
        listens for handshake messages until all expected storage units connect.
        """
        # TODO(zjj): Consider if retransmission is needed (assuming cases where Storage doesn't receive ACK)
        connected_storage_units = set()
        while len(connected_storage_units) < self.num_storage_units:
            identity, serialized_msg = self.handshake_socket.recv_multipart()
            request_msg = ZMQMessage.deserialize(serialized_msg)
            if request_msg.request_type == ZMQRequestType.HANDSHAKE:
                connected_storage_units.add(request_msg.sender_id)
                response_msg = ZMQMessage.create(
                    request_type=ZMQRequestType.HANDSHAKE_ACK,
                    sender_id=self.controller_id,
                    body={},
                ).serialize()
                self.handshake_socket.send_multipart([identity, response_msg])
                logger.info("Controller sent handshake ack successfully!")
        self.global_index_storage_id_mapping = np.array(sorted(list(connected_storage_units)))[
            self._global_index_storage_rank_mapping
        ]
        self.handshake_done.set()

    def _start_process_handshake(self):
        """Start the handshake process thread."""
        self.handshake_done = threading.Event()
        self.wait_connection_thread = Thread(
            target=self._wait_connection, name="TransferQueueControllerWaitConnectionThread", daemon=True
        )
        self.wait_connection_thread.start()

    def _start_process_update_data_status(self):
        """Start the data status update processing thread."""
        self.process_update_data_status_thread = Thread(
            target=self._update_data_status, name="TransferQueueControllerProcessUpdateDataStatusThread", daemon=True
        )
        self.process_update_data_status_thread.start()

    def _start_process_request(self):
        """Start the request processing thread."""
        self.process_request_thread = Thread(
            target=self._process_request, name="TransferQueueControllerProcessRequestThread", daemon=True
        )
        self.process_request_thread.start()

    def _process_request(self):
        """Main request processing loop.

        Handles various request types including metadata retrieval,
        consumption status checks, and clear operations.
        """
        self.handshake_done.wait()
        while True:
            # ROUTER socket receives multi-part messages
            identity, serialized_msg = self.request_handle_socket.recv_multipart()
            request_msg = ZMQMessage.deserialize(serialized_msg)

            if request_msg.request_type == ZMQRequestType.GET_META:
                params = request_msg.body
                logger.info("Controller preparing to get metadata...")
                metadata = self._get_metadata(
                    data_fields=params["data_fields"],
                    batch_size=params["batch_size"],
                    global_step=params["global_step"],
                    mode=params.get("mode", "fetch"),
                    task_name=params.get("task_name", None),
                    get_n_samples=params.get("get_n_samples", False),
                )
                response_msg = ZMQMessage.create(
                    request_type=ZMQRequestType.GET_META_RESPONSE,
                    sender_id=self.controller_id,
                    receiver_id=request_msg.sender_id,
                    body={"metadata": metadata},
                )
            elif request_msg.request_type == ZMQRequestType.GET_CLEAR_META:
                params = request_msg.body
                metadata = self._get_metadata(
                    data_fields=[],
                    batch_size=self.global_batch_size * self.num_n_samples,
                    global_step=params["global_step"],
                    mode="insert",
                )
                response_msg = ZMQMessage.create(
                    request_type=ZMQRequestType.GET_CLEAR_META_RESPONSE,
                    sender_id=self.controller_id,
                    receiver_id=request_msg.sender_id,
                    body={"metadata": metadata},
                )
            elif request_msg.request_type == ZMQRequestType.CLEAR_META:
                params = request_msg.body
                self.clear(global_step=params["global_step"])
                response_msg = ZMQMessage.create(
                    request_type=ZMQRequestType.CLEAR_META_RESPONSE,
                    sender_id=self.controller_id,
                    receiver_id=request_msg.sender_id,
                    body={"message": f"Clear operation completed by controller {self.controller_id}"},
                )
            elif request_msg.request_type == ZMQRequestType.CHECK_CONSUMPTION:
                # Check consumption status
                params = request_msg.body
                global_step = params["global_step"]

                consumer_status = self._get_consumption_status(params["task_name"])
                start_idx, end_idx = self._step_to_global_index_range(global_step)
                batch_status = consumer_status[start_idx:end_idx]
                consumed = torch.all(batch_status == 1).item()

                # Build response message
                response_msg = ZMQMessage.create(
                    request_type=ZMQRequestType.CONSUMPTION_RESPONSE,
                    sender_id=self.controller_id,
                    receiver_id=request_msg.sender_id,
                    body={
                        "global_step": global_step,
                        "consumed": consumed,
                    },
                )
            self.request_handle_socket.send_multipart([identity, response_msg.serialize()])
            logger.debug("Controller request_handle_socket sent multipart successfully!")

    def _update_data_status(self):
        """Process data status update messages from storage units.

        Continuously listens for data update notifications and updates
        internal production status and field information accordingly.
        """
        # Receive data status update information from storage
        while True:
            logger.debug("Preparing _update_data_status...")
            identity, serialized_msg = self.data_status_update_socket.recv_multipart()
            logger.debug("Controller received update_data_status request!")
            request_msg = ZMQMessage.deserialize(serialized_msg)
            logger.debug(f"[{self.controller_id}]: Controller received update_data_status request_msg: {request_msg}")

            if request_msg.request_type == ZMQRequestType.NOTIFY_DATA_UPDATE:
                message_data = request_msg.body

                fields = message_data.get("fields", [])
                global_indexes = message_data.get("global_indexes", [])
                per_tensor_dtypes = message_data.get("dtypes", {})  # Now a dict of lists
                per_tensor_shapes = message_data.get("shapes", {})  # Now a dict of lists
                # Update data production status
                logger.debug(f"global_indexes, fields: {global_indexes, fields}")
                self._update_production_status(global_indexes, fields)
                self._update_field_info(fields, per_tensor_dtypes, per_tensor_shapes, global_indexes)
                logger.info("Controller updated production status successfully!")

                # Send acknowledgment response
                response_msg = ZMQMessage.create(
                    request_type=ZMQRequestType.NOTIFY_DATA_UPDATE_ACK,
                    sender_id=self.controller_id,
                    body={
                        "controller_id": self.controller_id,
                        "message": f"Data update acknowledged from controller {self.controller_id}",
                    },
                )
                self.data_status_update_socket.send_multipart([identity, response_msg.serialize()])
                logger.info("Controller sent DATA_UPDATE_ACK successfully!")
            elif request_msg.request_type == ZMQRequestType.NOTIFY_DATA_UPDATE_ERROR:
                # Handle data update errors
                error_msg = request_msg.body.get("message", "Unknown error")
                logger.error(f"Data update error from storage: {error_msg}")

                # Send error acknowledgment response
                response_msg = ZMQMessage.create(
                    request_type=ZMQRequestType.NOTIFY_DATA_UPDATE_ACK,
                    sender_id=self.controller_id,
                    body={
                        "controller_id": self.controller_id,
                        "message": f"Error notification acknowledged from controller {self.controller_id}",
                    },
                )
                self.data_status_update_socket.send_multipart([identity, response_msg.serialize()])

    def get_zmq_server_info(self) -> ZMQServerInfo:
        """Get ZMQ server connection information.

        Returns:
            ZMQServerInfo object containing connection details
        """
        return self.zmq_server_info

    def clear(self, global_step: int):
        """Clear data for a specific global batch.

        Resets production and consumption status for all data in the specified
        global step. Currently only supports clearing single GBS at a time.

        Args:
            global_step: The global step to clear data for
        """
        start_idx, end_idx = self._step_to_global_index_range(global_step)

        self.data_production_status[start_idx:end_idx, :] = 0
        for task_name in self.data_consumption_status:
            self.data_consumption_status[task_name][start_idx:end_idx] = 0
