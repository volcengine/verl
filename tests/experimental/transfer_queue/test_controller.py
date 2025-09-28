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

import numpy as np
import pytest
import ray
import torch

from verl.experimental.transfer_queue.controller import TQ_INIT_FIELD_NUM, TransferQueueController
from verl.experimental.transfer_queue.storage import TransferQueueStorageSimpleUnit

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@pytest.fixture(scope="function")
def ray_setup():
    if ray.is_initialized():
        ray.shutdown()
    ray.init(
        ignore_reinit_error=True,
        runtime_env={"env_vars": {"RAY_DEBUG": "1", "RAY_DEDUP_LOGS": "0"}},
        log_to_driver=True,
    )
    yield
    if ray.is_initialized():
        ray.shutdown()
        logger.info("Ray has been shut down completely after test")


@pytest.fixture(scope="function")
def setup_teardown_transfer_queue_controller(ray_setup):
    # Used as the offset for the global index to distinguish which global step the data corresponds to
    global_batch_size = 8
    num_global_batch = 2
    num_n_samples = 2
    num_data_storage_units = 2

    tq_controller = TransferQueueController.remote(
        num_storage_units=num_data_storage_units,
        global_batch_size=global_batch_size,
        num_global_batch=num_global_batch,
        num_n_samples=num_n_samples,
    )
    yield tq_controller, global_batch_size, num_global_batch, num_n_samples
    ray.get(tq_controller.clear.remote(0))


@pytest.fixture(scope="function")
def setup_teardown_register_controller_info(setup_teardown_transfer_queue_controller):
    tq_controller, global_batch_size, num_global_batch, num_n_samples = setup_teardown_transfer_queue_controller
    total_storage_size = global_batch_size * num_global_batch * num_n_samples
    num_data_storage_units = 2

    data_system_storage_units = {}
    for storage_unit_rank in range(num_data_storage_units):
        storage_node = TransferQueueStorageSimpleUnit.remote(
            storage_size=math.ceil(total_storage_size / num_data_storage_units)
        )
        data_system_storage_units[storage_unit_rank] = storage_node
        logger.info(f"TransferQueueStorageSimpleUnit #{storage_unit_rank} has been created.")

    # Register controller info
    zmq_server_info = ray.get(tq_controller.get_zmq_server_info.remote())
    controller_infos = {zmq_server_info.id: zmq_server_info}

    ray.get(
        [
            storage_unit.register_controller_info.remote(controller_infos)
            for storage_unit in data_system_storage_units.values()
        ]
    )

    yield tq_controller, global_batch_size, num_n_samples, data_system_storage_units


class TestTransferQueueController:
    @pytest.mark.parametrize("num_n_samples", [1, 2])
    @pytest.mark.parametrize("num_global_batch", [1, 2])
    def test_build_index_storage_mapping(self, num_n_samples, num_global_batch, ray_setup):
        # Used as the offset for the global index to distinguish which global step the data corresponds to
        global_batch_size = 8
        num_data_storage_units = 2

        self.tq_controller = TransferQueueController.remote(
            num_storage_units=num_data_storage_units,
            global_batch_size=global_batch_size,
            num_global_batch=num_global_batch,
            num_n_samples=num_n_samples,
        )

        global_index_storage_mapping, global_index_local_index_mapping = ray.get(
            self.tq_controller.get_global_index_mapping.remote()
        )

        if num_global_batch == 1 and num_n_samples == 1:
            assert np.array_equal(global_index_storage_mapping, np.array([0, 0, 0, 0, 1, 1, 1, 1]))
            assert np.array_equal(global_index_local_index_mapping, np.array([0, 1, 2, 3, 0, 1, 2, 3]))
        # The data of a single GBS will be distributed across different storage units
        elif num_global_batch == 2 and num_n_samples == 1:
            assert np.array_equal(
                global_index_storage_mapping, np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1])
            )
            assert np.array_equal(
                global_index_local_index_mapping, np.array([0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7])
            )
        # When num_n_samples is larger than 1
        elif num_global_batch == 1 and num_n_samples == 2:
            assert np.array_equal(
                global_index_storage_mapping, np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
            )
            assert np.array_equal(
                global_index_local_index_mapping, np.array([0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7])
            )
        elif num_global_batch == 2 and num_n_samples == 2:
            assert np.array_equal(
                global_index_storage_mapping,
                np.array(
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
                ),
            )
            assert np.array_equal(
                global_index_local_index_mapping,
                np.array(
                    [
                        0,
                        1,
                        2,
                        3,
                        4,
                        5,
                        6,
                        7,
                        0,
                        1,
                        2,
                        3,
                        4,
                        5,
                        6,
                        7,
                        8,
                        9,
                        10,
                        11,
                        12,
                        13,
                        14,
                        15,
                        8,
                        9,
                        10,
                        11,
                        12,
                        13,
                        14,
                        15,
                    ]
                ),
            )

    def test_update_production_status(self, setup_teardown_transfer_queue_controller):
        tq_controller, global_batch_size, num_global_batch, num_n_samples = setup_teardown_transfer_queue_controller

        total_storage_size = global_batch_size * num_global_batch * num_n_samples
        # Initialize get_data_production_status and filed_name_mapping
        init_update_production_status = torch.zeros(total_storage_size, TQ_INIT_FIELD_NUM, dtype=torch.int8)
        assert torch.equal(ray.get(tq_controller.get_data_production_status.remote()), init_update_production_status)
        assert ray.get(tq_controller.get_field_name_mapping.remote()) == {}

        columns_list = ["test_prompts"]
        global_indexes = list(range(global_batch_size * num_n_samples))

        # update production status
        tq_controller._update_production_status.remote(global_indexes, columns_list)
        new_field_name_mapping = ray.get(tq_controller.get_field_name_mapping.remote())
        assert new_field_name_mapping["test_prompts"] == 0

        new_data_production_status = ray.get(tq_controller.get_data_production_status.remote())
        assert new_data_production_status[:, 0][: len(global_indexes)].sum() == len(global_indexes)

    def test_data_consumption_status(self, setup_teardown_transfer_queue_controller):
        tq_controller, global_batch_size, num_global_batch, num_n_samples = setup_teardown_transfer_queue_controller
        total_storage_size = global_batch_size * num_global_batch * num_n_samples

        init_data_consumption_status = {}
        assert ray.get(tq_controller.get_data_consumption_status.remote()) == init_data_consumption_status

        task_name = "test_task1"
        ray.get(tq_controller._get_consumption_status.remote(task_name))
        new_data_consumption_status = ray.get(tq_controller.get_data_consumption_status.remote())
        assert torch.equal(new_data_consumption_status[task_name], torch.zeros(total_storage_size, dtype=torch.int8))

    def test_get_prompt_metadata(self, setup_teardown_register_controller_info):
        tq_controller, global_batch_size, n_samples, _ = setup_teardown_register_controller_info

        data_fields = ["test_prompts"]
        global_step = 5

        metadata = ray.get(
            tq_controller._get_metadata.remote(
                data_fields=data_fields,
                batch_size=global_batch_size * n_samples,
                global_step=global_step,
                mode="insert",
            )
        )
        metadata.reorder([15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
        assert metadata.global_indexes == [
            31,
            30,
            29,
            28,
            27,
            26,
            25,
            24,
            23,
            22,
            21,
            20,
            19,
            18,
            17,
            16,
        ]
        assert metadata.local_indexes == [
            15,
            14,
            13,
            12,
            11,
            10,
            9,
            8,
            15,
            14,
            13,
            12,
            11,
            10,
            9,
            8,
        ]
        storage_ids = metadata.storage_ids
        assert len(set(storage_ids[: len(storage_ids) // 2])) == 1

    # TODO: Test case where multiple clients concurrently read datameta from a single controller,
    #  and each client receives the correct response
