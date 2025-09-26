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
import sys
import time
import uuid
from pathlib import Path
from threading import Thread
from unittest.mock import MagicMock

import pytest
import ray
import tensordict
import torch
import zmq
from tensordict import TensorDict

# Import your classes here
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

try:
    from verl.experimental.transfer_queue.storage import TransferQueueStorageSimpleUnit
    from verl.experimental.transfer_queue.utils.zmq_utils import ZMQMessage, ZMQRequestType, ZMQServerInfo
except ImportError:
    # For testing purposes if imports are not available
    TransferQueueStorageSimpleUnit = MagicMock()
    ZMQServerInfo = MagicMock()
    ZMQRequestType = MagicMock()
    ZMQMessage = MagicMock()


# Mock ZMQ utilities if not available in test environment
def create_zmq_socket(context, socket_type, identity=None):
    sock = context.socket(socket_type)
    if identity:
        sock.setsockopt(zmq.IDENTITY, identity)
    return sock


# Mock Controller to handle handshake and data updates
class MockController:
    def __init__(self, controller_id="controller_001"):
        self.controller_id = controller_id
        self.context = zmq.Context()

        # Socket for handshake
        self.handshake_socket = self.context.socket(zmq.ROUTER)
        self.handshake_port = self._bind_to_random_port(self.handshake_socket)

        # Socket for data status updates
        self.data_update_socket = self.context.socket(zmq.ROUTER)
        self.data_update_port = self._bind_to_random_port(self.data_update_socket)

        self.zmq_server_info = ZMQServerInfo.create(
            role="CONTROLLER",
            id=controller_id,
            ip="127.0.0.1",
            ports={"handshake_socket": self.handshake_port, "data_status_update_socket": self.data_update_port},
        )

        self.running = True
        self.handshake_thread = Thread(target=self._handle_handshake, daemon=True)
        self.data_update_thread = Thread(target=self._handle_data_updates, daemon=True)
        self.handshake_thread.start()
        self.data_update_thread.start()

    def _bind_to_random_port(self, socket):
        port = socket.bind_to_random_port("tcp://127.0.0.1")
        return port

    def _handle_handshake(self):
        poller = zmq.Poller()
        poller.register(self.handshake_socket, zmq.POLLIN)

        while self.running:
            try:
                socks = dict(poller.poll(100))  # 100ms timeout
                if self.handshake_socket in socks:
                    identity, msg_bytes = self.handshake_socket.recv_multipart()
                    ZMQMessage.deserialize(msg_bytes)

                    # Send handshake ack
                    ack_msg = ZMQMessage.create(
                        request_type=ZMQRequestType.HANDSHAKE_ACK,
                        sender_id=self.controller_id,
                        body={"message": "Handshake successful"},
                    )
                    self.handshake_socket.send_multipart([identity, ack_msg.serialize()])
            except zmq.Again:
                continue
            except Exception:
                if self.running:
                    pass

    def _handle_data_updates(self):
        poller = zmq.Poller()
        poller.register(self.data_update_socket, zmq.POLLIN)

        while self.running:
            try:
                socks = dict(poller.poll(100))  # 100ms timeout
                if self.data_update_socket in socks:
                    identity, msg_bytes = self.data_update_socket.recv_multipart()
                    ZMQMessage.deserialize(msg_bytes)

                    # Send data update ack
                    ack_msg = ZMQMessage.create(
                        request_type=ZMQRequestType.NOTIFY_DATA_UPDATE_ACK,
                        sender_id=self.controller_id,
                        body={"message": "Data update received"},
                    )
                    self.data_update_socket.send_multipart([identity, ack_msg.serialize()])
            except zmq.Again:
                continue
            except Exception:
                if self.running:
                    pass

    def stop(self):
        self.running = False
        time.sleep(0.1)  # Give threads time to stop
        self.handshake_socket.close()
        self.data_update_socket.close()


# Mock client to send PUT/GET requests
class MockClient:
    def __init__(self, storage_put_get_address):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.DEALER)
        self.socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second timeout
        self.socket.connect(storage_put_get_address)

    def send_put(self, client_id, global_indexes, local_indexes, field_data):
        msg = ZMQMessage.create(
            request_type=ZMQRequestType.PUT_DATA,
            sender_id=f"mock_client_{client_id}",
            body={"global_indexes": global_indexes, "local_indexes": local_indexes, "field_data": field_data},
        )
        self.socket.send(msg.serialize())
        return ZMQMessage.deserialize(self.socket.recv())

    def send_get(self, client_id, local_indexes, fields):
        msg = ZMQMessage.create(
            request_type=ZMQRequestType.GET_DATA,
            sender_id=f"mock_client_{client_id}",
            body={"local_indexes": local_indexes, "fields": fields},
        )
        self.socket.send(msg.serialize())
        return ZMQMessage.deserialize(self.socket.recv())

    def close(self):
        self.socket.close()
        self.context.term()


@pytest.fixture(scope="session")
def ray_setup():
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


@pytest.fixture
def storage_setup(ray_setup):
    storage_size = 10000
    tensordict.set_list_to_stack(True).set()

    # Start mock controller
    mock_controller = MockController(f"controller_{uuid.uuid4()}")
    time.sleep(0.5)  # Wait for controller sockets to be ready

    # Start Ray actor
    storage_actor = TransferQueueStorageSimpleUnit.options(max_concurrency=50, num_cpus=1).remote(storage_size)

    # Register controller info
    controller_infos = {mock_controller.controller_id: mock_controller.zmq_server_info}
    ray.get(storage_actor.register_controller_info.remote(controller_infos))

    # Get ZMQ address to connect client
    zmq_info = ray.get(storage_actor.get_zmq_server_info.remote())
    put_get_address = zmq_info.to_addr("put_get_socket")
    time.sleep(1)  # Wait for socket to be ready

    yield storage_actor, put_get_address, mock_controller

    # Cleanup
    mock_controller.stop()


def test_put_get_single_client(storage_setup):
    """Test basic put and get operations with a single client using TensorDict and torch tensors."""
    _, put_get_address, _ = storage_setup

    client = MockClient(put_get_address)

    # PUT data
    global_indexes = [0, 1, 2]
    local_indexes = [0, 1, 2]
    field_data = TensorDict(
        {
            "log_probs": [torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0, 6.0]), torch.tensor([7.0, 8.0, 9.0])],
            "rewards": [torch.tensor([10.0]), torch.tensor([20.0]), torch.tensor([30.0])],
        },
        batch_size=[],
    )

    response = client.send_put(0, global_indexes, local_indexes, field_data)
    assert response.request_type == ZMQRequestType.PUT_DATA_RESPONSE

    # GET data
    response = client.send_get(0, [0, 1], ["log_probs", "rewards"])
    assert response.request_type == ZMQRequestType.GET_DATA_RESPONSE

    retrieved_data = response.body["data"]
    assert "log_probs" in retrieved_data
    assert "rewards" in retrieved_data
    assert retrieved_data["log_probs"].size(0) == 2
    assert retrieved_data["rewards"].size(0) == 2

    # Verify data correctness
    torch.testing.assert_close(retrieved_data["log_probs"][0], torch.tensor([1.0, 2.0, 3.0]))
    torch.testing.assert_close(retrieved_data["log_probs"][1], torch.tensor([4.0, 5.0, 6.0]))
    torch.testing.assert_close(retrieved_data["rewards"][0], torch.tensor([10.0]))
    torch.testing.assert_close(retrieved_data["rewards"][1], torch.tensor([20.0]))

    client.close()


def test_put_get_multiple_clients(storage_setup):
    """Test put and get operations with multiple clients including overlapping local indexes"""
    _, put_get_address, _ = storage_setup

    num_clients = 5
    clients = [MockClient(put_get_address) for _ in range(num_clients)]

    # Each client puts unique data using different local_indexes
    for i, client in enumerate(clients):
        global_indexes = [i * 10 + 0, i * 10 + 1, i * 10 + 2]
        local_indexes = [i * 10 + 0, i * 10 + 1, i * 10 + 2]
        field_data = TensorDict(
            {
                "log_probs": [
                    torch.tensor([i, i + 1, i + 2]),
                    torch.tensor([i + 3, i + 4, i + 5]),
                    torch.tensor([i + 6, i + 7, i + 8]),
                ],
                "rewards": [torch.tensor([i * 10]), torch.tensor([i * 10 + 10]), torch.tensor([i * 10 + 20])],
            }
        )

        response = client.send_put(i, global_indexes, local_indexes, field_data)
        assert response.request_type == ZMQRequestType.PUT_DATA_RESPONSE

    # Now simulate a third client that writes to overlapping local_indexes (e.g., index 0)
    overlapping_client = MockClient(put_get_address)
    overlap_local_indexes = [0]  # Overlaps with first client's index 0
    overlap_field_data = TensorDict({"log_probs": [torch.tensor([999, 999, 999])], "rewards": [torch.tensor([999])]})
    response = overlapping_client.send_put(
        client_id=99, global_indexes=[0], local_indexes=overlap_local_indexes, field_data=overlap_field_data
    )
    assert response.request_type == ZMQRequestType.PUT_DATA_RESPONSE

    # Each original client gets its own data (except for index 0 which was overwritten)
    for i, client in enumerate(clients):
        response = client.send_get(i, [i * 10 + 0, i * 10 + 1], ["log_probs", "rewards"])
        assert response.request_type == ZMQRequestType.GET_DATA_RESPONSE

        retrieved_data = response.body["data"]
        assert retrieved_data["log_probs"].size(0) == 2
        assert retrieved_data["rewards"].size(0) == 2

        # For index 0, expect data from overlapping_client; others from original client
        if i == 0:
            # Index 0 was overwritten
            torch.testing.assert_close(retrieved_data["log_probs"][0], torch.tensor([999, 999, 999]))
            torch.testing.assert_close(retrieved_data["rewards"][0], torch.tensor([999]))
            # Index 1 remains original
            torch.testing.assert_close(retrieved_data["log_probs"][1], torch.tensor([3, 4, 5]))
            torch.testing.assert_close(retrieved_data["rewards"][1], torch.tensor([10]))
        else:
            # All data remains original
            torch.testing.assert_close(retrieved_data["log_probs"][0], torch.tensor([i, i + 1, i + 2]))
            torch.testing.assert_close(retrieved_data["log_probs"][1], torch.tensor([i + 3, i + 4, i + 5]))
            torch.testing.assert_close(retrieved_data["rewards"][0], torch.tensor([i * 10]))
            torch.testing.assert_close(retrieved_data["rewards"][1], torch.tensor([i * 10 + 10]))

    # Cleanup
    for client in clients:
        client.close()
    overlapping_client.close()


def test_performance_basic(storage_setup):
    """Basic performance test with larger data volume and proper index handling"""
    _, put_get_address, _ = storage_setup

    client = MockClient(put_get_address)

    # PUT performance test
    put_latencies = []
    num_puts = 50
    batch_size = 128

    for i in range(num_puts):
        start = time.time()

        # Use larger batch size and more complex index mapping
        global_indexes = list(range(i * batch_size, (i + 1) * batch_size))
        local_indexes = list(range(i * batch_size, (i + 1) * batch_size))

        # Create larger tensor data to increase data volume
        log_probs_data = []
        rewards_data = []

        for j in range(batch_size):
            # Each sample contains larger tensors to increase data transfer volume
            log_probs_tensor = torch.randn(32768)
            rewards_tensor = torch.randn(32768)
            log_probs_data.append(log_probs_tensor)
            rewards_data.append(rewards_tensor)

        field_data = TensorDict({"log_probs": log_probs_data, "rewards": rewards_data}, batch_size=[batch_size])

        response = client.send_put(0, global_indexes, local_indexes, field_data)
        latency = time.time() - start
        put_latencies.append(latency)
        assert response.request_type == ZMQRequestType.PUT_DATA_RESPONSE

    # GET performance test
    get_latencies = []
    num_gets = 50

    for i in range(num_gets):
        start = time.time()
        # Retrieve larger batch of data
        indices = list(range(i * batch_size, (i + 1) * batch_size))  # Retrieve batch_size indices of data each time
        response = client.send_get(0, indices, ["log_probs", "rewards"])
        latency = time.time() - start
        get_latencies.append(latency)
        assert response.request_type == ZMQRequestType.GET_DATA_RESPONSE

    avg_put_latency = sum(put_latencies) / len(put_latencies) * 1000  # ms
    avg_get_latency = sum(get_latencies) / len(get_latencies) * 1000  # ms

    # Adjust performance thresholds to accommodate larger data volume
    assert avg_put_latency < 5000, f"Avg PUT latency {avg_put_latency}ms exceeds threshold"
    assert avg_get_latency < 5000, f"Avg GET latency {avg_get_latency}ms exceeds threshold"

    client.close()


def test_put_get_nested_tensor_single_client(storage_setup):
    """Test basic put and get operations with a single client using TensorDict and nested tensors."""
    _, put_get_address, _ = storage_setup

    client = MockClient(put_get_address)

    # PUT data
    global_indexes = [0, 1, 2]
    local_indexes = [0, 1, 2]

    field_data = TensorDict(
        {
            "variable_length_sequences": [
                torch.tensor([-0.5, -1.2, -0.8]),
                torch.tensor([-0.3, -1.5, -2.1, -0.9]),
                torch.tensor([-1.1, -0.7]),
            ],
            "attention_mask": [torch.tensor([1, 1, 1]), torch.tensor([1, 1, 1, 1]), torch.tensor([1, 1])],
        },
        batch_size=[],
    )

    response = client.send_put(0, global_indexes, local_indexes, field_data)
    assert response.request_type == ZMQRequestType.PUT_DATA_RESPONSE

    # GET data
    response = client.send_get(0, [0, 2], ["variable_length_sequences", "attention_mask"])
    assert response.request_type == ZMQRequestType.GET_DATA_RESPONSE

    retrieved_data = response.body["data"]
    assert "variable_length_sequences" in retrieved_data
    assert "attention_mask" in retrieved_data
    assert retrieved_data["variable_length_sequences"].size(0) == 2
    assert retrieved_data["attention_mask"].size(0) == 2

    # Verify data correctness
    torch.testing.assert_close(retrieved_data["variable_length_sequences"][0], torch.tensor([-0.5, -1.2, -0.8]))
    torch.testing.assert_close(retrieved_data["variable_length_sequences"][1], torch.tensor([-1.1, -0.7]))
    torch.testing.assert_close(retrieved_data["attention_mask"][0], torch.tensor([1, 1, 1]))
    torch.testing.assert_close(retrieved_data["attention_mask"][1], torch.tensor([1, 1]))

    client.close()


def test_put_get_nested_nontensor_single_client(storage_setup):
    """Test basic put and get operations with a single client using non-tensor data (strings)."""
    _, put_get_address, _ = storage_setup

    client = MockClient(put_get_address)

    # PUT data
    global_indexes = [0, 1, 2]
    local_indexes = [0, 1, 2]
    field_data = TensorDict(
        {
            "prompt_text": ["Hello world!", "This is a longer sentence for testing", "Test case"],
            "response_text": ["Hi there!", "This is the response to the longer sentence", "Test response"],
        },
        batch_size=[],
    )

    response = client.send_put(0, global_indexes, local_indexes, field_data)
    assert response.request_type == ZMQRequestType.PUT_DATA_RESPONSE

    # GET data
    response = client.send_get(0, [0, 1, 2], ["prompt_text", "response_text"])
    assert response.request_type == ZMQRequestType.GET_DATA_RESPONSE

    retrieved_data = response.body["data"]
    assert "prompt_text" in retrieved_data
    assert "response_text" in retrieved_data

    # Verify data correctness
    assert isinstance(retrieved_data["prompt_text"][0], str)
    assert isinstance(retrieved_data["response_text"][0], str)

    assert retrieved_data["prompt_text"][0] == "Hello world!"
    assert retrieved_data["prompt_text"][1] == "This is a longer sentence for testing"
    assert retrieved_data["prompt_text"][2] == "Test case"
    assert retrieved_data["response_text"][0] == "Hi there!"
    assert retrieved_data["response_text"][1] == "This is the response to the longer sentence"
    assert retrieved_data["response_text"][2] == "Test response"

    client.close()


def test_put_get_single_item_single_client(storage_setup):
    """Test put and get operations for a single item with a single client."""
    _, put_get_address, _ = storage_setup

    client = MockClient(put_get_address)

    # PUT data
    field_data = TensorDict(
        {
            "prompt_text": ["Hello world!"],
            "attention_mask": [torch.tensor([1, 1, 1])],
        },
        batch_size=[],
    )

    response = client.send_put(0, [0], [0], field_data)
    assert response.request_type == ZMQRequestType.PUT_DATA_RESPONSE

    # GET data
    response = client.send_get(0, [0], ["prompt_text", "attention_mask"])
    assert response.request_type == ZMQRequestType.GET_DATA_RESPONSE

    retrieved_data = response.body["data"]
    assert "prompt_text" in retrieved_data
    assert "attention_mask" in retrieved_data

    assert retrieved_data["prompt_text"][0] == "Hello world!"
    assert retrieved_data["attention_mask"].shape == (1, 3)
    torch.testing.assert_close(retrieved_data["attention_mask"][0], torch.tensor([1, 1, 1]))
