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

from typing import Any

from transfer_queue.utils.zmq_utils import ZMQServerInfo

_TRANSFER_QUEUE_CONTROLLER_INFOS = None
_TRANSFER_QUEUE_STORAGE_INFOS = None


def set_transferqueue_server_info(controller_infos: dict[Any, ZMQServerInfo], storage_infos: dict[Any, ZMQServerInfo]):
    global _TRANSFER_QUEUE_CONTROLLER_INFOS, _TRANSFER_QUEUE_STORAGE_INFOS
    if _TRANSFER_QUEUE_CONTROLLER_INFOS is not None and _TRANSFER_QUEUE_STORAGE_INFOS is not None:
        return
    _TRANSFER_QUEUE_CONTROLLER_INFOS = controller_infos
    _TRANSFER_QUEUE_STORAGE_INFOS = storage_infos


def get_transferqueue_server_info():
    assert _TRANSFER_QUEUE_CONTROLLER_INFOS is not None and _TRANSFER_QUEUE_STORAGE_INFOS is not None, (
        "TransferQueue server infos have not been set yet."
    )
    return _TRANSFER_QUEUE_CONTROLLER_INFOS, _TRANSFER_QUEUE_STORAGE_INFOS
