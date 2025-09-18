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

from verl.utils.device import is_npu_available


def stateless_init_process_group(master_address, master_port, rank, world_size, device):
    """
    vLLM provides `StatelessProcessGroup` to create a process group
    without considering the global process group in torch.distributed.
    It is recommended to create `StatelessProcessGroup`, and then initialize
    the data-plane communication (NCCL) between external (train processes)
    and vLLM workers.
    """
    # NOTE: If it is necessary to support weight synchronization with the sglang backend in the future,
    # the following can be used:
    # from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator
    # from sglang.srt.distributed.utils import statelessprocessgroup
    if is_npu_available:
        from vllm_ascend.distributed.device_communicators.pyhccl import (
            PyHcclCommunicator as PyNcclCommunicator,
        )
    else:
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup

    pg = StatelessProcessGroup.create(host=master_address, port=master_port, rank=rank, world_size=world_size)
    pynccl = PyNcclCommunicator(pg, device=device)
    return pynccl
