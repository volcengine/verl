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

import os

PPO_RAY_RUNTIME_ENV = {
    "env_vars": {
        "TOKENIZERS_PARALLELISM": "true",
        "NCCL_DEBUG": "WARN",
        "VLLM_LOGGING_LEVEL": "WARN",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "HCCL_HOST_SOCKET_PORT_RANGE": "60000-60050",
        "HCCL_NPU_SOCKET_PORT_RANGE": "61000-61050",
        # "ASCEND_GLOBAL_LOG_LEVEL": "0",
        "VERL_PPO_LOGGING_LEVEL": "DEBUG",
        "VERL_LOGGING_LEVEL": "DEBUG",
        # "ASCEND_HOST_LOG_FILE_NUM":"1000",
        # "OOM_SNAPSHOT_ENABLE": "1",
        # "OOM_SNAPSHOT_PATH": "/home/l00878165/sglang/repos/logs/snapshot/launch",
        # "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "HCCL_CONNECT_TIMEOUT": "1500",
        "HCCL_EXEC_TIMEOUT": "1500",
    },
}


def get_ppo_ray_runtime_env():
    """
    A filter function to return the PPO Ray runtime environment.
    To avoid repeat of some environment variables that are already set.
    """
    runtime_env = {"env_vars": PPO_RAY_RUNTIME_ENV["env_vars"].copy()}
    for key in list(runtime_env["env_vars"].keys()):
        # if os.environ.get(key) is not None:
        #     runtime_env["env_vars"].pop(key, None)
        pass
    return runtime_env
