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

import json
from typing import Any

# magic numbers that ensure we are using the same LoRA adapter during the rollout and training process
VLLM_LORA_INT_ID = 123
VLLM_LORA_NAME = "123"
VLLM_LORA_PATH = "simon_lora_path"


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


def build_cli_args_from_config(config: dict[str, Any]) -> list[str]:
    """
    Convert a config dictionary to CLI arguments for vLLM server.

    Handles different value types appropriately:
    - None: skipped
    - bool True: adds '--key'
    - bool False: skipped
    - list: expands to '--key item1 item2 ...'
    - empty list: skipped (vLLM uses nargs="+" which requires at least one value)
    - dict: JSON serialized
    - other: string converted

    Args:
        config: Dictionary of configuration key-value pairs

    Returns:
        List of CLI argument strings
    """
    cli_args = []
    for k, v in config.items():
        if v is None:
            continue
        if isinstance(v, bool):
            if v:
                cli_args.append(f"--{k}")
        elif isinstance(v, list):
            if not v:
                # Skip empty lists - vLLM uses nargs="+" which requires at least one value
                continue
            # Lists need to be expanded as multiple separate arguments
            # e.g., --cuda-graph-sizes 1 2 4 8 becomes ['--cuda-graph-sizes', '1', '2', '4', '8']
            cli_args.append(f"--{k}")
            cli_args.extend([str(item) for item in v])
        else:
            cli_args.append(f"--{k}")
            # Use json.dumps for dict to ensure valid JSON format
            cli_args.append(json.dumps(v) if isinstance(v, dict) else str(v))
    return cli_args
