#!env python3
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
This module provides utility classes and functions for training Peft Models with LoRA.

Key Features:
- Utility functions for initializing and configuring LoRA layers.
"""

import os


def load_peft_config(peft_local_path: str):
    """
    Load peft config from local path.
    Args:
        peft_local_path (str): The local path to the Peft model.

    Returns:
        PeftConfig: The Peft configuration.
    """
    from peft import PeftConfig
    from peft.utils.constants import CONFIG_NAME

    config_file = os.path.join(peft_local_path, CONFIG_NAME)
    # Expected, may raise error: FileNotFoundError, JSONDecodeError, TypeError etc.
    peft_config = PeftConfig.from_json_file(config_file)
    return peft_config


def load_peft_weights(peft_local_path: str):
    """
    Loads the Peft weights from the given local path.

    Args:
        peft_local_path (str): The local path where the Peft weights are stored.

    Returns:
        Dict: peft_state_dict.
    """
    import peft

    return peft.load_peft_weights(peft_local_path)


def set_model_peft_weights(model, peft_state_dict: dict, adapter_name: str = None):
    """
    Sets the Peft weights to the given model.

    Args:
        model: The model to which the Peft weights need to be set.
        peft_state_dict (Dict): The Peft weights to be set.
        adapter_name (str, optional): The name of the adapter. None means the default adapter.
    """
    from peft.utils import set_peft_model_state_dict

    if adapter_name is None:
        adapter_name = "default"

    processed_adapter_state_dict = {}
    prefix = "base_model.model."
    for key, value in peft_state_dict.items():
        new_key = key[len(prefix) :] if key.startswith(prefix) else key
        processed_adapter_state_dict[new_key] = value

    incompatible_keys = set_peft_model_state_dict(model, processed_adapter_state_dict, adapter_name)

    if incompatible_keys and len(getattr(incompatible_keys, "unexpected_keys", [])) > 0:
        raise Exception(f"Unexpected keys in the state dict: {list(sorted(incompatible_keys.unexpected_keys))}")
