# Copyright 2025 Individual Contributor: Thibaut Barroyer
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

import ray
import torch

from verl import DataProto
from verl.utils.length_penalty import apply_length_penalty


def get_custom_reward_fn(config):
    import importlib.util
    import sys

    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    if not file_path:
        return None

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    module = importlib.util.module_from_spec(spec)
    try:
        sys.modules["custom_module"] = module
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Error loading module from '{file_path}': {e}") from e

    function_name = reward_fn_config.get("name")
    if not hasattr(module, function_name):
        raise AttributeError(f"Reward function '{function_name}' not found in '{file_path}'.")

    print(f"using customized reward function '{function_name}' from '{file_path}'")
    raw_fn = getattr(module, function_name)

    reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))

    def wrapped_fn(*args, **kwargs):
        return raw_fn(*args, **kwargs, **reward_kwargs)

    return wrapped_fn


def load_reward_manager(config, tokenizer, num_examine, **reward_kwargs):
    reward_manager_name = config.reward_model.get("reward_manager", "naive")
    if reward_manager_name == "naive":
        from verl.workers.reward_manager import NaiveRewardManager

        reward_manager_cls = NaiveRewardManager
    elif reward_manager_name == "prime":
        from verl.workers.reward_manager import PrimeRewardManager

        reward_manager_cls = PrimeRewardManager
    elif reward_manager_name == "batch":
        from verl.workers.reward_manager import BatchRewardManager

        reward_manager_cls = BatchRewardManager
    elif reward_manager_name == "dapo":
        from verl.workers.reward_manager import DAPORewardManager

        reward_manager_cls = DAPORewardManager
    else:
        raise NotImplementedError

    # Pass length penalty config to reward manager
    length_penalty_config = dict(config.reward_model.get("length_penalty", {}))
    reward_kwargs.update({"length_penalty_config": length_penalty_config})

    compute_score = get_custom_reward_fn(config)
    return reward_manager_cls(
        tokenizer=tokenizer,
        num_examine=num_examine,
        compute_score=compute_score,
        reward_fn_key=config.data.reward_fn_key,
        **reward_kwargs,
    )


def compute_reward(data: DataProto, reward_fn):
    """
    Compute reward for a batch of data.
    Args:
        data: DataProto object containing the input data.
        reward_fn: Reward function to compute the reward.
    Returns:
        Tuple of reward tensor and extra info dictionary.
    """
    try:
        reward_result = reward_fn(data, return_dict=True)
        reward_tensor = reward_result["reward_tensor"]
        reward_extra_infos_dict = reward_result["reward_extra_info"]
    except Exception as e:
        print(f"Error in reward_fn: {e}")
        reward_tensor = reward_fn(data)
        reward_extra_infos_dict = {}
    
    # Apply length penalty if configured and if sequence lengths are available
    if hasattr(reward_fn, "length_penalty_config") and reward_fn.length_penalty_config.get("enabled", False):
        sequence_lengths = None
        if "response_lengths" in data:
            sequence_lengths = data["response_lengths"]
        elif "attention_mask" in data and "prompt_lengths" in data:
            # Calculate response length by subtracting prompt length from total length
            total_lengths = torch.sum(data["attention_mask"], dim=1)
            prompt_lengths = data["prompt_lengths"]
            sequence_lengths = total_lengths - prompt_lengths
        
        if sequence_lengths is not None:
            # Apply length penalty
            alpha = reward_fn.length_penalty_config.get("alpha", 0.0)
            min_length = reward_fn.length_penalty_config.get("min_length", 0)
            max_length = reward_fn.length_penalty_config.get("max_length", None)
            
            reward_tensor = apply_length_penalty(
                reward_tensor, 
                sequence_lengths,
                alpha=alpha,
                min_length=min_length,
                max_length=max_length
            )
            
            # Add length penalty info to extra_infos if returning dict
            if reward_extra_infos_dict is not None:
                reward_extra_infos_dict["length_penalty_applied"] = True
                reward_extra_infos_dict["length_penalty_alpha"] = alpha

    return reward_tensor, reward_extra_infos_dict


@ray.remote(num_cpus=1)
def compute_reward_async(data: DataProto, config, tokenizer):
    """
    Load the reward manager and compute the reward for a batch of data.
    This is meant to be run in a separate Ray worker.
    """
    reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {}))
    return compute_reward(data, reward_fn)