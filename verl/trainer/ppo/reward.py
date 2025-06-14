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

import math
import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

import ray
import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score


def get_custom_reward_fn(config):
    import importlib.util
    import sys

    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    if not file_path:
        return None

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

    module_name = os.path.splitext(os.path.basename(file_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
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

    compute_score = get_custom_reward_fn(config)
    final_compute_score = compute_score

    if compute_score is None:
        sandbox_config = config.reward_model.get("sandbox_fusion")
        sandbox_url = sandbox_config.get("url") if sandbox_config else None
        if sandbox_url:
            sandbox_manager = multiprocessing.Manager()
            _concurrent_semaphore = sandbox_manager.Semaphore(sandbox_config.get("max_concurrent", 64))
            final_compute_score = partial(default_compute_score, sandbox_fusion_url=sandbox_url, concurrent_semaphore=_concurrent_semaphore)
        else:
            final_compute_score = default_compute_score

    return reward_manager_cls(
        tokenizer=tokenizer,
        num_examine=num_examine,
        compute_score=final_compute_score,
        reward_fn_key=config.data.reward_fn_key,
        **reward_kwargs,
    )


def _compute_reward_chunk(chunk: DataProto, reward_fn):
    """
    Compute reward for a batch of data.
    Args:
         data: DataProto object containing the input data.
         reward_fn: Reward function to compute the reward.
     Returns:
         Tuple of reward tensor and extra info dictionary.
    """
    try:
        reward_result = reward_fn(chunk, return_dict=True)
        reward_tensor = reward_result["reward_tensor"]
        reward_extra_infos_dict = reward_result["reward_extra_info"]
    except Exception as e:
        print(f"Error in reward_fn: {e}")
        reward_tensor = reward_fn(chunk)
        reward_extra_infos_dict = {}

    return reward_tensor, reward_extra_infos_dict


def compute_reward(data: DataProto, reward_fn, num_workers: int) -> tuple:
    """
    Computes rewards for a batch of data by splitting it into chunks and processing them in parallel.

    This function supports multithreading.

    Args:
        data (DataProto): Input dataset that supports slicing and `len()`.
        reward_fn (Callable): A function that computes rewards for a batch of data.
            It should return either a tensor or a dictionary containing the reward information.
        num_workers (int, optional): Number of parallel workers (threads or processes).
            Set to 0 to run everything in the main process.

    Returns:
        tuple:
            - merged_reward (torch.Tensor | list): Concatenated or aggregated rewards from all chunks.
            - merged_info (dict): Merged auxiliary information returned by each chunk.
    """
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative, got {}".format(num_workers))
    elif num_workers == 0:
        return _compute_reward_chunk(data, reward_fn)  # If num_workers is 0, run in the main process

    total = len(data)
    if total == 0:
        return [], {}

    Executor = ThreadPoolExecutor

    # calculate the number of chunks
    num_chunks = min(num_workers, total)
    chunk_size = math.ceil(total / num_chunks)

    # Split the data into chunks
    chunks = [data[i : i + chunk_size] for i in range(0, total, chunk_size)]

    # Initialize rewards and extras
    rewards, extras = [None] * len(chunks), {}  # Initialize rewards as a list of None, and extras as an empty dict
    with Executor(max_workers=num_workers) as executor:
        futures = {executor.submit(_compute_reward_chunk, chunk, reward_fn): i for i, chunk in enumerate(chunks)}
        for fut in as_completed(futures):
            idx = futures[fut]  # get the index of the completed future
            tensor_chunk, info_chunk = fut.result()
            rewards[idx] = tensor_chunk  # put the result in the corresponding index
            extras.update(info_chunk)

    # merge reward_tensor
    first = rewards[0]
    if isinstance(first, list):
        merged_reward = sum(rewards, [])
    else:
        merged_reward = torch.cat(rewards, dim=0)

    return merged_reward, extras


@ray.remote(num_cpus=1)
def compute_reward_async(data: DataProto, config, tokenizer):
    """
    Load the reward manager and compute the reward for a batch of data.
    This is meant to be run in a separate Ray worker.
    """
    reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {}))
    return compute_reward(data, reward_fn, config.reward_model.reward_fn_workers)
