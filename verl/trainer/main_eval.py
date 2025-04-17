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
Offline evaluate the performance of a generated file using reward model and ground truth verifier.
The input is a parquet file that contains N generated sequences and (optional) the ground truth.

"""

import hydra
from verl.utils.fs import copy_to_local
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import ray


def get_custom_reward_fn(config):
    import importlib.util, os, sys
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
        raise RuntimeError(f"Error loading module from '{file_path}': {e}")

    function_name = reward_fn_config.get("name")
    if not hasattr(module, function_name):
        raise AttributeError(f"Reward function '{function_name}' not found in '{file_path}'.")

    print(f"using customized reward function '{function_name}' from '{file_path}'")
    raw_fn = getattr(module, function_name)

    reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))

    def wrapped_fn(*args, **kwargs):
        return raw_fn(*args, **kwargs, **reward_kwargs)

    return wrapped_fn


@ray.remote
def process_item(reward_fn, data_source, response_lst, reward_data):
    ground_truth = reward_data['ground_truth']
    score_lst = [reward_fn(data_source, r, ground_truth) for r in response_lst]
    return data_source, np.mean(score_lst)


@hydra.main(config_path='config', config_name='evaluation', version_base=None)
def main(config):
    local_path = copy_to_local(config.data.path)
    dataset = pd.read_parquet(local_path)
    prompts = dataset[config.data.prompt_key]
    responses = dataset[config.data.response_key]
    data_sources = dataset[config.data.data_source_key]
    reward_model_data = dataset[config.data.reward_model_key]

    total = len(dataset)

    # Initialize Ray
    if not ray.is_initialized():
        ray.init()

    # evaluate test_score based on data source
    data_source_reward = defaultdict(list)
    compute_score = get_custom_reward_fn(config)

    # Create remote tasks
    remote_tasks = [
        process_item.remote(compute_score, data_sources[i], responses[i], reward_model_data[i]) for i in range(total)
    ]

    # Process results as they come in
    with tqdm(total=total) as pbar:
        while len(remote_tasks) > 0:
            # Use ray.wait to get completed tasks
            done_ids, remote_tasks = ray.wait(remote_tasks)
            for result_id in done_ids:
                data_source, score = ray.get(result_id)
                data_source_reward[data_source].append(score)
                pbar.update(1)

    metric_dict = {}
    for data_source, rewards in data_source_reward.items():
        metric_dict[f'test_score/{data_source}'] = np.mean(rewards)

    print(metric_dict)


if __name__ == '__main__':
    main()
