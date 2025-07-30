# Copyright 2024 Bytedance Ltd. and/or its affiliates
# The below code in this distribution has been modified by Tencent ("Tencent Modifications").
# All Tencent Modifications are Copyright (C) Tencent.
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

import datetime
import inspect
import os
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from queue import Queue

import numpy as np
import ray
import torch

from verl import DataProto


def get_custom_reward_fn(config):
    """Load and return a custom reward function from external file.

    Dynamically imports a reward function from a specified file path and wraps
    it with additional keyword arguments from the configuration.

    Args:
        config (dict): Configuration dictionary containing custom_reward_function
                      settings with 'path', 'name', and 'reward_kwargs' fields.

    Returns:
        callable or None: Wrapped reward function with merged kwargs, or None
                         if no custom reward function is configured.

    Raises:
        FileNotFoundError: If the specified reward function file doesn't exist.
        RuntimeError: If there's an error loading the module from file.
        AttributeError: If the specified function name isn't found in the module.
    """
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

    return raw_fn


@dataclass
class RewardRequest:
    """A data class representing a reward request.

    Attributes:
        group_dict (dict): A dictionary mapping group identifiers to their data and remaining group size.
        request_data (list): A list of individual reward requests.
        max_seq_len (int): The maximum sequence length for the request.
        group_size (int): The size of each group in the request.
    """

    group_dict: dict = field(default_factory=dict)  # 使用 default_factory
    request_data: list = field(default_factory=list)  # 使用 default_factory
    max_seq_len: int = 0
    group_size: int = 0


@ray.remote
class RayAsyncRewardAgent:
    def __init__(self, config, tokenizer):
        self.tokenizer = tokenizer
        self.max_concurrency = config.get("max_concurrency", 256)
        self.executor = ThreadPoolExecutor(max_workers=self.max_concurrency)
        self.pending_queue = Queue()
        self.completed_queue = Queue()
        self.running = True
        self.proxy_thread = threading.Thread(target=self.proxy_func)
        self.proxy_thread.start()
        self.reward_fn_key = config.data.reward_fn_key
        # Try to get a custom reward function based on the configuration
        compute_score = get_custom_reward_fn(config)
        if inspect.isclass(compute_score):
            self.agent = compute_score()
            self.user_defined_func = self.agent.compute_score
            print(f"Bind the agent: {self.agent.__class__.__name__}")
        else:
            self.agent = None
            self.user_defined_func = compute_score
            print(f"Bind the func: {compute_score.__name__}")

    def shutdown(self):
        self.running = False
        self.proxy_thread.join()
        self.executor.shutdown(wait=True)

    def get(self, chunk_size, hook_func=None, **kwargs):
        """Get rewards from the completed queue.

        Args:
            chunk_size (int): The number of rewards to retrieve.
            hook_func (function, optional): A callback function to process the retrieved rewards. Defaults to None.
            **kwargs: Additional keyword arguments to be passed to the callback function.

        Returns:
            tuple: A tuple containing the data indices, reward tensor, and reward extra infos dictionary.
        """
        start_ts = time.time()
        data_idxs, reward_extra_infos_dict = [], dict()
        print(f"wait for {chunk_size} rewards")
        rewards = []
        valid_response_lengths = []
        max_seq_len = 0

        while len(data_idxs) < chunk_size:
            if self.completed_queue.empty():
                time.sleep(0.1)
                continue
            data_idxs_, rewards_, valid_response_lengths_, max_seq_len_, reward_extra_info = self.completed_queue.get()
            data_idxs.extend(data_idxs_)
            if self.agent and hasattr(self.agent, "post_process_scores") and callable(self.agent.post_process_scores):
                rewards_ = self.agent.post_process_scores(rewards_)
            rewards.extend(rewards_)
            valid_response_lengths.extend(valid_response_lengths_)
            max_seq_len = max(max_seq_len, max_seq_len_)
            for k, v in reward_extra_info.items():
                if k not in reward_extra_infos_dict:
                    reward_extra_infos_dict[k] = np.array(v)
                else:
                    reward_extra_infos_dict[k] = np.concatenate([reward_extra_infos_dict[k], np.array(v)])

        index = np.argsort(data_idxs)
        data_idxs = np.sort(data_idxs).tolist()
        for k, v in reward_extra_infos_dict.items():
            reward_extra_infos_dict[k] = v[index]
        rewards = np.array(rewards)[index]

        reward_tensor = torch.zeros(chunk_size, max_seq_len, dtype=torch.float32)
        for i in range(len(rewards)):
            reward_tensor[i, valid_response_lengths[i] - 1] = rewards[i]
        end_ts = time.time()
        print(f"'get' starts from {start_ts}, end at {end_ts}, duration {end_ts - start_ts}")
        if hook_func:
            return hook_func(data_idxs, reward_tensor, reward_extra_infos_dict, **kwargs)

        return data_idxs, reward_tensor, reward_extra_infos_dict

    def proxy_func(self):
        while self.running:
            if not self.pending_queue.empty():
                start = time.time()
                request: RewardRequest = self.pending_queue.get()
                print(
                    f"Total {len(request.request_data)} reward requests, "
                    f"{len(request.group_dict)} groups, "
                    f"group size = {request.group_size}"
                )
                futures = []
                timestamps, queries, results, latencies, group_uids = [], [], [], [], []
                for (
                    data_source,
                    response_str,
                    ground_truth,
                    extra_info,
                    group_uid,
                    data_idx,
                    valid_response_length,
                ) in request.request_data:
                    future = self.executor.submit(
                        self.user_defined_func, data_source, response_str, ground_truth, extra_info
                    )
                    future.meta_info = [group_uid, data_idx, valid_response_length, time.time(), response_str]
                    futures.append(future)

                for future in as_completed(futures):
                    score, query, response = future.result()
                    end_time = time.time()
                    index, intra_data_index, valid_response_length, start_time, response_str = future.meta_info

                    if index not in request.group_dict:
                        print(f"Warning: index {index} not in request.group_dict, add it in func: proxy_func")
                        request.group_dict[index] = [dict(), request.group_size]

                    request.group_dict[index][0][intra_data_index] = (score, valid_response_length)
                    request.group_dict[index][1] -= 1
                    timestamps.append(datetime.datetime.now().isoformat())
                    queries.append(query)
                    results.append(response)
                    latencies.append(end_time - start_time)
                    group_uids.append(index)

                    if request.group_dict[index][1] == 0:
                        sorted_dict = dict(sorted(request.group_dict[index][0].items()))
                        rewards = []
                        valid_response_lengths = []
                        data_idxs = []
                        reward_extra_info = dict()
                        for idx, (reward, length) in sorted_dict.items():
                            if isinstance(reward, dict):
                                rewards.append(reward["score"])
                                for key, value in reward.items():
                                    if key not in reward_extra_info:
                                        reward_extra_info[key] = []
                                    reward_extra_info[key].append(value)
                            else:
                                rewards.append(reward)
                            valid_response_lengths.append(length)
                            data_idxs.append(idx)
                        self.completed_queue.put(
                            (data_idxs, rewards, valid_response_lengths, request.max_seq_len, reward_extra_info)
                        )
                        del request.group_dict[index]

                dur = time.time() - start
                print(f"Requesting the reward took {dur} seconds")
                if self.agent and hasattr(self.agent, "log") and callable(self.agent.log):
                    self.agent.log(timestamps, queries, results, latencies, group_uids)
            else:
                time.sleep(1)

    def compute_reward_pipeline(self, data: DataProto, group_size=1):
        """Process a batch of data through the asynchronous reward agent.

        Args:
            data (DataProto): The input data batch to process.
            group_size (int, optional): The size of each group in the batch. Defaults to 1.

        Returns:
            int: The number of processed data items.

        Raises:
            AssertionError: If the number of data items is not divisible by `group_size`.
        """
        num = len(data)
        assert num % group_size == 0, "The number of data items must be divisible by group_size"
        index = data.non_tensor_batch["uid"]
        max_seq_len = data.batch["responses"].shape[-1]
        request: RewardRequest = RewardRequest(max_seq_len=max_seq_len, group_size=group_size)
        for data_idx in range(len(data)):
            group_uid = index[data_idx]
            data_item = data[data_idx]  # DataProtoItem
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            # decode
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", dict())

            if group_uid not in request.group_dict:
                request.group_dict[group_uid] = [dict(), group_size]

            request.request_data.append(
                [data_source, response_str, ground_truth, extra_info, group_uid, data_idx, valid_response_length]
            )
        self.pending_queue.put(request)
        return num

    def verify(self, data):
        prompt_ids = data.batch["prompts"]
        response_ids = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]

        prompt_len = prompt_ids.shape[-1]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

        responses_str = []
        for i in range(len(data)):
            valid_len = valid_response_lengths[i]
            valid_response_ids = response_ids[i][:valid_len]
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            responses_str.append(response_str)

        ground_truths = [item.non_tensor_batch["reward_model"].get("ground_truth", None) for item in data]
        data_sources = data.non_tensor_batch[self.reward_fn_key]
        extras = data.non_tensor_batch["extra_info"]
        groups = defaultdict(dict)
        scores = [None] * len(responses_str)
        timestamps, queries, results, latencies, group_uids = [], [], [], [], []
        futures = []
        for idx in range(len(responses_str)):
            future = self.executor.submit(
                self.user_defined_func, data_sources[idx], responses_str[idx], ground_truths[idx], extras[idx]
            )
            future.meta_info = [idx, time.time(), responses_str[idx]]
            futures.append(future)
        for future in futures:
            score, query, response = future.result()
            idx, start_time, response_str = future.meta_info
            uid = extras[idx]["group_uid"]
            timestamps.append(datetime.datetime.now().isoformat())
            queries.append(query)
            results.append(response)
            latencies.append(time.time() - start_time)
            group_uids.append(uid)
            groups[uid][idx] = score
            if len(groups[uid]) == extras[idx]["group_size"]:
                values = list(groups[uid].values())
                if (
                    self.agent
                    and hasattr(self.agent, "post_process_scores")
                    and callable(self.agent.post_process_scores)
                ):
                    values = self.agent.post_process_scores(values)
                for i, k in enumerate(groups[uid].keys()):
                    scores[k] = values[i]

        if self.agent and hasattr(self.agent, "log") and callable(self.agent.log):
            self.agent.log(timestamps, queries, results, latencies, group_uids)
        return scores

    def compute_reward(self, data: DataProto, return_dict=False):
        """Compute rewards for a batch of data.

        Args:
            data (DataProto): The input data batch to process.
            return_dict (bool, optional): Whether to return results as a dictionary. Defaults to False.

        Returns:
            torch.Tensor or dict: The computed reward tensor or a dictionary
            containing the reward tensor and extra info.

        Notes:
            If `rm_scores` is already present in the data batch, it will be returned directly.
            Otherwise, rewards are computed via the reward model function.
        """
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]
        print(f"Total {len(data)} times reward API requests")
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        prompt_ids = data.batch["prompts"]
        prompt_len = prompt_ids.shape[-1]
        attention_mask = data.batch["attention_mask"]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

        scores = self.verify(data)
        rewards = []

        for i in range(len(data)):
            length = valid_response_lengths[i].item()
            score = scores[i]

            if isinstance(score, dict):
                reward = score["score"]
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            rewards.append(reward)
            reward_tensor[i, length - 1] = reward

        data.batch["acc"] = torch.tensor(rewards, dtype=torch.float32, device=prompt_ids.device)

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            return reward_tensor
