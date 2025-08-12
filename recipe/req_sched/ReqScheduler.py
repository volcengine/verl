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

import glob
import heapq
import json
import os

import numpy as np
import torch

from verl import DataProto
from verl.utils.seqlen_balancing import karmarkar_karp


def unpad_responses(padded_tensor, pad_token_id):
    if isinstance(pad_token_id, list):
        # from all worker
        if len(set(pad_token_id)) > 1:
            raise ValueError("pad_token_id is not the same across all workers")
        pad_token_id = pad_token_id[0]

    padded_tensor = padded_tensor.cpu()
    # Convert tensor to list if it's a tensor
    if isinstance(padded_tensor, torch.Tensor):
        padded_list = padded_tensor.tolist()
    else:
        padded_list = padded_tensor

    # Reconstruct original responses by removing padding tokens
    unpadded_responses = []
    for padded_response in padded_list:
        try:
            pad_start_idx = padded_response.index(pad_token_id)
            original_response = padded_response[:pad_start_idx]
        except ValueError:
            original_response = padded_response

        unpadded_responses.append(original_response)
    return unpadded_responses


class ReqScheduler:
    def __init__(self, config):
        self.config = config

        # prompt_ids -> len(reponse)
        self.table: dict[tuple[int], int] = self.load_table()

    def _aggregate_table(self, table: dict[tuple, list]) -> dict[tuple, int]:
        agg_method = self.config.get("agg", "mean")

        agg_functions = {
            "max": max,
            "min": min,
            "mean": lambda v: int(np.mean(v)),
            "median": lambda v: int(np.median(v)),
            "sum": sum,
        }

        if agg_method not in agg_functions:
            raise ValueError(f"Unknown agg method: {agg_method}")

        agg_func = agg_functions[agg_method]
        return {k: agg_func(v) for k, v in table.items()}

    def load_table(self):
        """
        {
            "prompts": [
                [prompt_token_ids_1],
                [prompt_token_ids_2],
                ...
            ],
            "lengths": [
                [120, 88, 85, 92, 95, 100, 90, 110],  // prompt 1(n samples)
                [105, 90, 95, 92, 100, 94, 90, 88],   // prompt 2
                ...
            ],
            "stats": [
                {"max": 120, "min": 85, "mean": 97.5, "std": 10.2, "sum": 780},
                {"max": 105, "min": 88, "mean": 94.3, "std": 5.6, "sum": 754},
                ...
            ]
        }
        """
        if self.config.seq_dir is None:
            return {}

        json_files = glob.glob(os.path.join(self.config.seq_dir, "*.json"))
        print(f"[ReqScheduler] Found {len(json_files)} JSON files to process")

        ans = {}
        for json_file in json_files:
            filename = os.path.basename(json_file)
            try:
                with open(json_file) as f:
                    data = json.load(f)

                print(f"[ReqScheduler] data keys = {data.keys()} in {filename}")

                ps = data["prompts"]
                ls = data["lengths"]
                for p, length in zip(ps, ls, strict=False):
                    p = tuple(p)
                    if p not in ans:
                        ans[p] = length
                print(f"[ReqScheduler] Processed {filename}, found {len(ans)} unique prompts")
            except Exception as e:
                print(f"[ReqScheduler] Error processing {filename}: {str(e)}")
                raise e

        ans = self._aggregate_table(ans)
        print(f"[ReqScheduler] Table-Size: {len(ans)=}")
        return ans

    def lookup_table(self, prompt):
        if isinstance(prompt, list):
            prompt = tuple(prompt)
        assert isinstance(prompt, tuple), f"prompt type {type(prompt)} is not supported"
        if prompt in self.table:
            return self.table[prompt]
        return None

    def update_table(self, raw_prompt_ids, responses):
        new_table = {}
        for p, r in zip(raw_prompt_ids, responses, strict=False):
            p = tuple(p)
            r = tuple(r)
            if p not in new_table:
                new_table[p] = []
            new_table[p].append(len(r))

        new_table = self._aggregate_table(new_table)

        for k, v in new_table.items():
            self.table[k] = v
        print(f"[ReqScheduler] in update_table, Table-Size: {len(self.table)=}")

    def log_seqlen(self, raw_prompt_ids, responses, prefix):
        print(
            f"[ReqScheduler] in log_seqlen, {type(raw_prompt_ids)},"
            f"{type(responses)}, {len(raw_prompt_ids)}, {len(responses)}"
        )
        assert len(raw_prompt_ids) == len(responses), f"{len(raw_prompt_ids)}, {len(responses)}"
        prompts_dict = {}
        for p, r in zip(raw_prompt_ids, responses, strict=True):
            if tuple(p) not in prompts_dict:
                prompts_dict[tuple(p)] = []
            prompts_dict[tuple(p)].append(len(r))

        prompts = list(prompts_dict.keys())
        response = list(prompts_dict.values())

        log_dir = self.config.log_dir
        os.makedirs(log_dir, exist_ok=True)
        data_files = glob.glob(f"{log_dir}/{prefix}_*.json")
        file_num = len(data_files) + 1
        output_file = f"{log_dir}/{prefix}_{file_num}.json"
        with open(output_file, "w") as f:
            json.dump({"prompts": prompts, "lengths": response}, f)

    def restore_order(
        self,
        gen_batch_output: DataProto,
        reqs_idx,
        n_samples,
    ):
        bs = len(gen_batch_output)
        assert bs % n_samples == 0, f"bs {bs} must be divisible by n_samples {n_samples}"
        assert bs // n_samples == len(reqs_idx), f"bs//n_samples {bs // n_samples} != len(reqs_idx) {len(reqs_idx)}"
        print(f"[ReqScheduler] restore_order, {bs=}, {n_samples=}, {len(reqs_idx)=}")
        expanded_reqs_idx = np.repeat(reqs_idx, n_samples)
        perm = np.argsort(expanded_reqs_idx, kind="stable")
        inv_perm = np.empty_like(perm)
        inv_perm[perm] = np.arange(bs)

        assert len(inv_perm) == bs, f"len(global_idx) {len(inv_perm)} != bs {bs}"

        global_idx = torch.tensor(inv_perm)
        gen_batch_output.reorder(global_idx)

    def sched(
        self,
        batch_dict: dict,
        world_size: int,
        config,
    ):
        print(f"[ReqScheduler] sched, {world_size=}, {config=}")

        pre_outlens = []
        for raw_prompt_ids in batch_dict["raw_prompt_ids"]:
            outlen = self.lookup_table(raw_prompt_ids)
            pre_outlens.append(outlen)

        # sched
        tp_size = config.rollout.tensor_model_parallel_size
        assert world_size % tp_size == 0, f"world_size {world_size} must be divisible by tp_size {tp_size}"
        dp_size = world_size // tp_size
        res = self._sched(pre_outlens, dp_size, tp_size)

        batch_dict["reqs_idx"] = res
        batch_dict["pre_outlens"] = np.array(pre_outlens, dtype=np.int32)

    def print_stats(self, outlens, res):
        longest = max(outlens)
        shortest = min(outlens)
        avg = np.mean(outlens)
        std = np.std(outlens)
        print(f"[ReqScheduler] Stats: {longest=}, {shortest=}, avg: {avg:.2f}, std: {std:.2f}")
        num_group = np.unique(res)
        group = [0 for _ in range(len(num_group))]
        for v in res:
            group[v] += 1
        print(f"[ReqScheduler] Group: {group}")

    def _sched(self, outlens, dp_size, tp_size):
        algo = self.config.algo

        has_none = False
        for outlen in outlens:
            if outlen is None:
                has_none = True
                break

        agg = self.config.get("agg", "mean")
        if has_none:
            print(f"[ReqScheduler] has None, reset {algo} to even_prompt; {agg=}")
            algo = "even_prompt"

            for i in range(len(outlens)):
                outlens[i] = -1
        else:
            print(f"[ReqScheduler] algo: {algo}, {agg=}")

        method = getattr(self, algo)
        res = method(outlens, dp_size)
        self.print_stats(outlens, res)
        return res

    def even_prompt(self, outlens: list[int], dp_size: int):
        num_prompts = len(outlens)
        if num_prompts == 0:
            return np.array([], dtype=np.int32)
        if dp_size <= 0:
            raise ValueError("dp_size must be a positive integer.")

        base_prompts_per_dp = num_prompts // dp_size
        remainder_prompts = num_prompts % dp_size
        res = []
        for i in range(dp_size):
            num_in_group = base_prompts_per_dp + (1 if i < remainder_prompts else 0)
            res.extend([i] * num_in_group)
        return np.array(res, dtype=np.int32)

    def even_token(self, outlens: list[int], dp_size: int):
        prompt_indices = list(range(len(outlens)))
        sorted_pairs = sorted(zip(outlens, prompt_indices, strict=False), reverse=True)
        heap = [(0, i) for i in range(dp_size)]
        heapq.heapify(heap)
        res = [None] * len(outlens)
        for token_len, orig_idx in sorted_pairs:
            total, group = heapq.heappop(heap)
            res[orig_idx] = group
            heapq.heappush(heap, (total + token_len, group))
        return np.array(res, dtype=np.int32)

    def even_token_kk(self, outlens: list[int], dp_size: int):
        """
        Schedules requests to balance the total number of tokens per DP group
        using the Karmarkar-Karp (KK) number partitioning algorithm.
        """
        if not outlens:
            return np.array([], dtype=np.int32)

        print(f"[ReqScheduler] Running Karmarkar-Karp for {len(outlens)} prompts into {dp_size} groups.")

        partitions = karmarkar_karp(
            seqlen_list=outlens,
            k_partitions=dp_size,
            equal_size=False,  # We want to balance sum of tokens, not count of prompts
        )

        res = [None] * len(outlens)
        for group_idx, partition_indices in enumerate(partitions):
            for original_prompt_idx in partition_indices:
                res[original_prompt_idx] = group_idx

        assert None not in res, "Karmarkar-Karp scheduling failed: not all prompts were assigned a group."

        return np.array(res, dtype=np.int32)
