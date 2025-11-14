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
Metrics related to the RepExp trainer.
"""

from collections import defaultdict
from functools import partial
from typing import Any

import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.metric_utils import _compute_response_info, bootstrap_metric, calc_maj_val


def _compute_three_case_stats(data: DataProto, extrinsic_reward_tensor: torch.Tensor) -> dict:
    """
    Compute the fraction of samples that have no rollouts correct, some rollouts correct, and all rollouts correct.

    Args:
        data (DataProto): The data proto containing the batch data.
        extrinsic_reward_tensor (torch.Tensor): The extrinsic reward tensor.

    Returns:
        dict[str, float]: A dictionary containing the fraction of samples that have no rollouts correct,
        some rollouts correct, and all rollouts correct.
    """
    no_rollouts_correct = 0
    some_rollouts_correct = 0
    all_rollouts_correct = 0

    visited_uids = set()
    for uid in data.non_tensor_batch["uid"]:
        if uid in visited_uids:
            continue

        visited_uids.add(uid)
        mask = torch.from_numpy(data.non_tensor_batch["uid"] == uid)

        # Split into three cases
        if extrinsic_reward_tensor[mask].sum() == 0:
            no_rollouts_correct += 1
        elif extrinsic_reward_tensor[mask].sum() == mask.sum():
            all_rollouts_correct += 1
        elif extrinsic_reward_tensor[mask].sum() > 0 and extrinsic_reward_tensor[mask].sum() < mask.sum():
            some_rollouts_correct += 1
        else:
            raise ValueError(f"Invalid extrinsic reward tensor: {extrinsic_reward_tensor[mask].sum()}")

    # Sanity checks
    assert len(visited_uids) == no_rollouts_correct + some_rollouts_correct + all_rollouts_correct

    return {
        "no_rollouts_correct_frac": no_rollouts_correct / len(visited_uids),
        "some_rollouts_correct_frac": some_rollouts_correct / len(visited_uids),
        "all_rollouts_correct_frac": all_rollouts_correct / len(visited_uids),
    }


def compute_data_metrics(batch: DataProto, use_critic: bool = True, elliptical: bool = False) -> dict[str, Any]:
    """
    Computes various metrics from a batch of data for PPO training.

    This function calculates metrics related to scores, rewards, advantages, returns, values,
    and sequence lengths from a batch of data. It provides statistical information (mean, max, min)
    for each metric category.

    Args:
        batch: A DataProto object containing batch data with token-level scores, rewards, advantages, etc.
        use_critic: Whether to include critic-specific metrics. Defaults to True.
        elliptical: Whether to include elliptical-specific metrics. Defaults to False.

    Returns:
        A dictionary of metrics including:
            - critic/score/mean, max, min: Statistics about sequence scores
            - critic/rewards/mean, max, min: Statistics about sequence rewards
            - critic/advantages/mean, max, min: Statistics about advantages
            - critic/returns/mean, max, min: Statistics about returns
            - critic/values/mean, max, min: Statistics about critic values (if use_critic=True)
            - critic/vf_explained_var: Explained variance of the value function (if use_critic=True)
            - response_length/mean, max, min, clip_ratio: Statistics about response lengths
            - prompt_length/mean, max, min, clip_ratio: Statistics about prompt lengths
            - num_turns/mean, max, min: Statistics about the number of multi-turn conversations
    """
    sequence_score = batch.batch["token_level_scores"].sum(-1)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)

    if elliptical:
        sequence_intrinsic_reward = batch.non_tensor_batch["intrinsic_reward"].sum(-1)
        sequence_beta_scaled_intrinsic_reward = batch.non_tensor_batch["beta_scaled_intrinsic_reward"].sum(-1)
        sequence_extrinsic_reward = batch.non_tensor_batch["extrinsic_reward"].sum(-1)
        sequence_total_reward = batch.non_tensor_batch["total_reward"].sum(-1)
        sequence_raw_bonuses = batch.non_tensor_batch["raw_bonuses"].sum(-1)

        three_case_stats = _compute_three_case_stats(batch, batch.non_tensor_batch["extrinsic_reward"])

    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    max_response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-max_response_length].bool()
    response_mask = batch.batch["response_mask"].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info["prompt_length"]
    response_length = response_info["response_length"]

    aborted_mask = (response_length == 0).bool()
    non_aborted_mask = ~aborted_mask

    non_aborted_sequence_score = sequence_score[non_aborted_mask]
    non_aborted_sequence_reward = sequence_reward[non_aborted_mask]

    score_mean = torch.mean(non_aborted_sequence_score).detach().item()
    score_max = torch.max(non_aborted_sequence_score).detach().item()
    score_min = torch.min(non_aborted_sequence_score).detach().item()

    reward_mean = torch.mean(non_aborted_sequence_reward).detach().item()
    reward_max = torch.max(non_aborted_sequence_reward).detach().item()
    reward_min = torch.min(non_aborted_sequence_reward).detach().item()

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    # Aborted samples and non-aborted response length statistics
    # response_length_non_aborted/*: statistics computed on non-aborted samples only
    aborted_ratio = torch.mean(aborted_mask.float()).detach().item()

    non_aborted_response_length = response_length[non_aborted_mask]
    if non_aborted_response_length.numel() > 0:
        non_aborted_response_length_mean = torch.mean(non_aborted_response_length).detach().item()
        non_aborted_response_length_max = torch.max(non_aborted_response_length).detach().item()
        non_aborted_response_length_min = torch.min(non_aborted_response_length).detach().item()
        non_aborted_response_length_clip_ratio = (
            torch.mean(torch.eq(non_aborted_response_length, max_response_length).float()).detach().item()
        )
    else:
        raise ValueError("All samples are aborted, this should not happen.")

    metrics = {
        # score
        "critic/score/mean": score_mean,
        "critic/score/max": score_max,
        "critic/score/min": score_min,
        # reward
        "critic/rewards/mean": reward_mean,
        "critic/rewards/max": reward_max,
        "critic/rewards/min": reward_min,
        # adv
        "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
        "critic/advantages/max": torch.max(valid_adv).detach().item(),
        "critic/advantages/min": torch.min(valid_adv).detach().item(),
        # returns
        "critic/returns/mean": torch.mean(valid_returns).detach().item(),
        "critic/returns/max": torch.max(valid_returns).detach().item(),
        "critic/returns/min": torch.min(valid_returns).detach().item(),
        **(
            {
                # values
                "critic/values/mean": torch.mean(valid_values).detach().item(),
                "critic/values/max": torch.max(valid_values).detach().item(),
                "critic/values/min": torch.min(valid_values).detach().item(),
                # vf explained var
                "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
            }
            if use_critic
            else {}
        ),
        **(
            {
                # raw bonuses
                "critic/raw_bonuses/mean": np.mean(sequence_raw_bonuses).item(),
                "critic/raw_bonuses/max": np.max(sequence_raw_bonuses).item(),
                "critic/raw_bonuses/min": np.min(sequence_raw_bonuses).item(),
                "critic/raw_bonuses/std": np.std(sequence_raw_bonuses).item(),
                # intrinsic_reward
                "critic/intrinsic_reward/mean": np.mean(sequence_intrinsic_reward).item(),
                "critic/intrinsic_reward/max": np.max(sequence_intrinsic_reward).item(),
                "critic/intrinsic_reward/min": np.min(sequence_intrinsic_reward).item(),
                "critic/intrinsic_reward/std": np.std(sequence_intrinsic_reward).item(),
                # beta_scaled_intrinsic_reward
                "critic/beta_scaled_intrinsic_reward/mean": np.mean(sequence_beta_scaled_intrinsic_reward).item(),
                "critic/beta_scaled_intrinsic_reward/max": np.max(sequence_beta_scaled_intrinsic_reward).item(),
                "critic/beta_scaled_intrinsic_reward/min": np.min(sequence_beta_scaled_intrinsic_reward).item(),
                "critic/beta_scaled_intrinsic_reward/std": np.std(sequence_beta_scaled_intrinsic_reward).item(),
                # extrinsic_reward
                "critic/extrinsic_reward/mean": np.mean(sequence_extrinsic_reward).item(),
                "critic/extrinsic_reward/max": np.max(sequence_extrinsic_reward).item(),
                "critic/extrinsic_reward/min": np.min(sequence_extrinsic_reward).item(),
                "critic/extrinsic_reward/std": np.std(sequence_extrinsic_reward).item(),
                # three_case_stats
                "critic/extrinsic_reward/no_rollouts_correct_frac": three_case_stats["no_rollouts_correct_frac"],
                "critic/extrinsic_reward/some_rollouts_correct_frac": three_case_stats["some_rollouts_correct_frac"],
                "critic/extrinsic_reward/all_rollouts_correct_frac": three_case_stats["all_rollouts_correct_frac"],
                # total_reward
                "critic/total_reward/mean": np.mean(sequence_total_reward).item(),
                "critic/total_reward/max": np.max(sequence_total_reward).item(),
                "critic/total_reward/min": np.min(sequence_total_reward).item(),
                "critic/total_reward/std": np.std(sequence_total_reward).item(),
            }
            if elliptical
            else {}
        ),
        # response length
        "response_length/mean": torch.mean(response_length).detach().item(),
        "response_length/max": torch.max(response_length).detach().item(),
        "response_length/min": torch.min(response_length).detach().item(),
        "response_length/clip_ratio": torch.mean(torch.eq(response_length, max_response_length).float())
        .detach()
        .item(),
        # response length (non-aborted only)
        # These statistics exclude aborted samples to avoid skew from zeros
        "response_length_non_aborted/mean": non_aborted_response_length_mean,
        "response_length_non_aborted/max": non_aborted_response_length_max,
        "response_length_non_aborted/min": non_aborted_response_length_min,
        "response_length_non_aborted/clip_ratio": non_aborted_response_length_clip_ratio,
        # aborted ratio
        # Fraction of samples whose response length is zero
        "response/aborted_ratio": aborted_ratio,
        # prompt length
        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "prompt_length/max": torch.max(prompt_length).detach().item(),
        "prompt_length/min": torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio": torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }

    # multi-turn conversation
    if "__num_turns__" in batch.non_tensor_batch:
        num_turns = batch.non_tensor_batch["__num_turns__"]
        metrics["num_turns/min"] = num_turns.min()
        metrics["num_turns/max"] = num_turns.max()
        metrics["num_turns/mean"] = num_turns.mean()

    if "tool_call_counts" in batch.non_tensor_batch:
        tool_call_counts = batch.non_tensor_batch["tool_call_counts"]
        metrics["tool_call_counts/min"] = tool_call_counts.min()
        metrics["tool_call_counts/max"] = tool_call_counts.max()
        metrics["tool_call_counts/mean"] = tool_call_counts.mean()

    return metrics


def comb_estimator(n: int, c: int, k: int) -> float:
    """Calculates 1 - comb(n - c, k) / comb(n, k)."""
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def process_validation_metrics(
    data_sources: list[str], sample_uids: list[str], infos_dict: dict[str, list[Any]], seed: int = 42
) -> dict[str, dict[str, dict[str, float]]]:
    """
    Process validation metrics into a structured format with statistical analysis.

    This function organizes validation metrics by data source and prompt, then computes
    various statistical measures including means, standard deviations, best/worst values,
    and majority voting results. It also performs bootstrap sampling to estimate statistics
    for different sample sizes.

    Args:
        data_sources: List of data source identifiers for each sample.
        sample_uids: List of sample uids corresponding to each sample.
        infos_dict: Dictionary mapping variable names to lists of values for each sample.
        seed: Random seed for bootstrap sampling. Defaults to 42.

    Returns:
        A nested dictionary with the structure:
        {
            data_source: {
                variable_name: {
                    metric_name: value
                }
            }
        }

        Where metric_name includes:
        - "mean@N": Mean value across N samples
        - "std@N": Standard deviation across N samples
        - "best@N/mean": Mean of the best values in bootstrap samples of size N
        - "best@N/std": Standard deviation of the best values in bootstrap samples
        - "worst@N/mean": Mean of the worst values in bootstrap samples
        - "worst@N/std": Standard deviation of the worst values in bootstrap samples
        - "maj@N/mean": Mean of majority voting results in bootstrap samples (if "pred" exists)
        - "maj@N/std": Standard deviation of majority voting results (if "pred" exists)

    Example:
        >>> data_sources = ["source1", "source1", "source2"]
        >>> sample_uids = ["uid1", "uid1", "uid2"]
        >>> infos_dict = {"score": [0.8, 0.9, 0.7], "pred": ["A", "A", "B"]}
        >>> result = process_validation_metrics(data_sources, sample_uids, infos_dict)
        >>> # result will contain statistics for each data source and variable
    """
    # Group metrics by data source, prompt and variable
    data_src2uid2var2vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for sample_idx, data_source in enumerate(data_sources):
        uid = sample_uids[sample_idx]
        var2vals = data_src2uid2var2vals[data_source][uid]
        for var_name, var_vals in infos_dict.items():
            var2vals[var_name].append(var_vals[sample_idx])

    # Calculate metrics for each group
    data_src2uid2var2metric = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for data_source, uid2var2vals in data_src2uid2var2vals.items():
        for uid, var2vals in uid2var2vals.items():
            for var_name, var_vals in var2vals.items():
                if isinstance(var_vals[0], str):
                    continue

                metric = {}
                n_resps = len(var_vals)
                metric[f"mean@{n_resps}"] = np.mean(var_vals)
                metric["pass@1/mean"] = comb_estimator(n_resps, np.sum(var_vals), 1)

                if n_resps > 1:
                    metric[f"std@{n_resps}"] = np.std(var_vals)

                    ns = []
                    n = 2
                    while n < n_resps:
                        ns.append(n)
                        n *= 2
                    ns.append(n_resps)

                    for n in ns:
                        # [(bon_mean, bon_std), (won_mean, won_std)] = bootstrap_metric(
                        #     data=var_vals, subset_size=n, reduce_fns=[np.max, np.min], seed=seed
                        # )
                        # metric[f"best@{n}/mean"], metric[f"best@{n}/std"] = bon_mean, bon_std
                        # metric[f"worst@{n}/mean"], metric[f"worst@{n}/std"] = won_mean, won_std
                        metric[f"pass@{n}/mean"] = comb_estimator(n_resps, np.sum(var_vals), n)
                        if var2vals.get("pred", None) is not None:
                            vote_data = [
                                {"val": val, "pred": pred} for val, pred in zip(var_vals, var2vals["pred"], strict=True)
                            ]
                            [(maj_n_mean, maj_n_std)] = bootstrap_metric(
                                data=vote_data,
                                subset_size=n,
                                reduce_fns=[partial(calc_maj_val, vote_key="pred", val_key="val")],
                                seed=seed,
                            )
                            metric[f"maj@{n}/mean"], metric[f"maj@{n}/std"] = maj_n_mean, maj_n_std

                data_src2uid2var2metric[data_source][uid][var_name] = metric

    # Aggregate metrics across uids
    data_src2var2metric2uid_vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for data_source, uid2var2metric in data_src2uid2var2metric.items():
        for uid, var2metric in uid2var2metric.items():
            for var_name, metric in var2metric.items():
                for metric_name, metric_val in metric.items():
                    data_src2var2metric2uid_vals[data_source][var_name][metric_name].append(metric_val)

    data_src2var2metric2val = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for data_source, var2metric2uid_vals in data_src2var2metric2uid_vals.items():
        for var_name, metric2uid_vals in var2metric2uid_vals.items():
            for metric_name, uid_vals in metric2uid_vals.items():
                data_src2var2metric2val[data_source][var_name][metric_name] = np.mean(uid_vals)

    return data_src2var2metric2val
