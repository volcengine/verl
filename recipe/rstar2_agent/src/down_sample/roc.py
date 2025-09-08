# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import re
import numpy as np
import torch
from pprint import pprint
from typing import List
from transformers import PreTrainedTokenizerFast

from verl.protocol import DataProto
from .utils import filter_by_mask, decode_prompt_response_str


def resample_of_correct(batch: DataProto, tokenizer: PreTrainedTokenizerFast, config: dict, do_sample=True, world_size=None):
    roc_error_ratio = config["roc_error_ratio"]
    roc_answer_format = config["roc_answer_format"]
    min_zero_reward_trace_num = config["min_zero_reward_trace_num"]
    min_non_zero_reward_trace_num = config["min_non_zero_reward_trace_num"]
    down_sample_to_n = config["down_sample_to_n"]
    assert min_zero_reward_trace_num + min_non_zero_reward_trace_num <= down_sample_to_n, \
        f"Invalid down sampling configuration: {min_zero_reward_trace_num=}, {min_non_zero_reward_trace_num=}, {down_sample_to_n=}"

    _, response_text = decode_prompt_response_str(batch, tokenizer)
    penalty_weights = np.zeros(len(response_text))
    metrics = {}

    # calculate error ratio penalty weight
    _penalty_weights, _metrics = calc_error_ratio_penalty_weights(response_text)
    metrics.update(_metrics)
    if roc_error_ratio:
        penalty_weights += _penalty_weights

    # calculate format penalty weight
    _penalty_weights, _metrics = calc_format_penalty_weights(response_text)
    metrics.update(_metrics)
    if roc_answer_format:
        penalty_weights += _penalty_weights

    # sample by penalty weights
    if do_sample and down_sample_to_n > 0:
        uids = batch.non_tensor_batch['uid']
        unique_uids = np.unique(uids)
        valid_mask = torch.zeros(len(uids), dtype=torch.bool)

        for uid in unique_uids:
            indices = np.where(uids == uid)[0]
            if len(indices) < down_sample_to_n:
                continue  # Not enough samples for this uid, skip
            if len(indices) == down_sample_to_n:
                valid_mask[indices] = True
                continue
            uid_mask = uids == uid
            uid_rewards = batch.batch['token_level_scores'][uid_mask].sum(-1)
            
            zero_reward_pairs = [(indice, penalty_weight) for indice, uid_reward, penalty_weight in zip(indices, uid_rewards, penalty_weights[uid_mask]) if uid_reward <= 0]
            non_zero_reward_pairs = [(indice, penalty_weight) for indice, uid_reward, penalty_weight in zip(indices, uid_rewards, penalty_weights[uid_mask]) if uid_reward > 0]
            non_zero_reward_pairs.sort(key=lambda x: x[1])
            zero_reward_trace_num = round(len(zero_reward_pairs) * down_sample_to_n / len(indices))
            non_zero_reward_trace_num = round(len(non_zero_reward_pairs) * down_sample_to_n / len(indices))
            if zero_reward_trace_num < min_zero_reward_trace_num and non_zero_reward_trace_num < min_non_zero_reward_trace_num:
                pprint(f"Total trace number before down sampling: {len(indices)}, smaller than {min_zero_reward_trace_num=} + {min_non_zero_reward_trace_num=}")
                valid_mask[indices] = True
            else:
                if zero_reward_trace_num <= min(min_zero_reward_trace_num, len(zero_reward_pairs)):
                    zero_reward_trace_num = min(min_zero_reward_trace_num, len(zero_reward_pairs))
                    non_zero_reward_trace_num = down_sample_to_n - zero_reward_trace_num
                if non_zero_reward_trace_num <= min(min_non_zero_reward_trace_num, len(non_zero_reward_pairs)):
                    non_zero_reward_trace_num = min(min_non_zero_reward_trace_num, len(non_zero_reward_pairs))
                    zero_reward_trace_num = down_sample_to_n - non_zero_reward_trace_num
                choices = [non_zero_reward_pair[0] for non_zero_reward_pair in non_zero_reward_pairs[:non_zero_reward_trace_num]] \
                    + [zero_reward_pair[0] for zero_reward_pair in zero_reward_pairs[:zero_reward_trace_num]]
                assert len(choices) == down_sample_to_n, f"{down_sample_to_n=} != {len(choices)}"
                valid_mask[choices] = True

        batch = filter_by_mask(batch, valid_mask, world_size)
    return batch, metrics


def calc_error_ratio_penalty_weights(response_text: List[str]):
    def error_ratio(text, pattern=r'<tool_response>.*?</tool_response>'):
        matches = re.findall(pattern, text, re.DOTALL)
        error_count = len([match for match in matches if 'error' in match.lower()])
        if len(matches) == 0:
            return 0.5, 0, 0
        else:
            return error_count / len(matches), error_count, len(matches)

    penalty_weights = []
    total_error_count, total_res_count = 0, 0

    for text in response_text:
        penalty_weight, error_count, res_count = error_ratio(text) 
        penalty_weights.append(penalty_weight)
        total_error_count += error_count
        total_res_count += res_count
    metrics = {
        'roc_error_ratio/global_err_ratio': total_error_count / total_res_count if total_res_count > 0 else 0,
        'roc_error_ratio/penalty_weight': np.mean(penalty_weights) if penalty_weights else 0,
    }
    return np.array(penalty_weights), metrics


def calc_format_penalty_weights(response_text: List[str]):
    def answer_tag_repetition(text: str, answer_tags=["<answer>", "</answer>"], answer_pattern=r'<answer>.*?</answer>', turn_pattern=r'<\|im_start\|>assistant.*?<\|im_end\|>'):
        if any(ans_tag not in text for ans_tag in answer_tags):
            return 1.0, 0

        answer_tags_count = [text.count(ans_tag) for ans_tag in answer_tags]
        closed_ans_tag_count = len(re.findall(answer_pattern, text, re.DOTALL))
        if any(ans_tag_count!=closed_ans_tag_count for ans_tag_count in answer_tags_count):
            return 1.0, closed_ans_tag_count

        matches = re.findall(turn_pattern, text, re.DOTALL)
        num_turns = len(matches)
        if num_turns == 0:
            return 1.0, closed_ans_tag_count

        penalty_weight = min((closed_ans_tag_count - 1) / num_turns, 1.0)
        return penalty_weight, closed_ans_tag_count

    penalty_weights = []
    total_ans_count, zero_ans_count, one_ans_count, gt_one_ans_count = 0, 0, 0, 0
    for text in response_text:
        penalty_weight, ans_tag_count = answer_tag_repetition(text)
        penalty_weights.append(penalty_weight)
        total_ans_count += ans_tag_count
        zero_ans_count += (1 if ans_tag_count == 0 else 0)
        one_ans_count += (1 if ans_tag_count == 1 else 0)
        gt_one_ans_count += (1 if ans_tag_count > 1 else 0)

    metrics = {
        'roc_answer_format/answer_per_rollout_mean': total_ans_count / len(response_text),
        'roc_answer_format/zero_answer_count': zero_ans_count,
        'roc_answer_format/one_answer_count': one_ans_count,
        'roc_answer_format/gt_one_answer_count': gt_one_ans_count,
    }
    return np.array(penalty_weights), metrics
