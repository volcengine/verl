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

import logging
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def extract_solution(solution_str, method="strict"):
    assert method in ["strict", "flexible"]

    if method == "strict":
        # this also tests the formatting of the model
        solutions = re.findall("#### (\\-?[0-9\\.\\,]+)", solution_str)
        if len(solutions) == 0:
            final_answer = None
        else:
            # take the last solution
            final_answer = solutions[-1].replace(",", "").replace("$", "")
    elif method == "flexible":
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            # no reward is there is no answer
            pass
        else:
            invalid_str = ["", "."]
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    return final_answer


def compute_score(data_source, solution_str: str, ground_truth: str, extra_info=None, latency=40) -> float:
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual
    Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
        latency: the latency of the request
    """
    answer = extract_solution(solution_str=solution_str)
    sleep_time = random.uniform(1, latency)
    time.sleep(sleep_time)
    if answer is None:
        return 0
    else:
        if answer == ground_truth:
            return 1.0
        else:
            return 0.0


def compute_score_per_sample(data_source, solution_str: str, ground_truth: str, extra_info=None):
    score = compute_score(data_source, solution_str, ground_truth, extra_info)
    return score, solution_str, f"score: {score}"


def compute_batch_scores(data_sources, solution_strs, ground_truths, extra_infos=None):
    start = time.time()
    logging.error(f"solution_strs:{len(solution_strs)}")
    scores = [None] * len(solution_strs)
    with ThreadPoolExecutor(max_workers=256) as executor:
        futures = dict()
        for i in range(len(solution_strs)):
            data_source = data_sources[i]
            solution_str = solution_strs[i]
            ground_truth = ground_truths[i]
            extra_info = extra_infos[i]
            futures[executor.submit(compute_score, data_source, solution_str, ground_truth, extra_info)] = (
                i,
                time.time(),
            )
        for future in futures:
            i, _ = futures[future]
            try:
                scores[i] = future.result()
            except Exception as e:
                logger.error(f"Parallel processing failed for query {i}: {str(e)}")
    duration = time.time() - start
    logging.error(f"Reward api request costs {duration} seconds, start from {start}")
    return [dict(score=scores[i]) for i in range(len(scores))]


class Gsm8kAgent:
    def __init__(self):
        self.latency = 40

    def compute_score(
        self,
        data_source: Any,
        solution_str: str,
        ground_truth: str,
        extra_info: Optional[dict] = None,
    ) -> tuple[float, str, str]:
        score = compute_score(data_source, solution_str, ground_truth, extra_info, latency=self.latency)
        return score, solution_str, f"score: {score}"
