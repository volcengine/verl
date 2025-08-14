# Copyright 2025 Bytedance Ltd.and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance wit the License
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#

from mathruler.grader import extract_boxed_content, grade_answer

def extract_solution(solution_str, ground_truth):
    extract_output = extract_boxed_content(solution_str)
    if ground_truth in solution_str or extract_output == ground_truth:
        return 1.0
    return 0.0

def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """The scoring function for deepscaler.

    Reference: Trung, Luong
    Args:
        solution_str (str): the solution text
        ground_truth (str): the ground truth
        method: the method to use to compute the score. Defaults to "strict".
        format_score: the format of the score. Defaults to 0.0.
        score: the score. Defaults to 1.0.
    """
    score = extract_solution(solution_str=solution_str, ground_truth=ground_truth)
    return score