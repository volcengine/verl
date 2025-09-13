# Copyright 2025 Bytedance Ltd.and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance wit the License
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#

import re
from mathruler.grader import extract_boxed_content

from verl.utils.reward_score.prime_math import compute_score as compute_score_math


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    extract_output = extract_boxed_content(solution_str)
    if ground_truth in solution_str or extract_output == ground_truth:
        return 1.0
    
    original_is_correct, original_format_correctness, extracted_model_output = compute_score_math(
        extract_output, ground_truth
    )
    if original_is_correct:
        return 1.0
    
    # process answers like r'\text{odd}'
    cleaned_truth = ground_truth.strip()
    cleaned_model = extracted_model_output.strip()
    text_pattern = r'^\\text\s*{([^}]+)}$'
    match = re.match(text_pattern, cleaned_model)
    if match:
        text_content = match.group(1).strip()
        if text_content == cleaned_truth:
            return 1.0
    
    return 0.0
