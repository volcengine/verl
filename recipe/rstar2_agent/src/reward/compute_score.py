# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from verl.utils.reward_score.prime_math import compute_score as prime_compute_score
from verl.utils.reward_score.math_verify import compute_score as math_verify_compute_score

def compute_score(model_output: str, ground_truth: str) -> bool:
    try:
        prime_score = prime_compute_score(model_output, ground_truth)[0]
        if prime_score:
            return 1.0
    except Exception as e:
        prime_score = 0.0
    try:
        math_verify_score = math_verify_compute_score(model_output, ground_truth)
        if math_verify_score:
            return 1.0
    except Exception as e:
        return 0.0
    return 0.0
