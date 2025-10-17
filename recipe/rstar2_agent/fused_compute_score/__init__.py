# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Copy math verify and prime math compute score functions from verl.utils.reward_score under this folder
# to reduce the dependency and memory cost when fused_compute_score is used by the code judge tool.

from .math_verify import compute_score as math_verify_compute_score
from .prime_math import compute_score as prime_compute_score


def compute_score(model_output: str, ground_truth: str) -> float:
    try:
        # prime_compute_score returns a tuple, we need the first element.
        if prime_compute_score(model_output, ground_truth)[0]:
            return 1.0
    except Exception:
        # If prime_compute_score fails, fall back to math_verify_compute_score.
        pass

    try:
        if math_verify_compute_score(model_output, ground_truth):
            return 1.0
    except Exception:
        # If math_verify_compute_score also fails, return 0.0.
        return 0.0

    # If both ran successfully but did not return a positive score.
    return 0.0
