# examples/sglang_multiturn/reward/gsm8k_toolcall_shaping.py
from __future__ import annotations

from typing import Any, Optional

from verl.utils.reward_score.gsm8k import compute_score as gsm8k_compute_score


def toolcall_shaping_reward(
    data_source: Optional[str],
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict[str, Any]] = None,
    *,
    method: str = "strict",
    format_score: float = 0.1,
    score: float = 1.0,
    shaping_reward: float = 0.1,
    trigger_substring: str = "<tool_call>",
    **kwargs,
) -> float:
    """
    GSM8K reward + tool-call shaping reward (trajectory-level).
    """
    base = gsm8k_compute_score(solution_str, ground_truth, method, format_score, score)

    bonus = shaping_reward if (trigger_substring and trigger_substring in solution_str) else 0.0
    return float(base + bonus)


# Optional: keep a default name for convenience in verl config (default is compute_score) [web:59][web:65]
def compute_score(
    data_source: Optional[str],
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict[str, Any]] = None,
    **kwargs,
) -> float:
    return toolcall_shaping_reward(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        **kwargs,
    )
