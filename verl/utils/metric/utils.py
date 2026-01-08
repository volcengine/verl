# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
Metrics utils.
"""

from typing import Any, Optional

import numpy as np


def weighted_mean(values: list, weights: Optional[list] = None) -> float:
    """
    Calculate weighted mean of values.

    Args:
        values: List of values to average
        weights: Optional list of weights (e.g., sample counts per rank)

    Returns:
        Weighted mean if weights provided, otherwise simple mean

    Example:
        >>> weighted_mean([0.8, 0.6], [100, 200])
        0.6666666666666666  # (100*0.8 + 200*0.6) / 300
    """
    if weights is None or len(weights) == 0:
        return np.mean(values)

    values = np.array(values)
    weights = np.array(weights)

    if len(values) != len(weights):
        raise ValueError(f"Values (len={len(values)}) and weights (len={len(weights)}) must have same length")

    weight_sum = np.sum(weights)
    if weight_sum == 0:
        return np.nan

    return np.sum(values * weights) / weight_sum


def reduce_metrics(
    metrics: dict[str, list[Any]],
    weights: Optional[dict[str, list[float]]] = None,
    reduction_strategy: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """
    Reduces a dictionary of metric lists by computing aggregated values.

    This function properly handles metric reduction across distributed ranks,
    supporting weighted averaging for non-linear metrics like means.

    The reduce operation is determined by (in priority order):
    1. Custom reduction_strategy if provided
    2. Key name patterns:
       - If key contains "max" -> np.max
       - If key contains "min" -> np.min
       - If key contains "sum" -> np.sum
    3. Otherwise -> weighted mean (if weights provided) or simple mean

    Args:
        metrics: A dictionary mapping metric names to lists of metric values.
        weights: Optional dictionary mapping metric names to lists of weights
                (e.g., sample counts per rank). If provided for a metric,
                weighted average will be used instead of simple average.
        reduction_strategy: Optional dictionary mapping metric names to reduction
                          methods ("mean", "weighted_mean", "max", "min", "sum").

    Returns:
        A dictionary with the same keys but with each list replaced by its reduced value.

    Example:
        >>> # Simple usage (backward compatible)
        >>> metrics = {
        ...     "loss": [1.0, 2.0, 3.0],
        ...     "max_reward": [5.0, 8.0, 6.0],
        ... }
        >>> reduce_metrics(metrics)
        {"loss": 2.0, "max_reward": 8.0}

        >>> # With weights for proper distributed averaging
        >>> metrics = {"mean_accuracy": [0.8, 0.6]}
        >>> weights = {"mean_accuracy": [100, 200]}  # samples per rank
        >>> reduce_metrics(metrics, weights)
        {"mean_accuracy": 0.6666666666666666}  # (100*0.8 + 200*0.6) / 300

        >>> # With custom strategy
        >>> strategy = {"total_tokens": "sum"}
        >>> metrics = {"total_tokens": [1000, 2000, 3000]}
        >>> reduce_metrics(metrics, reduction_strategy=strategy)
        {"total_tokens": 6000}
    """
    result = {}

    for key, values in metrics.items():
        if len(values) == 0:
            result[key] = np.nan
            continue

        # Determine reduction method
        if reduction_strategy and key in reduction_strategy:
            method = reduction_strategy[key]
        elif "max" in key:
            method = "max"
        elif "min" in key:
            method = "min"
        elif "sum" in key:
            method = "sum"
        else:
            # Use weighted mean if weights are provided for this metric
            method = "weighted_mean" if weights and key in weights else "mean"

        # Apply reduction
        if method == "max":
            result[key] = np.max(values)
        elif method == "min":
            result[key] = np.min(values)
        elif method == "sum":
            result[key] = np.sum(values)
        elif method == "weighted_mean":
            w = weights.get(key) if weights else None
            result[key] = weighted_mean(values, w)
        else:  # mean
            result[key] = np.mean(values)

    return result
