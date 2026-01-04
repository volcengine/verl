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

from enum import Enum
from typing import Any, Optional, Union

import numpy as np
import torch


def reduce_metrics(metrics: dict[str, Union["MetricList", list[Any]]]) -> dict[str, Any]:
    """
    Reduces a dictionary of metric lists by computing the mean, max, or min of each list.
    The reduce operation is determined by the key name:
    - If the key contains "max", np.max is used
    - If the key contains "min", np.min is used
    - Otherwise, np.mean is used

    Args:
        metrics: A dictionary mapping metric names to lists of metric values.

    Returns:
        A dictionary with the same keys but with each list replaced by its reduced value.

    Example:
        >>> metrics = {
        ...     "loss": [1.0, 2.0, 3.0],
        ...     "accuracy": [0.8, 0.9, 0.7],
        ...     "max_reward": [5.0, 8.0, 6.0],
        ...     "min_error": [0.1, 0.05, 0.2]
        ... }
        >>> reduce_metrics(metrics)
        {"loss": 2.0, "accuracy": 0.8, "max_reward": 8.0, "min_error": 0.05}
    """
    for key, val in metrics.items():
        if isinstance(val, MetricList):
            metrics[key] = val.aggregate()
        elif "max" in key:
            metrics[key] = np.max(val)
        elif "min" in key:
            metrics[key] = np.min(val)
        else:
            metrics[key] = np.mean(val)
    return metrics


class AggregationType(Enum):
    MEAN = "mean"
    SUM = "sum"
    MIN = "min"
    MAX = "max"


Numeric = Union[int, float, torch.Tensor]


def tensor_to_float(value: torch.Tensor) -> float:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"Expected torch.Tensor, got {type(value)}")
    if value.numel() != 1:
        raise ValueError("Only scalar tensors can be converted to float")
    return value.detach().item()


class MetricValue:
    def __init__(self, value: Numeric, aggregation: str | AggregationType) -> None:
        if isinstance(value, torch.Tensor):
            value = tensor_to_float(value)
        if not isinstance(value, Numeric):
            raise ValueError(f"Unsupported value type: {type(value)}")
        self.value = value
        if isinstance(aggregation, str):
            self.aggregation = AggregationType(aggregation)
        else:
            self.aggregation = aggregation
        if not isinstance(self.aggregation, AggregationType):
            raise ValueError(f"Unsupported aggregation type: {aggregation}")

    @classmethod
    def from_dict(cls, data: dict[str, Numeric], aggregation: str | AggregationType) -> dict[str, "MetricValue"]:
        return {key: cls(value, aggregation) for key, value in data.items()}

    def init_list(self) -> "MetricList":
        return MetricList(aggregation=self.aggregation)


class MetricList:
    def __init__(self, aggregation: str | AggregationType, values: Optional[list[float]] = None) -> None:
        if isinstance(aggregation, str):
            self.aggregation = AggregationType(aggregation)
        else:
            self.aggregation = aggregation
        if not isinstance(self.aggregation, AggregationType):
            raise ValueError(f"Unsupported aggregation type: {aggregation}")
        self.values = values if values is not None else []

    def append(self, value: float | MetricValue) -> None:
        if isinstance(value, MetricValue):
            if value.aggregation != self.aggregation:
                raise AggregationTypeMismatchError(self.aggregation, value.aggregation)
            value = value.value
        if isinstance(value, torch.Tensor):
            value = tensor_to_float(value)
        if not isinstance(value, Numeric):
            raise ValueError(f"Unsupported value type: {type(value)}")
        self.values.append(value)

    def extend(self, values: Union["MetricList", list[float | MetricValue]]) -> None:
        if isinstance(values, MetricList):
            if values.aggregation != self.aggregation:
                raise AggregationTypeMismatchError(self.aggregation, values.aggregation)
            values = values.values
        for value in values:
            self.append(value)

    def aggregate(self) -> float:
        match self.aggregation:
            case AggregationType.MEAN:
                return np.mean(self.values)
            case AggregationType.SUM:
                return np.sum(self.values)
            case AggregationType.MIN:
                return np.min(self.values)
            case AggregationType.MAX:
                return np.max(self.values)

    @classmethod
    def chain(cls, metric_lists: list["MetricList"]) -> "MetricList":
        if len(metric_lists) == 0:
            return cls(aggregation=AggregationType.MEAN)
        aggregation = metric_lists[0].aggregation
        chained = cls(aggregation=aggregation)
        for ml in metric_lists:
            chained.extend(ml)
        return chained


class AggregationTypeMismatchError(Exception):
    def __init__(self, agg1: AggregationType, agg2: AggregationType):
        msg = f"Aggregation type mismatch: {agg1.value} != {agg2.value}"
        super().__init__(msg)
