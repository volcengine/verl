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
from typing import Any, Dict, List

import numpy as np


def reduce_metrics(metrics: Dict[str, List[Any]]) -> Dict[str, Any]:
    """
    Reduces a dictionary of metric lists by computing the mean of each list.
    The reduce operation is done via ``np.mean``.
    
    Args:
        metrics: A dictionary mapping metric names to lists of metric values.
        
    Returns:
        A dictionary with the same keys but with each list replaced by its mean value.
        
    Example:
        >>> metrics = {"loss": [1.0, 2.0, 3.0], "accuracy": [0.8, 0.9, 0.7]}
        >>> reduce_metrics(metrics)
        {"loss": 2.0, "accuracy": 0.8}
    """
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics