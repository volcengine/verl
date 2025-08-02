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
Standardized reward function result classes for better API design.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RewardResult:
    """
    Standardized result class for reward functions.
    
    This class provides a flexible and user-friendly way to return rewards and metrics
    from reward functions, addressing common issues with the previous dict-based approach.
    
    Attributes:
        score: The main reward score (required)
        metrics: Dictionary of additional metrics (optional)
        metadata: Additional metadata for debugging/logging (optional)
    
    Examples:
        # Simple usage - just a score
        return RewardResult(score=0.85)
        
        # With custom metrics
        return RewardResult(
            score=0.85,
            metrics={
                "accuracy_overall": 0.92,
                "accuracy_reply": 0.88,
                "tool_recall": 0.45
            }
        )
        
        # With sparse metrics (only relevant samples)
        metrics = {}
        if is_reply_type:
            metrics["accuracy_reply"] = 1.0 if correct else 0.0
        return RewardResult(score=base_score, metrics=metrics)
    """
    score: float | int
    metrics: dict[str, float | int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary format for backward compatibility.
        
        Returns:
            Dict containing 'score' key and all metrics as separate keys.
        """
        result = {"score": self.score}
        result.update(self.metrics)
        if self.metadata:
            result["metadata"] = self.metadata
        return result
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RewardResult":
        """
        Create RewardResult from dictionary format.
        
        Args:
            data: Dictionary with 'score' key and optional metric keys
            
        Returns:
            RewardResult instance
            
        Raises:
            ValueError: If 'score' key is missing
        """
        if "score" not in data:
            raise ValueError("Dictionary must contain 'score' key")
        
        score = data["score"]
        metadata = data.pop("metadata", {})
        metrics = {k: v for k, v in data.items() if k != "score"}
        
        return cls(score=score, metrics=metrics, metadata=metadata)
    
    def add_metric(self, name: str, value: float | int, 
                   condition: bool = True) -> "RewardResult":
        """
        Conditionally add a metric (fluent interface).
        
        Args:
            name: Metric name
            value: Metric value
            condition: Only add if True
            
        Returns:
            Self for method chaining
        """
        if condition:
            self.metrics[name] = value
        return self
    
    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access for backward compatibility."""
        if key == "score":
            return self.score
        elif key in self.metrics:
            return self.metrics[key]
        elif key == "metadata" and self.metadata:
            return self.metadata
        else:
            raise KeyError(f"Key '{key}' not found")
    
    def __contains__(self, key: str) -> bool:
        """Allow 'in' operator for backward compatibility."""
        return key == "score" or key in self.metrics or (key == "metadata" and self.metadata)
    
    def keys(self):
        """Return all available keys for backward compatibility."""
        keys = ["score"] + list(self.metrics.keys())
        if self.metadata:
            keys.append("metadata")
        return keys
    
    def items(self):
        """Return key-value pairs for backward compatibility."""
        yield "score", self.score
        yield from self.metrics.items()
        if self.metadata:
            yield "metadata", self.metadata


class MetricConfig:
    """
    Configuration for metric handling and TensorBoard integration.
    
    This class allows users to explicitly configure how metrics should be processed
    and displayed, eliminating the need for implicit naming conventions.
    """
    
    def __init__(self,
                 core_metrics: list | None = None,
                 sparse_metrics: list | None = None,
                 tensorboard_prefix: str = "val-aux",
                 auto_detect_accuracy: bool = True):
        """
        Args:
            core_metrics: List of metric names to treat as core metrics for TensorBoard
            sparse_metrics: List of metric names that may not be present for all samples
            tensorboard_prefix: Prefix for TensorBoard metric names
            auto_detect_accuracy: Whether to auto-detect accuracy metrics by name patterns
        """
        self.core_metrics = core_metrics or []
        self.sparse_metrics = sparse_metrics or []
        self.tensorboard_prefix = tensorboard_prefix
        self.auto_detect_accuracy = auto_detect_accuracy
    
    def is_core_metric(self, metric_name: str) -> bool:
        """Check if a metric should be treated as a core metric."""
        if metric_name in self.core_metrics:
            return True
        
        if self.auto_detect_accuracy:
            # Auto-detect accuracy-like metrics
            acc_patterns = ["acc", "accuracy", "precision", "recall", "f1"]
            return any(pattern in metric_name.lower() for pattern in acc_patterns)
        
        return False
    
    def is_sparse_metric(self, metric_name: str) -> bool:
        """Check if a metric is allowed to be sparse (not present for all samples)."""
        return metric_name in self.sparse_metrics