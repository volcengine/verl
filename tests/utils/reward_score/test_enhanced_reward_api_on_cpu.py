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
Tests for the enhanced reward function API.
"""

import pytest
from verl.utils.reward_score.result import RewardResult, MetricConfig


class TestRewardResult:
    """Test the RewardResult class functionality."""
    
    def test_basic_creation(self):
        """Test basic RewardResult creation."""
        result = RewardResult(score=0.85)
        assert result.score == 0.85
        assert result.metrics == {}
        assert result.metadata == {}
    
    def test_with_metrics(self):
        """Test RewardResult with metrics."""
        result = RewardResult(
            score=0.85,
            metrics={"accuracy": 0.92, "precision": 0.88}
        )
        assert result.score == 0.85
        assert result.metrics["accuracy"] == 0.92
        assert result.metrics["precision"] == 0.88
    
    def test_fluent_interface(self):
        """Test the fluent interface for adding metrics."""
        result = (RewardResult(score=0.8)
                 .add_metric("accuracy", 0.9)
                 .add_metric("precision", 0.85, condition=True)
                 .add_metric("recall", 0.75, condition=False))  # Should not be added
        
        assert result.score == 0.8
        assert result.metrics["accuracy"] == 0.9
        assert result.metrics["precision"] == 0.85
        assert "recall" not in result.metrics
    
    def test_to_dict_conversion(self):
        """Test conversion to dictionary format."""
        result = RewardResult(
            score=0.85,
            metrics={"accuracy": 0.92},
            metadata={"debug": "test"}
        )
        
        dict_result = result.to_dict()
        expected = {
            "score": 0.85,
            "accuracy": 0.92,
            "metadata": {"debug": "test"}
        }
        assert dict_result == expected
    
    def test_from_dict_conversion(self):
        """Test creation from dictionary format."""
        data = {
            "score": 0.85,
            "accuracy": 0.92,
            "precision": 0.88
        }
        
        result = RewardResult.from_dict(data)
        assert result.score == 0.85
        assert result.metrics["accuracy"] == 0.92
        assert result.metrics["precision"] == 0.88
    
    def test_from_dict_missing_score(self):
        """Test error when dictionary is missing score key."""
        data = {"accuracy": 0.92}
        
        with pytest.raises(ValueError, match="Dictionary must contain 'score' key"):
            RewardResult.from_dict(data)
    
    def test_backward_compatibility_access(self):
        """Test dict-like access for backward compatibility."""
        result = RewardResult(
            score=0.85,
            metrics={"accuracy": 0.92}
        )
        
        # Test __getitem__
        assert result["score"] == 0.85
        assert result["accuracy"] == 0.92
        
        # Test __contains__
        assert "score" in result
        assert "accuracy" in result
        assert "missing" not in result
        
        # Test keys()
        keys = list(result.keys())
        assert "score" in keys
        assert "accuracy" in keys
        
        # Test items()
        items = dict(result.items())
        assert items["score"] == 0.85
        assert items["accuracy"] == 0.92


class TestMetricConfig:
    """Test the MetricConfig class functionality."""
    
    def test_default_config(self):
        """Test default MetricConfig behavior."""
        config = MetricConfig()
        
        assert config.core_metrics == []
        assert config.sparse_metrics == []
        assert config.tensorboard_prefix == "val-aux"
        assert config.auto_detect_accuracy is True
    
    def test_core_metric_detection(self):
        """Test core metric detection logic."""
        config = MetricConfig(
            core_metrics=["custom_metric"],
            auto_detect_accuracy=True
        )
        
        # Explicitly configured core metrics
        assert config.is_core_metric("custom_metric") is True
        
        # Auto-detected accuracy metrics
        assert config.is_core_metric("accuracy_overall") is True
        assert config.is_core_metric("precision_score") is True
        assert config.is_core_metric("recall_rate") is True
        assert config.is_core_metric("f1_measure") is True
        
        # Non-core metrics
        assert config.is_core_metric("random_metric") is False
    
    def test_sparse_metric_detection(self):
        """Test sparse metric detection."""
        config = MetricConfig(sparse_metrics=["tool_accuracy", "conditional_metric"])
        
        assert config.is_sparse_metric("tool_accuracy") is True
        assert config.is_sparse_metric("conditional_metric") is True
        assert config.is_sparse_metric("regular_metric") is False
    
    def test_auto_detect_disabled(self):
        """Test behavior when auto-detection is disabled."""
        config = MetricConfig(
            core_metrics=["explicit_core"],
            auto_detect_accuracy=False
        )
        
        assert config.is_core_metric("explicit_core") is True
        assert config.is_core_metric("accuracy_overall") is False  # Not auto-detected


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""
    
    def test_legacy_dict_handling(self):
        """Test that legacy dict-based returns still work."""
        def legacy_reward_function():
            return {
                "score": 0.85,
                "accuracy": 0.92,
                "precision": 0.88
            }
        
        # This should work with existing reward manager code
        result = legacy_reward_function()
        assert result["score"] == 0.85
        assert "score" in result
    
    def test_migration_helper(self):
        """Test migration from legacy to new format."""
        def migrate_result(legacy_result):
            if isinstance(legacy_result, dict):
                return RewardResult.from_dict(legacy_result)
            elif isinstance(legacy_result, (int, float)):
                return RewardResult(score=legacy_result)
            else:
                return legacy_result
        
        # Test dict migration
        dict_result = {"score": 0.85, "accuracy": 0.92}
        migrated = migrate_result(dict_result)
        assert isinstance(migrated, RewardResult)
        assert migrated.score == 0.85
        assert migrated.metrics["accuracy"] == 0.92
        
        # Test scalar migration
        scalar_result = 0.75
        migrated = migrate_result(scalar_result)
        assert isinstance(migrated, RewardResult)
        assert migrated.score == 0.75
        assert migrated.metrics == {}
        
        # Test already migrated
        reward_result = RewardResult(score=0.9)
        migrated = migrate_result(reward_result)
        assert migrated is reward_result


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_metric_access(self):
        """Test error handling for invalid metric access."""
        result = RewardResult(score=0.85)
        
        with pytest.raises(KeyError):
            _ = result["nonexistent_metric"]
    
    def test_metric_type_validation(self):
        """Test that metrics accept appropriate types."""
        result = RewardResult(score=0.85)
        
        # Should work with int and float
        result.add_metric("int_metric", 1)
        result.add_metric("float_metric", 0.5)
        
        assert result.metrics["int_metric"] == 1
        assert result.metrics["float_metric"] == 0.5


if __name__ == "__main__":
    pytest.main([__file__])