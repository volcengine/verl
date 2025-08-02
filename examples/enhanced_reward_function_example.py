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
Enhanced Reward Function Example - Demonstrates Enhanced Reward API best practices

This example shows how to use the Enhanced Reward API to solve the three key pain points:
1. KeyError 'score': Use RewardResult class for standardized API
2. Metric length mismatch: Use sparse metrics for natural metric handling
3. TensorBoard limitations: Use MetricConfig for flexible metric organization
"""

import json
from verl.utils.reward_score.result import RewardResult, MetricConfig


# Example 1: Simple reward function using RewardResult
def simple_reward_function(solution_str, ground_truth, **kwargs):
    """
    Simple example showing basic RewardResult usage.
    
    This eliminates the need to remember the 'score' key requirement.
    """
    # Your scoring logic here
    base_score = 0.8 if "correct" in solution_str.lower() else 0.3
    
    # Simple usage - just return score
    return RewardResult(score=base_score)


# Example 2: Enhanced reward function showing best practices
def enhanced_reward_function(solution_str, ground_truth, extra_info=None, **kwargs):
    """
    Enhanced reward function demonstrating best practices.
    
    Key improvements:
    1. Uses RewardResult instead of raw dict (solves KeyError 'score')
    2. Supports sparse metrics naturally (solves metric length mismatch)
    3. Better error handling and flexibility
    4. Cleaner API design
    """
    # Extract task information
    task_type = extra_info.get("task_type", "general") if extra_info else "general"
    
    # Calculate base score based on exact match
    base_score = 1.0 if solution_str.strip() == ground_truth.strip() else 0.0
    
    # Partial credit for partial matches
    if base_score == 0.0:
        solution_words = set(solution_str.lower().split())
        truth_words = set(ground_truth.lower().split())
        overlap = len(solution_words & truth_words)
        total = len(truth_words)
        base_score = overlap / max(total, 1) * 0.8  # Max 0.8 for partial match
    
    # 🚀 Use Enhanced API - eliminates KeyError issues
    result = RewardResult(score=base_score)
    
    # ✨ Core metrics - available for all samples
    result.add_metric("accuracy", 1.0 if base_score > 0.9 else 0.0)
    
    quality_score = min(1.0, len(solution_str) / max(len(ground_truth), 1))
    result.add_metric("quality_score", quality_score)
    
    # 🎯 Sparse metrics - only add when relevant (no forced zero padding!)
    
    # Classification-specific metrics
    if task_type == "classification":
        # For classification, exact match is required
        classification_correct = solution_str.strip() == ground_truth.strip()
        result.add_metric("classification_accuracy", 1.0 if classification_correct else 0.0)
    
    # Generation-specific metrics
    elif task_type == "generation":
        # For generation, evaluate fluency and creativity
        word_count = len(solution_str.split())
        fluency_score = min(1.0, word_count / 50)  # Normalize by expected length
        result.add_metric("fluency_score", fluency_score)
        
        # Creativity based on unique words
        unique_ratio = len(set(solution_str.split())) / max(len(solution_str.split()), 1)
        result.add_metric("creativity_score", unique_ratio)
    
    # Question-answering specific metrics
    elif task_type == "qa":
        # Check if answer addresses the question
        if "?" in ground_truth:
            contains_answer = any(word in solution_str.lower() 
                                for word in ["yes", "no", "because", "due to"])
            result.add_metric("answer_completeness", 1.0 if contains_answer else 0.0)
    
    # Safety metrics for safety-critical samples
    if extra_info and extra_info.get("safety_critical", False):
        # Simple safety check (placeholder for real safety classifier)
        unsafe_patterns = ["harmful", "dangerous", "illegal", "violence"]
        is_safe = not any(pattern in solution_str.lower() for pattern in unsafe_patterns)
        result.add_metric("safety_score", 1.0 if is_safe else 0.0)
    
    # Length-based metrics for long responses
    if len(solution_str.split()) > 50:
        # For long responses, check coherence (simple proxy)
        sentences = solution_str.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        coherence_score = min(1.0, avg_sentence_length / 20)  # Normalize
        result.add_metric("coherence_score", coherence_score)
    
    return result


# Example 3: Configuration for the enhanced reward function
def create_example_metric_config():
    """
    Create a metric configuration demonstrating flexible TensorBoard organization.
    
    This explicitly defines which metrics are core and which are sparse,
    eliminating the need for implicit naming conventions.
    """
    return MetricConfig(
        # 🎯 Core metrics - most important metrics (displayed in val-core group)
        core_metrics=[
            "reward",        # Primary RL training signal
            "accuracy",      # Key performance metric
            "quality_score"  # Overall quality assessment
        ],
        
        # 🔄 Sparse metrics - task-specific metrics (displayed in val-aux group)
        sparse_metrics=[
            "classification_accuracy",  # Only for classification tasks
            "fluency_score",            # Only for generation tasks
            "creativity_score",         # Only for generation tasks
            "answer_completeness",      # Only for QA tasks
            "safety_score",             # Only for safety-critical samples
            "coherence_score"           # Only for long responses
        ],
        
        # 📊 TensorBoard configuration
        tensorboard_prefix="example-metrics",  # Custom prefix for organization
        auto_detect_accuracy=True              # Enable automatic accuracy detection
    )


# Example 4: Using the enhanced reward manager in configuration
EXAMPLE_CONFIG = {
    "reward_model": {
        "reward_manager": "enhanced",  # Use the new enhanced reward manager
        "reward_kwargs": {
            "metric_config": create_example_metric_config(),
            "strict_mode": False  # Use warnings instead of errors for metric issues
        }
    },
    
    "custom_reward_function": {
        "path": "path/to/this/file.py",
        "name": "enhanced_reward_function"
    }
}


# Example 5: Migration helper for existing reward functions
def migrate_legacy_reward_function(legacy_result):
    """
    Helper to migrate existing dict-based reward functions to RewardResult.
    
    This can be used as a wrapper for existing reward functions during migration.
    """
    if isinstance(legacy_result, dict):
        return RewardResult.from_dict(legacy_result)
    elif isinstance(legacy_result, (int, float)):
        return RewardResult(score=legacy_result)
    else:
        return legacy_result  # Already a RewardResult


if __name__ == "__main__":
    # Test the enhanced reward function
    print("🚀 Enhanced Reward Function Example")
    print("===================================")
    
    # Test 1: Simple classification task
    solution1 = "This is a positive sentiment"
    ground_truth1 = "This is a positive sentiment"
    extra_info1 = {"task_type": "classification"}
    
    result1 = enhanced_reward_function(solution1, ground_truth1, extra_info1)
    print(f"✅ Classification Test - Score: {result1.score:.2f}, Metrics: {result1.metrics}")
    
    # Test 2: Generation task
    solution2 = "This is a creative story about a brave knight who ventured into the unknown realm."
    ground_truth2 = "Write a creative story"
    extra_info2 = {"task_type": "generation"}
    
    result2 = enhanced_reward_function(solution2, ground_truth2, extra_info2)
    print(f"✅ Generation Test - Score: {result2.score:.2f}, Metrics: {result2.metrics}")
    
    # Test 3: Safety-critical sample
    solution3 = "This is a helpful and safe response"
    ground_truth3 = "Provide helpful information"
    extra_info3 = {"task_type": "qa", "safety_critical": True}
    
    result3 = enhanced_reward_function(solution3, ground_truth3, extra_info3)
    print(f"✅ Safety Test - Score: {result3.score:.2f}, Metrics: {result3.metrics}")
    
    # Show backward compatibility
    dict_result = result1.to_dict()
    print(f"✅ Backward Compatible Dict: {dict_result}")
    
    # Show dict-like access
    print(f"✅ Dict-like Access: score={result1['score']}, accuracy={result1['accuracy']}")
    
    # Test metric configuration
    config = create_example_metric_config()
    print(f"✅ Metric Config - Core: {config.core_metrics}")
    print(f"✅ Metric Config - Sparse: {config.sparse_metrics}")
    
    print("\n🎉 Enhanced Reward API working correctly!")