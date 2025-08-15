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
Enhanced reward manager with improved API design and better error handling.

This module provides an Enhanced Reward Manager that addresses three key limitations
of the original VERL reward API:

1. KeyError 'score': Rigid API requiring specific dictionary format
2. Metric length mismatch: Overly strict validation requiring all samples to have all metrics  
3. TensorBoard metric recognition: Implicit naming conventions requiring 'acc' prefix

The Enhanced Reward Manager introduces:
- RewardResult class for standardized return format
- Configurable sparse metrics support
- Explicit MetricConfig for TensorBoard integration
- Graceful error handling with warnings instead of crashes
- 100% backward compatibility with existing code

Example usage:
    # Basic usage with default configuration
    manager = EnhancedRewardManager(tokenizer, num_examine=1)
    
    # Advanced usage with custom configuration  
    config = MetricConfig(
        core_metrics=["reward", "accuracy_overall"],
        sparse_metrics=["accuracy_reply", "tool_call_accuracy"],
        tensorboard_prefix="my-metrics"
    )
    manager = EnhancedRewardManager(tokenizer, num_examine=1, metric_config=config)
"""

import warnings
from collections import defaultdict
from typing import Any

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score.result import MetricConfig, RewardResult
from verl.workers.reward_manager import register


@register("enhanced")
class EnhancedRewardManager:
    """
    Enhanced reward manager with improved API design.
    
    Key improvements:
    1. Flexible return format handling (RewardResult, dict, or scalar)
    2. Better error messages with debugging information
    3. Configurable metric handling (core vs sparse metrics)
    4. Graceful degradation for missing metrics
    5. Enhanced logging and debugging support
    """

    def __init__(self, 
                 tokenizer, 
                 num_examine: int,
                 compute_score=None, 
                 reward_fn_key: str = "data_source",
                 metric_config: MetricConfig | None = None,
                 strict_mode: bool = False) -> None:
        """
        Initialize the EnhancedRewardManager.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print for debugging.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data.
            metric_config: Configuration for metric handling. If None, creates a default config.
            strict_mode: If True, raises errors for metric inconsistencies. If False, logs warnings.
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.metric_config = self._normalize_metric_config(metric_config)
        self.strict_mode = strict_mode

    def _normalize_metric_config(self, metric_config: MetricConfig | dict | Any | None) -> MetricConfig:
        """
        Normalize metric_config to handle both MetricConfig instances and Hydra structured configs.
        
        Args:
            metric_config: Either a MetricConfig instance, a dictionary, or a Hydra structured config
            
        Returns:
            A MetricConfig instance
        """
        if metric_config is None:
            return MetricConfig()
        
        # If it's already a MetricConfig instance, return as-is
        if isinstance(metric_config, MetricConfig):
            return metric_config
        
        # Handle dictionary or Hydra structured config
        try:
            # Check if it's a Hydra config with _target_
            if hasattr(metric_config, '_target_') or (hasattr(metric_config, 'get') and metric_config.get('_target_')):
                # Use Hydra instantiate to create the actual MetricConfig
                try:
                    from hydra.utils import instantiate
                    actual_config = instantiate(metric_config)
                    if isinstance(actual_config, MetricConfig):
                        return actual_config
                except Exception:
                    pass
            
            # Try to access as dictionary/structured config
            core_metrics = getattr(metric_config, 'core_metrics', None) or metric_config.get('core_metrics', [])
            sparse_metrics = getattr(metric_config, 'sparse_metrics', None) or metric_config.get('sparse_metrics', [])
            tensorboard_prefix = (
                getattr(metric_config, 'tensorboard_prefix', None) or 
                metric_config.get('tensorboard_prefix', 'val-aux')
            )
            auto_detect_accuracy = getattr(metric_config, 'auto_detect_accuracy', None)
            if auto_detect_accuracy is None:
                auto_detect_accuracy = metric_config.get('auto_detect_accuracy', True)
            
            return MetricConfig(
                core_metrics=core_metrics,
                sparse_metrics=sparse_metrics,
                tensorboard_prefix=tensorboard_prefix,
                auto_detect_accuracy=auto_detect_accuracy
            )
        except (AttributeError, TypeError) as e:
            # Fallback to default config
            warnings.warn(
                f"Could not parse metric_config {type(metric_config)}, using default configuration. Error: {e}",
                stacklevel=2
            )
            return MetricConfig()

    def _normalize_score_result(self, 
                               score: float | int | dict | RewardResult, 
                               sample_idx: int,
                               data_source: str) -> tuple[float, dict[str, Any]]:
        """
        Normalize different score return formats into a consistent format.
        
        Args:
            score: The raw score result from compute_score function
            sample_idx: Index of the current sample (for debugging)
            data_source: Data source identifier (for debugging)
            
        Returns:
            Tuple of (reward_value, extra_info_dict)
            
        Raises:
            ValueError: If score format is invalid or missing required 'score' key
        """
        try:
            # Handle RewardResult objects
            if isinstance(score, RewardResult):
                return score.score, score.metrics
            
            # Handle dictionary format
            elif isinstance(score, dict):
                if "score" not in score:
                    available_keys = list(score.keys())
                    raise ValueError(
                        f"Dictionary result must contain 'score' key. "
                        f"Available keys: {available_keys}. "
                        f"Sample {sample_idx} from data_source '{data_source}'"
                    )
                
                reward_value = score["score"]
                extra_info = {k: v for k, v in score.items() if k != "score"}
                return reward_value, extra_info
            
            # Handle scalar values
            elif isinstance(score, int | float):
                return float(score), {}
            
            else:
                raise ValueError(
                    f"Unsupported score type: {type(score)}. "
                    f"Expected float, int, dict, or RewardResult. "
                    f"Sample {sample_idx} from data_source '{data_source}'"
                )
                
        except Exception as e:
            # Provide detailed error context
            error_msg = (
                f"Error processing score result for sample {sample_idx} "
                f"from data_source '{data_source}': {str(e)}\n"
                f"Score value: {score}\n"
                f"Score type: {type(score)}"
            )
            raise ValueError(error_msg) from e

    def _validate_and_collect_metrics(self, 
                                    all_extra_info: dict[str, list],
                                    total_samples: int) -> dict[str, list]:
        """
        Validate metric consistency and handle sparse metrics gracefully.
        
        Args:
            all_extra_info: Dictionary mapping metric names to lists of values
            total_samples: Expected total number of samples
            
        Returns:
            Validated and normalized metric dictionary
        """
        validated_metrics = {}
        issues = []
        
        for metric_name, values in all_extra_info.items():
            current_length = len(values)
            
            # Perfect case - all samples have this metric
            if current_length == total_samples:
                validated_metrics[metric_name] = values
                continue
            
            # Empty case - no samples have this metric
            if current_length == 0:
                if not self.metric_config.is_sparse_metric(metric_name):
                    issues.append(f"Metric '{metric_name}' has no values but is not configured as sparse")
                continue
            
            # Partial case - some samples have this metric
            is_sparse = self.metric_config.is_sparse_metric(metric_name)
            # Check if this is a sparse metric that doesn't need to be present for all samples
            if is_sparse:
                # For sparse metrics, pad with None or 0 as appropriate
                # This allows downstream processing to handle missing values
                padded_values = self._pad_sparse_metric(values, total_samples, metric_name)
                validated_metrics[metric_name] = padded_values
                # Padded sparse metric for consistent processing
            else:
                issues.append(
                    f"Metric '{metric_name}' has {current_length} values but expected {total_samples}. "
                    f"Consider adding it to sparse_metrics configuration."
                )
        
        # Handle validation issues
        if issues:
            error_msg = "Metric validation issues:\n" + "\n".join(f"  - {issue}" for issue in issues)
            if self.strict_mode:
                raise ValueError(error_msg)
            else:
                warnings.warn(
                    f"Metric validation warnings:\n{error_msg}", 
                    UserWarning, 
                    stacklevel=2
                )
        
        return validated_metrics

    def _pad_sparse_metric(self, 
                          values: list, 
                          target_length: int, 
                          metric_name: str) -> list:
        """
        Pad sparse metrics to match the expected length.
        
        For sparse metrics, we need to indicate which samples don't have values.
        This is better than the current approach of forcing 0 values.
        """
        if len(values) >= target_length:
            return values[:target_length]
        
        # For now, pad with 0.0 to maintain backward compatibility
        # In future versions, we could use None or a special sentinel value
        padding_value = 0.0
        padded = values + [padding_value] * (target_length - len(values))
        
        # Note: Sparse metric was padded from {len(values)} to {target_length} values
        
        return padded

    def __call__(self, data: DataProto, return_dict=False):
        """
        Process data and compute rewards with enhanced error handling.
        
        Args:
            data: DataProto containing batch data
            return_dict: Whether to return additional metric information
            
        Returns:
            Reward tensor, or dict with reward tensor and extra info
        """
        # Check for pre-computed RM scores
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]
            
            try:
                # Extract input data
                prompt_ids = data_item.batch["prompts"]
                prompt_length = prompt_ids.shape[-1]

                response_ids = data_item.batch["responses"]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                valid_response_ids = response_ids[:valid_response_length]

                # Decode text
                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

                # Get metadata
                ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
                data_source = data_item.non_tensor_batch[self.reward_fn_key]
                extra_info = data_item.non_tensor_batch.get("extra_info", {})
                num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
                extra_info["num_turns"] = num_turns

                # Compute score with enhanced error handling
                score = self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                )

                # Normalize the result
                reward_value, extra_metrics = self._normalize_score_result(
                    score, i, data_source
                )

                # Store reward
                reward_tensor[i, valid_response_length - 1] = reward_value

                # Collect extra metrics
                for key, value in extra_metrics.items():
                    reward_extra_info[key].append(value)

                # Debug printing
                if data_source not in already_print_data_sources:
                    already_print_data_sources[data_source] = 0

                if already_print_data_sources[data_source] < self.num_examine:
                    already_print_data_sources[data_source] += 1
                    # Debug information available but not printed to avoid stdout pollution
                    # Users can enable detailed logging via logging framework if needed

            except Exception as e:
                error_msg = (
                    f"Error processing sample {i}: {str(e)}\n"
                    f"Data source: {data_item.non_tensor_batch.get(self.reward_fn_key, 'unknown')}"
                )
                print(f"[Enhanced Reward Manager Error] {error_msg}")
                
                if self.strict_mode:
                    raise RuntimeError(error_msg) from e
                else:
                    # Use default reward value and continue
                    reward_tensor[i, -1] = 0.0
                    warnings.warn(
                        f"Using default reward (0.0) for failed sample {i}: {str(e)}",
                        stacklevel=2
                    )

        # Validate metrics if returning dict
        if return_dict:
            validated_metrics = self._validate_and_collect_metrics(
                dict(reward_extra_info), 
                len(data)
            )
            
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": validated_metrics,
            }
        else:
            return reward_tensor