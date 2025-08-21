import os
import sys
import importlib.util
import os
import torch
import asyncio


async def conversation_level_reward_func(data_source, messages, ground_truth, extra_info, metrics, **kwargs) -> torch.Tensor:
    """
    Async version of conversation-level reward function.
    
    Apply conversation-level reward function to the future interactions between the user simulator 
    and policy model, which are generated from `verl/interactions/collabllm_interation.py`
    """
    num_retries = kwargs.get('num_retries', 3)
    
    rewards = {}
    for metric in metrics:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        metric_file_path = os.path.join(current_dir, f"metrics/{metric}.py")
        
        if not os.path.exists(metric_file_path):
            print(f"Error: Metric file '{metric_file_path}' not found. Assigning 0 to metric '{metric}'.")
            rewards[metric] = 0.0
            continue
        
        spec = importlib.util.spec_from_file_location(f"metric_{metric}", metric_file_path)
        if spec is None:
            print(f"Error: Could not create spec for metric '{metric}'. Assigning 0 to metric '{metric}'.")
            rewards[metric] = 0.0
            continue
            
        module = importlib.util.module_from_spec(spec)
        
        try:
            sys.modules[f"metric_{metric}"] = module
            assert spec.loader is not None
            spec.loader.exec_module(module)
        except Exception as e:
            print(f"Error loading metric module from '{metric_file_path}': {e}. Assigning 0 to metric '{metric}'.")
            rewards[metric] = 0.0
            continue
        
        # Assume each metric file has a compute_score function
        if not hasattr(module, 'compute_score'):
            print(f"Error: Function 'compute_score' not found in '{metric_file_path}'. Assigning 0 to metric '{metric}'.")
            rewards[metric] = 0.0
            continue
        
        compute_score_fn = getattr(module, 'compute_score')
        
        # Retry mechanism for calling the metric function
        for attempt in range(num_retries):
            try:
                # Call the metric function (await if it's async)
                if asyncio.iscoroutinefunction(compute_score_fn):
                    rewards[metric] = await compute_score_fn(data_source, messages, ground_truth, extra_info, **kwargs)
                else:
                    rewards[metric] = compute_score_fn(data_source, messages, ground_truth, extra_info, **kwargs)
                break  # Success, exit retry loop
            except Exception as e:
                if attempt == num_retries - 1:  # Last attempt
                    print(f"Error: Failed to compute metric '{metric}' after {num_retries} attempts. Last error: {e}. Assigning 0 to metric '{metric}'.")
                    rewards[metric] = 0.0
                else:
                    print(f"Attempt {attempt + 1} failed for metric '{metric}': {e}. Retrying...")
    
    # Return dict with metric names as keys
    return {metric: torch.tensor(reward, dtype=torch.float32) for metric, reward in rewards.items()}