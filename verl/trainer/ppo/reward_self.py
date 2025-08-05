import os
import sys
import importlib.util
from omegaconf import OmegaConf, DictConfig

def get_custom_cat_fn(config: DictConfig):
    try:
        cat_fn_config = config.actor_self_judgement.custom_cat_function
        file_path = cat_fn_config.path
        function_name = cat_fn_config.name
    except Exception as e:
        print(f"[ERROR] Could not access 'config.actor_self_judgement.custom_cat_function': {e}")
        print("Please ensure the configuration structure is correct and passed completely.")
        return None
    
    if not file_path or not function_name:
        print(" No custom cat function path or name provided. Skipping.")
        return None

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Custom cat function file '{file_path}' not found.")

    try:
        spec = importlib.util.spec_from_file_location("custom_cat_module", file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["custom_cat_module"] = module
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Error loading module from '{file_path}': {e}") from e

    if not hasattr(module, function_name):
        raise AttributeError(f"Cat function '{function_name}' not found in module '{file_path}'.")

    print(f" Using customized cat function '{function_name}' from '{file_path}'")
    raw_fn = getattr(module, function_name)
    cat_kwargs = dict(cat_fn_config.get("cat_kwargs", {}))

    def wrapped_fn(*args, **kwargs):
        final_kwargs = {**kwargs, **cat_kwargs}
        return raw_fn(*args, **final_kwargs)

    return wrapped_fn

def get_custom_format_validator(config: DictConfig):

    try:
        validator_config = config.actor_self_judgement.custom_reward_format_validator
        file_path = validator_config.path
        function_name = validator_config.name
    except Exception as e:
        print(f"[ERROR] Could not access 'config.actor_self_judgement.custom_reward_format_validator': {e}")
        return None
    
    if not file_path or not function_name:
        print(" No custom reward format validator path or name provided. Skipping.")
        return None

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward format validator file '{file_path}' not found.")

    try:
        spec = importlib.util.spec_from_file_location("custom_validator_module", file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["custom_validator_module"] = module
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Error loading module from '{file_path}': {e}") from e

    if not hasattr(module, function_name):
        raise AttributeError(f"Validator function '{function_name}' not found in module '{file_path}'.")

    print(f" Using customized reward format validator '{function_name}' from '{file_path}'")
    validator_fn = getattr(module, function_name)

    return validator_fn
