# Copyright 2024 Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import Dict, Tuple, Any, Callable


def get_customized_dataset_init_params(init_func: Callable) -> Tuple[Dict[str, Dict[str, Any]], bool]:
    """
    Analyzes the signature of a class's __init__ method to extract needed parameter.
    
    This function inspects the initialization method to determine:
    1. Required and optional parameters (excluding self)
    2. Default values for optional parameters
    3. Whether the method accepts arbitrary keyword arguments (**kwargs)
    
    Args:
        init_func: The __init__ method of a class to analyze
        
    Returns:
        A tuple containing:
        - A dictionary mapping parameter names to their metadata:
          {
            'param_name': {
                'required': bool,  # Whether the parameter is required
                'default': Any     # Default value if available, None otherwise
            },
            ...
          }
        - A boolean indicating whether the method accepts **kwargs
    """
    signature = inspect.signature(init_func)

    # Skip the first parameter (self)
    params = list(signature.parameters.values())[1:]

    params_info = {}
    has_kwargs = False

    for param in params:
        # Check if the parameter is **kwargs
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            has_kwargs = True
            continue

        if param.default is not param.empty:
            params_info[param.name] = {'required': False, 'default': param.default}
        else:
            params_info[param.name] = {'required': True, 'default': None}

    return params_info, has_kwargs
