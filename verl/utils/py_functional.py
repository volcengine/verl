# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Contain small python utility functions
"""

from types import SimpleNamespace
from typing import Dict


def union_two_dict(dict1: Dict, dict2: Dict):
    """Union two dict. Will throw an error if there is an item not the same object with the same key.

    Args:
        dict1:
        dict2:

    Returns:

    """
    for key, val in dict2.items():
        if key in dict1:
            assert dict2[key] == dict1[key], f"{key} in meta_dict1 and meta_dict2 are not the same object"
        dict1[key] = val

    return dict1


def append_to_dict(data: Dict, new_data: Dict):
    """Append values from new_data to lists in data.

    For each key in new_data, this function appends the corresponding value to a list
    stored under the same key in data. If the key doesn't exist in data, a new list is created.

    Args:
        data (Dict): The target dictionary containing lists as values.
        new_data (Dict): The source dictionary with values to append.

    Returns:
        None: The function modifies data in-place.
    """
    for key, val in new_data.items():
        if key not in data:
            data[key] = []
        data[key].append(val)


class NestedNamespace(SimpleNamespace):
    """A nested version of SimpleNamespace that recursively converts dictionaries to namespaces.

    This class allows for dot notation access to nested dictionary structures by recursively
    converting dictionaries to NestedNamespace objects.

    Example:
        config_dict = {"a": 1, "b": {"c": 2, "d": 3}}
        config = NestedNamespace(config_dict)
        # Access with: config.a, config.b.c, config.b.d

    Args:
        dictionary: The dictionary to convert to a nested namespace.
        **kwargs: Additional attributes to set on the namespace.
    """

    def __init__(self, dictionary, **kwargs):
        super().__init__(**kwargs)
        for key, value in dictionary.items():
            if isinstance(value, dict):
                self.__setattr__(key, NestedNamespace(value))
            else:
                self.__setattr__(key, value)
