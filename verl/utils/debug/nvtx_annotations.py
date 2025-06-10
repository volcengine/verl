# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

import nvtx


def mark_start_range(message=None, color=None, domain=None, category=None):
    """Start a mark range in the profiler.

    Args:
        message (str, optional):
            The message to be displayed in the profiler. Defaults to None.
        color (str, optional):
            The color of the range. Defaults to None.
        domain (str, optional):
            The domain of the range. Defaults to None.
        category (str, optional):
            The category of the range. Defaults to None.
    """
    return nvtx.start_range(message=message, color=color, domain=domain, category=category)


def mark_end_range(range_id):
    """End a mark range in the profiler.

    Args:
        range_id (str):
            The id of the mark range to end.
    """
    return nvtx.end_range(range_id)


def mark_annotate(message=None, color=None, domain=None, category=None):
    """Decorate a function to annotate a mark range along with the function life cycle.

    Args:
        message (str, optional):
            The message to be displayed in the profiler. Defaults to None.
        color (str, optional):
            The color of the range. Defaults to None.
        domain (str, optional):
            The domain of the range. Defaults to None.
        category (str, optional):
            The category of the range. Defaults to None.
    """

    def decorator(func):
        profile_message = message or func.__name__
        return nvtx.annotate(profile_message, color=color, domain=domain, category=category)(func)

    return decorator
