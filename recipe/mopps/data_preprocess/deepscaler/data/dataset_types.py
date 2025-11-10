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
"""Dataset type definitions for DeepScaler.

This module defines enums for training and testing datasets used in DeepScaler,
as well as a union type for both dataset types.
"""

import enum
from typing import Union


class TrainDataset(enum.Enum):
    """Enum for training datasets.

    Contains identifiers for various math problem datasets used during training.
    """

    AIME = "AIME"  # American Invitational Mathematics Examination
    AMC = "AMC"  # American Mathematics Competition
    OMNI_MATH = "OMNI_MATH"  # Omni Math
    NUMINA_OLYMPIAD = "OLYMPIAD"  # Unique Olympiad problems from NUMINA
    MATH = "MATH"  # Dan Hendrycks Math Problems
    STILL = "STILL"  # STILL dataset
    DEEPSCALER = "DEEPSCALER"  # DeepScaler (AIME, AMC, OMNI_MATH, MATH, STILL)


class TestDataset(enum.Enum):
    """Enum for testing/evaluation datasets.

    Contains identifiers for datasets used to evaluate model performance.
    """

    AIME = "AIME"  # American Invitational Mathematics Examination
    AMC = "AMC"  # American Mathematics Competition
    MATH = "MATH"  # Math 500 problems
    MINERVA = "MINERVA"  # Minerva dataset
    OLYMPIAD_BENCH = "OLYMPIAD_BENCH"  # Olympiad benchmark problems


"""Type alias for either training or testing dataset types."""
Dataset = Union[TrainDataset, TestDataset]
