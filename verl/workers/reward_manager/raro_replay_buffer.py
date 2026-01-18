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
Replay Buffer for RARO (Relativistic Adversarial Reasoning Optimization)

Stores historical policy generations to prevent catastrophic forgetting
in the Critic during adversarial training.
"""

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class RAROReplayBuffer:
    """FIFO replay buffer for storing historical (question, expert_answer, policy_answer) triplets.

    This buffer stores past policy generations to mix with current generations
    during Critic training, preventing catastrophic forgetting.

    Attributes:
        capacity: Maximum number of samples to store
        buffer: Deque storing the triplets
        question_ids: Set of tracked question IDs for deduplication
    """

    capacity: int = 10000
    buffer: deque = field(default_factory=lambda: deque(maxlen=10000))
    question_ids: set = field(default_factory=set)

    def __post_init__(self):
        """Initialize buffer with the specified capacity."""
        self.buffer = deque(maxlen=self.capacity)

    def add(
        self,
        question_id: str | int,
        question: str,
        expert_answer: str,
        policy_answer: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Add a new triplet to the buffer.

        Args:
            question_id: Unique identifier for the question
            question: The question text
            expert_answer: The expert/expert answer
            policy_answer: The policy-generated answer
            metadata: Optional additional metadata
        """
        # Avoid duplicate questions
        if question_id in self.question_ids:
            return

        triplet = {
            "question_id": question_id,
            "question": question,
            "expert_answer": expert_answer,
            "policy_answer": policy_answer,
            "metadata": metadata or {},
        }

        self.buffer.append(triplet)
        self.question_ids.add(question_id)

    def sample(self, n_samples: int, seed: Optional[int] = None) -> list[dict[str, Any]]:
        """Sample n_samples triplets from the buffer.

        Args:
            n_samples: Number of samples to draw
            seed: Optional random seed for reproducibility

        Returns:
            List of sampled triplets
        """
        if len(self.buffer) == 0:
            return []

        if seed is not None:
            random.seed(seed)

        n_samples = min(n_samples, len(self.buffer))
        return random.sample(list(self.buffer), n_samples)

    def sample_with_replacement(self, n_samples: int, seed: Optional[int] = None) -> list[dict[str, Any]]:
        """Sample n_samples triplets with replacement.

        Args:
            n_samples: Number of samples to draw
            seed: Optional random seed for reproducibility

        Returns:
            List of sampled triplets (with possible duplicates)
        """
        if len(self.buffer) == 0:
            return []

        if seed is not None:
            random.seed(seed)

        buffer_list = list(self.buffer)
        return [random.choice(buffer_list) for _ in range(n_samples)]

    def clear(self) -> None:
        """Clear all samples from the buffer."""
        self.buffer.clear()
        self.question_ids.clear()

    def __len__(self) -> int:
        """Return the current number of samples in the buffer."""
        return len(self.buffer)

    def is_full(self) -> bool:
        """Check if the buffer has reached capacity."""
        return len(self.buffer) >= self.capacity

    def get_statistics(self) -> dict[str, Any]:
        """Get buffer statistics.

        Returns:
            Dictionary with buffer statistics
        """
        return {
            "current_size": len(self.buffer),
            "capacity": self.capacity,
            "utilization": len(self.buffer) / self.capacity if self.capacity > 0 else 0,
            "unique_questions": len(self.question_ids),
        }


class ReservoirBuffer:
    """Reservoir sampling buffer for uniform sampling from a stream.

    Uses reservoir sampling to maintain a uniform random sample of
    encountered data points, regardless of total stream size.
    """

    def __init__(self, capacity: int):
        """Initialize the reservoir buffer.

        Args:
            capacity: Maximum number of samples to store
        """
        self.capacity = capacity
        self.buffer: list[dict[str, Any]] = []
        self.count = 0

    def add(
        self,
        question_id: str | int,
        question: str,
        expert_answer: str,
        policy_answer: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Add a new triplet using reservoir sampling.

        Args:
            question_id: Unique identifier for the question
            question: The question text
            expert_answer: The expert/expert answer
            policy_answer: The policy-generated answer
            metadata: Optional additional metadata
        """
        triplet = {
            "question_id": question_id,
            "question": question,
            "expert_answer": expert_answer,
            "policy_answer": policy_answer,
            "metadata": metadata or {},
        }

        if len(self.buffer) < self.capacity:
            self.buffer.append(triplet)
        else:
            # Reservoir sampling: replace with probability 1/(count+1)
            r = random.randint(0, self.count)
            if r < self.capacity:
                self.buffer[r] = triplet

        self.count += 1

    def sample(self, n_samples: int, seed: Optional[int] = None) -> list[dict[str, Any]]:
        """Sample n_samples triplets from the buffer.

        Args:
            n_samples: Number of samples to draw
            seed: Optional random seed for reproducibility

        Returns:
            List of sampled triplets
        """
        if len(self.buffer) == 0:
            return []

        if seed is not None:
            random.seed(seed)

        n_samples = min(n_samples, len(self.buffer))
        return random.sample(self.buffer, n_samples)

    def clear(self) -> None:
        """Clear all samples from the buffer."""
        self.buffer.clear()
        self.count = 0

    def __len__(self) -> int:
        """Return the current number of samples in the buffer."""
        return len(self.buffer)
