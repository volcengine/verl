#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

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

from dataclasses import dataclass
from typing import Optional

from verl import DataProto


@dataclass
class DynamicFilterState:
    """State tracking for dynamic filtering during batch processing."""

    num_gen_batches: int = 0
    num_prompt_in_batch: int = 0
    accumulated_batch: Optional[DataProto] = None

    def reset(self) -> None:
        """Reset all state variables for the next training step."""
        self.num_gen_batches = 0
        self.num_prompt_in_batch = 0
        self.accumulated_batch = None

    def increment_gen_batches(self) -> None:
        """Increment the generation batch counter."""
        self.num_gen_batches += 1

    def add_prompts(self, count: int) -> None:
        """Add to the prompt count."""
        self.num_prompt_in_batch += count

    def accumulate_batch(self, batch: DataProto) -> None:
        """Accumulate a batch, concatenating with existing if present."""
        self.accumulated_batch = (
            batch if self.accumulated_batch is None else DataProto.concat([self.accumulated_batch, batch])
        )
