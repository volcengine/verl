# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""The set of token ids a rollout backend never samples.

The rollout samples from the softmax over the token ids outside this set. The actor and the
reference model evaluate the policy over the same set, so that sampling, log-probabilities,
entropy and the KL term refer to one distribution.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RolloutVocabMask:
    """Token ids excluded from the sampling distribution of a rollout backend.

    Attributes:
        vocab_size: Number of leading token ids the rollout may sample. Every id ``>= vocab_size``
            is excluded.
        banned_token_ids: Ids below ``vocab_size`` that are excluded as well.
    """

    vocab_size: int
    banned_token_ids: tuple[int, ...] = ()

    def banned_mask(self, num_logits: int) -> torch.Tensor | None:
        """Return the boolean mask over ``num_logits`` logits that is ``True`` at every excluded id.

        Returns ``None`` if no id in ``range(num_logits)`` is excluded.

        Raises:
            ValueError: If a banned id lies outside ``range(num_logits)``.
        """
        out_of_range = [token_id for token_id in self.banned_token_ids if not 0 <= token_id < num_logits]
        if out_of_range:
            raise ValueError(f"banned token ids {out_of_range} lie outside the {num_logits} logits of the model")
        mask = torch.zeros(num_logits, dtype=torch.bool)
        mask[self.vocab_size :] = True
        mask[list(self.banned_token_ids)] = True
        if not mask.any():
            return None
        return mask
