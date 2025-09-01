# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

from typing import Optional

import torch


def flash_attention_forward(
    self,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float,
    sliding_window: Optional[int] = None,
    position_ids: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    # Use the integration entry which we monkey-patch in monkey_patch.apply_monkey_patch
    from transformers.integrations import flash_attention as hf_flash_attention

    attn_output = hf_flash_attention._flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout,
        sliding_window,
        position_ids=position_ids,
        **kwargs,
    )
    return attn_output, None
