# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023 The vLLM team.
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
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/transformers_utils/tokenizer_group/tokenizer_group.py

from typing import Optional

from transformers import PreTrainedTokenizer
from vllm.transformers_utils.tokenizer_group import TokenizerGroup
from vllm.utils import LRUCache


class TokenizerGroup(TokenizerGroup):
    """A group of tokenizers that can be used for LoRA adapters."""

    def __init__(self, tokenizer: PreTrainedTokenizer, enable_lora: bool, max_num_seqs: int,
                 max_input_length: Optional[int]):
        self.enable_lora = enable_lora
        self.max_input_length = max_input_length
        self.tokenizer = tokenizer
        self.lora_tokenizers = LRUCache[PreTrainedTokenizer](capacity=max_num_seqs) if enable_lora else None

    # FIXME(sgm): for simplicity, we assign the special token here
    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id
