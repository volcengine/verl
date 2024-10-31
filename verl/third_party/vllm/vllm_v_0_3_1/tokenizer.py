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

from typing import List, Optional, Tuple, Union

from transformers import (AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast)

from vllm.lora.request import LoRARequest
from vllm.utils import make_async, LRUCache
from vllm.transformers_utils.tokenizers import *


class TokenizerGroup:
    """A group of tokenizers that can be used for LoRA adapters."""

    def __init__(self, tokenizer: PreTrainedTokenizer, enable_lora: bool, max_num_seqs: int,
                 max_input_length: Optional[int]):
        self.enable_lora = enable_lora
        self.max_input_length = max_input_length
        self.tokenizer = tokenizer
        if enable_lora:
            self.lora_tokenizers = LRUCache(capacity=max_num_seqs)
        else:
            self.lora_tokenizers = None

    def encode(self,
               prompt: str,
               request_id: Optional[str] = None,
               lora_request: Optional[LoRARequest] = None) -> List[int]:
        tokenizer = self.get_lora_tokenizer(lora_request)
        return tokenizer.encode(prompt)

    async def encode_async(self,
                           prompt: str,
                           request_id: Optional[str] = None,
                           lora_request: Optional[LoRARequest] = None) -> List[int]:
        tokenizer = await self.get_lora_tokenizer_async(lora_request)
        return tokenizer.encode(prompt)

    def get_lora_tokenizer(self, lora_request: Optional[LoRARequest]) -> "PreTrainedTokenizer":
        if not lora_request or not self.enable_lora:
            return self.tokenizer
        if lora_request.lora_int_id not in self.lora_tokenizers:
            # TODO(sgm): the lora tokenizer is also passed, but may be different
            tokenizer = self.tokenizer
            # tokenizer = (get_lora_tokenizer(
            #     lora_request, **self.tokenizer_config) or self.tokenizer)
            self.lora_tokenizers.put(lora_request.lora_int_id, tokenizer)
            return tokenizer
        else:
            return self.lora_tokenizers.get(lora_request.lora_int_id)

    # FIXME(sgm): for simplicity, we assign the special token here
    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id
