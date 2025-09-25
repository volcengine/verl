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

from verl.utils.device import is_cuda_available, is_npu_available

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    try:
        from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input
    except ImportError:
        # Since transformers v4.55.1, index_first_axis, pad_input, and unpad_input
        # have been consolidated into `transformers.modeling_flash_attention_utils`.
        from einops import rearrange
        from transformers.modeling_flash_attention_utils import _index_first_axis as index_first_axis
        from transformers.modeling_flash_attention_utils import _pad_input as pad_input
        from transformers.modeling_flash_attention_utils import _unpad_input as unpad_input
else:
    raise RuntimeError("Unsupported device type")


__all__ = ["index_first_axis", "pad_input", "rearrange", "unpad_input"]
