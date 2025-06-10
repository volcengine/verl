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
Apply monkey-patch function to models
"""

import importlib.metadata
import sys
from functools import lru_cache
from typing import Optional

import torch
from packaging import version
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.modeling_utils import PreTrainedModel

from verl.utils.import_utils import is_trl_available
from verl.utils.ulysses import (
    gather_heads_scatter_seq,
    gather_seq_scatter_heads,
    get_ulysses_sequence_parallel_group,
    get_ulysses_sequence_parallel_world_size,
    slice_input_tensor,
)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=2, repeats=n_rep). The hidden states go from (batch,
    seqlen, num_key_value_heads, head_dim) to (batch, seqlen, num_attention_heads, head_dim)
    """
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, :, None, :].expand(batch, slen, num_key_value_heads, n_rep, head_dim)
    return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep, head_dim)


def _ulysses_flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    *args,
    position_ids: Optional[torch.Tensor] = None,
    **kwargs,
):
    """Insert all-to-all before and after flash attention.
    DeepSpeed-Ulysses: https://arxiv.org/pdf/2309.14509

    Args:
        query_states (torch.Tensor): (batch_size, seqlen/sp_size, nheads, head_dim)
        key_states (torch.Tensor): (batch_size, seqlen/sp_size, nheads_k, head_dim)
        value_states (torch.Tensor): (batch_size, seqlen/sp_size, nheads_k, head_dim)
        position_ids (torch.Tensor, optional): (batch_size, seqlen/sp_size)

    Returns:
        torch.Tensor: (batch_size, seqlen/sp_size, nheads, head_dim)
    """
    ulysses_sp_size = get_ulysses_sequence_parallel_world_size()

    ########## AlltoAll for Ulysses ##########
    if ulysses_sp_size > 1:
        assert position_ids is not None, "position_ids is required for Ulysses sequence parallelism"

        # NOTE: repeat kv heads to be divided by sequence parallel. Instead of repeating nheads_q//nheads_k,
        # we choose to repeat sp_size//nheads_k, since flash_attention supports MQA/GQA.
        # For example:
        # - nheads_k=4, sp=8, repeats=2
        # - nheads_k=8, sp=8, repeats=1
        # - nheads_k=16, sp=8, repeats=1
        repeats = max(ulysses_sp_size // key_states.size(2), 1)
        key_states = repeat_kv(key_states, repeats)
        value_states = repeat_kv(value_states, repeats)

        # (bsz, seq_len/n, n_head, head_dim) -> (bsz, seq_len, n_head/n, head_dim)
        query_states = gather_seq_scatter_heads(query_states, seq_dim=1, head_dim=2)
        key_states = gather_seq_scatter_heads(key_states, seq_dim=1, head_dim=2)
        value_states = gather_seq_scatter_heads(value_states, seq_dim=1, head_dim=2)

        # TODO: all_gather position_ids because `prepare_fa2_from_position_ids` needs it, we can eliminate
        # this all_gather by passing cu_seq_lens_q, cu_seq_lens_k, max_length_k, max_length_q explicitly.
        # https://github.com/huggingface/transformers/pull/33932

        # (bsz, seq_len/n) -> (bsz, seq_len)
        position_ids_list = [torch.empty_like(position_ids) for _ in range(ulysses_sp_size)]
        torch.distributed.all_gather(position_ids_list, position_ids, group=get_ulysses_sequence_parallel_group())
        position_ids = torch.concat(position_ids_list, dim=-1)

    # (bsz, seq_len, n_head/n, head_dim)
    attn_output = _flash_attention_forward(query_states, key_states, value_states, *args, position_ids=position_ids, **kwargs)

    ########## AlltoAll for Ulysses ##########
    if ulysses_sp_size > 1:
        # (bsz, seq_len, n_head/n, head_dim) -> (bsz, seq_len/n, n_head, head_dim)
        attn_output = gather_heads_scatter_seq(attn_output, seq_dim=1, head_dim=2)

    return attn_output


def patch_vlm_for_ulysses_input_slicing(model_class: type):
    """
    Applies a monkey patch to the forward method of a given model class
    to enable Ulysses sequence parallelism input slicing.
    """

    def _create_ulysses_wrapped_decoder_forward(original_forward):
        def ulysses_wrapped_decoder_forward(self, *args, **kwargs):
            inputs_embeds = kwargs.get("inputs_embeds")
            call_kwargs = kwargs.copy()

            current_ulysses_sp_size = get_ulysses_sequence_parallel_world_size()

            slice_now = inputs_embeds is not None and current_ulysses_sp_size > 1 and getattr(self, "_needs_initial_slice", True)
            if slice_now:
                call_kwargs["inputs_embeds"] = slice_input_tensor(inputs_embeds, dim=1, padding=False)
                self._needs_initial_slice = False
            try:
                return original_forward(self, *args, **call_kwargs)
            finally:
                if slice_now:
                    self._needs_initial_slice = True

        return ulysses_wrapped_decoder_forward

    original_forward = model_class.forward
    wrapped_forward = _create_ulysses_wrapped_decoder_forward(original_forward)
    model_class.forward = wrapped_forward
    print(f"Monkey patch {model_class.__name__}.forward for Ulysses SP input slicing.")


def apply_monkey_patch(
    model: PreTrainedModel,
    ulysses_sp_size: int = 1,
    use_remove_padding: bool = True,
    use_fused_kernels: bool = False,
):
    """Replace _flash_attention_forward to _ulysses_flash_attention_forward"""
    module = sys.modules[model.__module__]

    try:
        num_attention_heads, num_key_value_heads = model.config.num_attention_heads, model.config.num_key_value_heads
    except AttributeError:
        num_attention_heads, num_key_value_heads = model.config.text_config.num_attention_heads, model.config.text_config.num_key_value_heads
    
    assert num_attention_heads % ulysses_sp_size == 0, f"num_attention_heads {num_attention_heads} must be divisible by ulysses_sp_size {ulysses_sp_size}"
    assert num_key_value_heads % ulysses_sp_size == 0 or ulysses_sp_size % num_key_value_heads == 0, (
        f"num_key_value_heads {num_key_value_heads} must be divisible by ulysses_sp_size {ulysses_sp_size}or vise versa. Upon ulysses_sp_size % num_key_value_heads == 0,kv heads are repeated to ensure correctness."
    )

    if is_trl_available():
        from trl import AutoModelForCausalLMWithValueHead

        def state_dict(self, *args, **kwargs):
            return torch.nn.Module.state_dict(self, *args, **kwargs)

        AutoModelForCausalLMWithValueHead.state_dict = state_dict
        print("Monkey patch state_dict in AutoModelForCausalLMWithValueHead. ")

    # TODO: VLM models only, unify monkey patch to LLM models.
    if model.config.model_type == "qwen2_5_vl":
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            Qwen2_5_VLFlashAttention2,
            Qwen2_5_VLForConditionalGeneration,
        )

        if use_remove_padding or ulysses_sp_size > 1:
            from verl.models.transformers.qwen2_vl import ulysses_flash_attn_forward

            Qwen2_5_VLFlashAttention2.forward = ulysses_flash_attn_forward
            print("Monkey patch FlashAttention2.forward in Qwen2.5VL")

        if ulysses_sp_size > 1:
            if is_transformers_version_in_range(min_version="4.52.0"):
                from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLTextModel
                patch_vlm_for_ulysses_input_slicing(Qwen2_5_VLTextModel)
            else:
                from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel
                patch_vlm_for_ulysses_input_slicing(Qwen2_5_VLModel)

        if use_fused_kernels:
            from verl.models.transformers.qwen2_5_vl import forward_for_ppo

            Qwen2_5_VLForConditionalGeneration.forward = forward_for_ppo

        return

    elif model.config.model_type == "qwen2_vl":
        from transformers.models.qwen2_vl.modeling_qwen2_vl import (
            Qwen2VLFlashAttention2,
            Qwen2VLForConditionalGeneration,
        )

        if use_remove_padding or ulysses_sp_size > 1:
            from verl.models.transformers.qwen2_vl import ulysses_flash_attn_forward

            Qwen2VLFlashAttention2.forward = ulysses_flash_attn_forward
            print("Monkey patch FlashAttention2.forward in Qwen2VL")

        if ulysses_sp_size > 1:
            if is_transformers_version_in_range(min_version="4.52.0"):
                from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLTextModel
                patch_vlm_for_ulysses_input_slicing(Qwen2VLTextModel)
            else:
                from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLModel
                patch_vlm_for_ulysses_input_slicing(Qwen2VLModel)

        if use_fused_kernels:
            from verl.models.transformers.qwen2_vl import forward_for_ppo

            Qwen2VLForConditionalGeneration.forward = forward_for_ppo

        return

    elif model.config.model_type == "kimi_vl":
        if use_remove_padding or ulysses_sp_size > 1:
            # TODO: Changes need to be made when transformers are adapted.
            from verl.models.transformers.kimi_vl import _ulysses_flash_attn_forward

            module.DeepseekV3FlashAttention2.forward = _ulysses_flash_attn_forward
            print("Monkey patch FlashAttention2.forward in KimiVL")
            
        if ulysses_sp_size > 1:
            patch_vlm_for_ulysses_input_slicing(module.DeepseekV3ForCausalLM)
            
        if use_fused_kernels:
            print("Not support fused kernels for KimiVL")

        return

    # transformers<=4.47.1
    if use_remove_padding or ulysses_sp_size > 1:
        if hasattr(module, "_flash_attention_forward"):
            module._flash_attention_forward = _ulysses_flash_attention_forward
            print(f"Monkey patch _flash_attention_forward in {model.__module__}")
        else:
            # transformers>=4.48.0
            from transformers.integrations import flash_attention

            flash_attention._flash_attention_forward = _ulysses_flash_attention_forward
            print(f"Monkey patch _flash_attention_forward in {flash_attention.__name__}")

    if use_fused_kernels:
        from verl.models.transformers.llama import forward_for_ppo

        model.__class__.forward = forward_for_ppo


@lru_cache
def is_transformers_version_in_range(min_version: Optional[str] = None, max_version: Optional[str] = None) -> bool:
    try:
        # Get the installed version of the transformers library
        transformers_version_str = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError as e:
        raise ModuleNotFoundError("The `transformers` package is not installed.") from e

    transformers_version = version.parse(transformers_version_str)

    lower_bound_check = True
    if min_version is not None:
        lower_bound_check = version.parse(min_version) <= transformers_version

    upper_bound_check = True
    if max_version is not None:
        upper_bound_check = transformers_version <= version.parse(max_version)

    return lower_bound_check and upper_bound_check
