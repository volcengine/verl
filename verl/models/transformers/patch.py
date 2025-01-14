import os
import torch
from typing import Optional, List, Union, Tuple, Unpack, Callable


from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, ALL_ATTENTION_FUNCTIONS, eager_attention_forward
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.cache_utils import Cache
from transformers.utils import logging

from verl.utils.ulysses import gather_heads_scatter_seq, gather_seq_scatter_heads, get_ulysses_sequence_parallel_world_size

logger = logging.get_logger(__name__)

def llama_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim) # (bsz, seq_len, n_head, head_dim)

        # from (bsz, seq_len/n, n_head, head_dim) -> (bsz, n_head, seq_len/n, head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        ########## AlltoAll for Ulysses ##########
        ulysses_sp_size = get_ulysses_sequence_parallel_world_size()
        if ulysses_sp_size > 1:
            # (bsz, n_head, seq_len/n, head_dim) -> (bsz, n_head/n, seq_len, head_dim)
            query_states = gather_seq_scatter_heads(query_states, seq_dim=2, head_dim=1)
            key_states = gather_seq_scatter_heads(key_states, seq_dim=2, head_dim=1)
            value_states = gather_seq_scatter_heads(value_states, seq_dim=2, head_dim=1)

        # the position_embeddings are computed by full postition_ids
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        # after reshape: (bsz, seq_len, n_head/n, head_dim)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        if ulysses_sp_size > 1:
            attn_output = gather_heads_scatter_seq(attn_output, seq_dim=1, head_dim=2)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights