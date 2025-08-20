from typing import Optional, Tuple
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
) -> Tuple[torch.Tensor, None]:
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