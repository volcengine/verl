# Copyright (c) 2024 Alibaba PAI and Nvidia Megatron-LM Team.
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

from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from megatron.core.transformer.transformer_block import TransformerBlock
    from megatron.core.transformer.transformer_config import TransformerConfig

import logging
from copy import deepcopy

import torch
from megatron.core import parallel_state
from megatron.core.models.common.embeddings.rope_utils import get_pos_emb_on_this_cp_rank
from torch import Tensor, nn

logger = logging.getLogger(__name__)


__all__ = ["Qwen2VLRotaryEmbedding"]


class Qwen2VLRotaryEmbedding(nn.Module):
    """Rotary Embedding for language model.

    Args:
        kv_channels (int): Projection weights dimension in multi-head attention. Obtained from transformer config
        rotary_percent (float): Percent of rotary dimension to use for rotary position embeddings.
        seq_len_interpolation_factor (float, optional): scale of linearly interpolating RoPE for longer sequences. The value must be a float larger than 1.0. Defaults to None
        rotary_base (int, optional): Base period for rotary position embeddings. Defaults to 10000.
    """

    def __init__(
        self,
        kv_channels: int,
        rotary_percent: float,
        rotary_interleaved: bool = False,
        seq_len_interpolation_factor: float = None,
        rotary_base: int = 10000,
        mrope_section: List[int] = None,
    ) -> None:
        super().__init__()

        dim = kv_channels
        if rotary_percent < 1.0:
            dim = int(dim * rotary_percent)
        self.rotary_interleaved = rotary_interleaved

        self.seq_len_interpolation_factor = seq_len_interpolation_factor
        self.inv_freq = 1.0 / (
            rotary_base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=torch.cuda.current_device()) / dim)
        )
        self.mrope_section = mrope_section

    def forward(self, position_ids: torch.Tensor, offset: int = 0, packed_seq: bool = False) -> Tensor:
        """Forward pass of multimodal RoPE embedding.

        Args:
            position_ids (torch.Tensor): A postion_id tensor with shape [3, batchsize, seqlens]
            mrope_section (list[int]): Multimodal rope section is for channel dimension of temporal, height and width in rope calculation.

        Returns:
            Tensor: Embeddings after applying RoPE.
        """
        mrope_section = deepcopy(self.mrope_section)
        seq = position_ids.to(device=self.inv_freq.device, dtype=self.inv_freq.dtype)

        if self.seq_len_interpolation_factor is not None:
            seq *= 1 / self.seq_len_interpolation_factor

        inv_freq_expanded = self.inv_freq[None, None, :, None].expand(3, seq.shape[1], -1, 1)  # shape (3, bs, dim, 1)
        seq_expanded = seq[:, :, None, :].float()  # shape (3, bs, 1, seq_length)
        freqs = (inv_freq_expanded @ seq_expanded).transpose(2, 3)  # shape (3, bs, seq_length, dim)

        # first part even vector components, second part odd vector components,
        #  2 * dim in dimension size
        if not self.rotary_interleaved:
            emb = torch.cat((freqs, freqs), dim=-1)  # shape (3, bs, seq_length, 2 * dim)
        else:
            bs = freqs.shape[1]
            emb = torch.stack((freqs.view(3, bs, -1, 1), freqs.view(3, bs, -1, 1)), dim=-1).view(
                3, bs, freqs.shape[0], -1
            )

        # generate freqs with mrope_section
        # shape (bs, seq_length, 2 * dim)
        mrope_section = mrope_section * 2
        emb = torch.cat([m[i % 3] for i, m in enumerate(emb.split(mrope_section, dim=-1))], dim=-1)

        # shape (seq_length, bs, 1, 2 * dim)
        emb = emb[..., None, :].transpose(0, 1).contiguous()
        # if parallel_state.get_context_parallel_world_size() > 1:
        #     # slice rotary_pos_emb along sequence dimension and select the parition of the current CP rank
        #     emb = get_pos_emb_on_this_cp_rank(emb, 1)

        if parallel_state.get_context_parallel_world_size() > 1 and not packed_seq:
            # slice rotary_pos_emb along sequence dimension and select the parition of the current
            # CP rank
            emb = get_pos_emb_on_this_cp_rank(emb, 0)
        return emb

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        state_dict.pop(f"{prefix}inv_freq", None)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def get_rotary_seq_len(
        self,
        inference_params,
        transformer: TransformerBlock,
        transformer_input: Tensor,
        transformer_config: TransformerConfig,
        packed_seq_params: PackedSeqParams,
    ) -> float:
        """Function to get the rotary sequence length.

        Args:
            inference_params : Used during Inference time
            transformer (TransformerBlock): The transformer block (decoder/encoder) used by the model
            transformer_input (Tensor): _description_
            transformer_config (TransformerConfig): Transformer config used by the model

        Returns:
            float: The rotary sequence length
        """
        if packed_seq_params is not None:
            # max_seqlen are the max sequence length in the packed sequence before being divived
            # by the tp and cp size.
            return max(packed_seq_params.max_seqlen_q, packed_seq_params.max_seqlen_kv)
        if inference_params is not None:
            rotary_seq_len = inference_params.max_sequence_length
        else:
            if transformer.input_tensor is not None:
                rotary_seq_len = transformer.input_tensor.size(0)
            else:
                rotary_seq_len = transformer_input.size(0)

            if transformer_config.sequence_parallel:
                rotary_seq_len *= transformer_config.tensor_model_parallel_size

        rotary_seq_len *= transformer_config.context_parallel_size

        return rotary_seq_len
