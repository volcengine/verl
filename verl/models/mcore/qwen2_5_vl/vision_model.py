# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024 Alibaba PAI Team.
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
from megatron.core import InferenceParams
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import nn
from torch.nn import functional as F

from .vision_transformer_block import Qwen2_5VisionTransformerBlock as TransformerBlock


# copied from https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py
class PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(-1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size)
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


# copied from https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py
class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs.float()


class Qwen2_5VisionModel(VisionModule):
    """Qwen2.5 ViT vision model.

    Args:
        transformer_config (TransformerConfig): Transformer config.
        transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers.
        ln_pre_impl (ModuleSpec or type): Specifies the layer norm type to use for ln_pre.
        add_class_token (bool, optional): Include a class token. Defaults to True.
        class_token_len (int): Class token length. Defaults to 1 but 8 may be faster.
        patch_dim (int): Image patch size.
        img_h (int): Input image height.
        img_w (int): Input image width.
    """

    def __init__(self, transformer_config: TransformerConfig, transformer_layer_spec: ModuleSpec, projection_config: TransformerConfig, projection_layer_spec: ModuleSpec, projection_type: str = "mlp", pre_process: bool = True, post_process: bool = False) -> None:
        super().__init__(config=transformer_config)

        self.spatial_merge_size = transformer_config.spatial_merge_size

        embed_dim = transformer_config.hidden_size
        num_heads = transformer_config.num_attention_heads
        temporal_patch_size = transformer_config.temporal_patch_size
        patch_size = transformer_config.patch_size
        in_channels = transformer_config.in_channels

        self.patch_size = transformer_config.patch_size
        self.fullatt_block_indexes = transformer_config.fullatt_block_indexes
        self.window_size = transformer_config._qwen2_5_vl_window_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.max_sequence_length = transformer_config.seq_length
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        head_dim = embed_dim // num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.model_type = ModelType.encoder_or_decoder
        self.pre_process = pre_process
        self.post_process = post_process

        # Transformer layers.
        # TODO: Follow-up changes will make pre and post_process configurable. They are needed for supporting pipeline parallelism.
        # NOTE: a final layer norm and/or linear layer present in some implementations are omitted here.
        self.decoder = TransformerBlock(config=transformer_config, spec=transformer_layer_spec, pre_process=self.pre_process, post_process=self.post_process, post_layer_norm=True)

        self.merge_hidden_size = projection_config.ffn_hidden_size
        self.square_merge_size = self.merge_hidden_size // embed_dim

        if self.post_process:
            self.projection = MultimodalProjector(projection_config, projection_layer_spec, projection_type, projection_config.ffn_hidden_size)
        else:
            self.projection = None

        self.input_tensor = None

    def set_input_tensor(self, input_tensor: torch.Tensor) -> None:
        """Sets input tensor to the model.

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        if self.pre_process:  # always True
            self.input_tensor = input_tensor
        else:
            raise NotImplementedError()

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0).to(grid_thw.device)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size).to(grid_thw.device)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def get_window_index(self, grid_thw):
        window_index: list = []
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = self.window_size // self.spatial_merge_size // self.patch_size

        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (
                grid_h // self.spatial_merge_size,
                grid_w // self.spatial_merge_size,
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)

        return window_index, cu_window_seqlens

    def forward(
        self,
        vision_data: Optional[torch.Tensor],
        grid_thw: torch.Tensor,
        inference_params: Optional[InferenceParams] = None,
        extra_block_kwargs: dict = None,
    ) -> torch.Tensor:
        """Forward function of the Qwen2 Vision Model. This function passes the input tensors
        through the embedding layer and then the transformer.

        Args:
            x (torch.Tensor): input image/video data of shape [n_tokens, n_dims]
            grid_thw (torch.Tensor): the size tensor indicates grid size of each image/frame
            packed_seq_params (PackedSeqParams): parameters to build attention mask in the backend

        Returns:
            x (torch.Tensor): output after final transformer block of shape [b, s, h].
        """
        assert grid_thw is not None
        assert self.input_tensor is None
        assert inference_params is None

        # Rotary positional embeddings (embedding is None for PP intermediate devices)
        vision_data = self.patch_embed(vision_data)
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=vision_data.device,
            dtype=torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = vision_data.size()
        vision_data = vision_data.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        vision_data = vision_data[window_index, :, :]
        vision_data = vision_data.reshape(seq_len, 1, -1)

        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, 1, 1, -1).repeat(1, 1, 1, 2)

        hidden_states = self.decoder(
            hidden_states=vision_data,
            attention_mask=None,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=self.build_packed_seq_params(None, cu_window_seqlens),
            packed_seq_params_full=self.build_packed_seq_params(grid_thw),
            fullatt_block_indexes=self.fullatt_block_indexes,
            **(extra_block_kwargs or {}),
        )

        hidden_states = self.projection(hidden_states.view(-1, self.merge_hidden_size))
        reverse_indices = torch.argsort(window_index)
        return hidden_states[reverse_indices, :]

    def build_packed_seq_params(
        self,
        grid_thw: Optional[torch.Tensor],
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> PackedSeqParams:
        # NOTE: each frame is a sequence (rather than each grid)
        if grid_thw is not None:
            seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
            cu_seqlens = seqlens.cumsum(dim=0)
            cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0).int()
        else:
            seqlens = cu_seqlens[1:] - cu_seqlens[:-1]

        max_seqlen_q = seqlens.max()
        return PackedSeqParams(cu_seqlens_q=cu_seqlens, cu_seqlens_kv=cu_seqlens, qkv_format="thd", max_seqlen_q=max_seqlen_q, max_seqlen_kv=max_seqlen_q)
