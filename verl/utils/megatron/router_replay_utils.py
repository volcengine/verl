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
Router Replay Utilities
Utilities for handling router replay functionality in Megatron models.
"""

import torch
from megatron.core.tensor_parallel import gather_from_sequence_parallel_region, scatter_to_sequence_parallel_region

from verl.models.mcore.util import postprocess_packed_seqs, preprocess_packed_seqs
from verl.utils.megatron.router_replay_patch import RouterReplay, RoutingMode


def merge_router_topk_indices(attention_mask, input_ids, mini_layer_topk_idx_list):
    """
    Merge recorded topk indices from all router instances.

    Args:
        attention_mask: Attention mask tensor
        input_ids: Input IDs tensor
        mini_layer_topk_idx_list: List to append the merged topk indices

    Returns:
        None (appends result to mini_layer_topk_idx_list)
    """
    with torch.no_grad():
        layers_topk_idx = []
        for router in RouterReplay.router_instances:
            layers_topk_idx.append(router.recorded_topk_idx)  # dynamic_bs, topk

        # layer_num, dynamic_bs, topk  -> dynamic_bs, layer_num, topk
        layers_topk_idx = torch.stack(layers_topk_idx).permute(1, 0, 2).cuda()

        # dynamic_bs, layer_num, topk -> 1, dynamic_bs_all, layer_num, topk
        layers_topk_idx = gather_from_sequence_parallel_region(
            layers_topk_idx, tensor_parallel_output_grad=False
        ).unsqueeze(0)

        batch_size, seq_len = attention_mask.shape[:2]
        _, packed_seq_params = preprocess_packed_seqs(input_ids, attention_mask, pre_process=True)
        layers_topk_idx = postprocess_packed_seqs(
            layers_topk_idx, packed_seq_params, attention_mask, batch_size, seq_len, post_process=True
        )

        mini_layer_topk_idx_list.append(layers_topk_idx)


def set_router_replay_data(layers_topk_idx, attention_mask):
    """
    Set router replay data for replay mode.

    Args:
        layers_topk_idx: Topk indices tensor to be set as replay data
        attention_mask: Attention mask tensor

    Returns:
        None (sets replay data in RouterReplay)
    """
    with torch.no_grad():
        layers_topk_idx_rmpad, _ = preprocess_packed_seqs(layers_topk_idx, attention_mask, pre_process=True)
        layers_topk_idx_rmpad = layers_topk_idx_rmpad.contiguous()  # 1, dynamic_bs_all, layer_num, topk

        # 1, dynamic_bs_split, layer_num, topk
        layers_topk_idx_rmpad_split = scatter_to_sequence_parallel_region(
            layers_topk_idx_rmpad.cuda().squeeze(dim=0)
        ).unsqueeze(dim=0)

        mask = layers_topk_idx_rmpad_split.sum(dim=1) == 0

        # dynamic_bs_split, layer_num, topk -> layer_num, dynamic_bs_split, topk
        layers_topk_idx_reshape = layers_topk_idx_rmpad_split.permute(0, 2, 1, 3).squeeze(
            dim=0
        )  # layer_num, dynamic_bs_all, topk

        RouterReplay.set_replay_data(list(layers_topk_idx_reshape.to(torch.int64)))


class RouterReplayHelper:
    """Helper class to simplify router replay mode checking."""

    @staticmethod
    def is_r2_record_mode(router_replay) -> bool:
        """Check if current mode is R2 RECORD."""
        return router_replay.mode == "R2" and RouterReplay.router_instances[0].routing_mode == RoutingMode.RECORD

    @staticmethod
    def is_replay_forward_mode() -> bool:
        """Check if current mode is REPLAY_FORWARD."""
        return RouterReplay.router_instances[0].routing_mode == RoutingMode.REPLAY_FORWARD

    @staticmethod
    def is_replay_backward_mode() -> bool:
        """Check if current mode is REPLAY_BACKWARD."""
        return RouterReplay.router_instances[0].routing_mode == RoutingMode.REPLAY_BACKWARD

    @staticmethod
    def is_r2_or_r3_mode(router_replay) -> bool:
        """Check if current mode is R2 or R3."""
        return router_replay.mode in ["R2", "R3"]
