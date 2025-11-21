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
from copy import deepcopy

from megatron.core.tensor_parallel import gather_from_sequence_parallel_region, scatter_to_sequence_parallel_region
from megatron.core import parallel_state as mpu
from megatron.core.pipeline_parallel.schedules import get_schedule_table
# from megatron.core.transformer.transformer_block import get_num_layers_to_build
# from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

from verl.models.mcore.util import postprocess_packed_seqs, preprocess_packed_seqs
from verl.utils.megatron.router_replay_patch import RouterReplay, RoutingMode

def merge_router_topk_indices(attention_mask, input_ids, mini_layer_topk_idx_list, tf_config, vp_rank=None):
    """
    Merge recorded router top-k indices across sequence-parallel ranks for all router instances,
    then pack/unpack them to align with the original (batch, seq_len) layout and append the result.

    Args:
        attention_mask (torch.Tensor): Attention mask of shape [batch_size, seq_len]. Used to determine
            the valid token positions during pack/unpack.
        input_ids (torch.Tensor): Input token IDs of shape [batch_size, seq_len]. Used together with
            attention_mask for sequence packing/unpacking.
        mini_layer_topk_idx_list (list): A Python list to which the merged top-k indices tensor will be appended.
        tf_config: Megatron/Transformer engine configuration object. Used to locate router instances for
            the current micro-batch.
        vp_rank (Optional[int]): Virtual pipeline stage rank override. If None, the current VP rank from
            Megatron parallel state will be used.

    Returns:
        None: The function has side effects only; it appends a tensor of shape
        [1, dynamic_bs_all, layer_num, topk] to mini_layer_topk_idx_list.
    """
    with torch.no_grad():
        router_instances_list = RouterReplayHelper.get_micro_batch_router_list(tf_config, vp_rank)
        layers_topk_idx = []
        for router in router_instances_list:
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


def set_router_replay_data(layers_topk_idx, attention_mask, tf_config, vp_rank=None):
    """
    Scatter the packed router top-k indices back to sequence-parallel ranks and update each local
    RouterReplay instance with target indices for replay mode.

    This function prepares the per-layer, per-sample top-k routing decisions (recorded during an earlier
    forward) so that subsequent replay passes can follow exactly the same routing.

    Args:
        layers_topk_idx (torch.Tensor): Router top-k indices with shape [bs, max_seq_len, layer_num, topk].
            This should be the merged output produced by merge_router_topk_indices.
        attention_mask (torch.Tensor): Attention mask [batch_size, seq_len] used for pack/unpack alignment.
        tf_config: Megatron/Transformer engine configuration object.
        vp_rank (Optional[int]): Virtual pipeline stage rank override. If None, the current VP rank from
            Megatron parallel state will be used.

    Returns:
        None: The function updates internal RouterReplay instances in-place.
    """
    with torch.no_grad():
        layers_topk_idx_rmpad, _ = preprocess_packed_seqs(layers_topk_idx, attention_mask, pre_process=True)
        layers_topk_idx_rmpad = layers_topk_idx_rmpad.contiguous()  # 1, dynamic_bs_all, layer_num, topk

        # 1, dynamic_bs_split, layer_num, topk
        layers_topk_idx_rmpad_split = scatter_to_sequence_parallel_region(
            layers_topk_idx_rmpad.cuda().squeeze(dim=0)
        ).unsqueeze(dim=0)

        # dynamic_bs_split, layer_num, topk -> layer_num, dynamic_bs_split, topk
        layers_topk_idx_reshape = layers_topk_idx_rmpad_split.permute(0, 2, 1, 3).squeeze(
            dim=0
        )  # layer_num, dynamic_bs_all, topk
        # num_layers_to_build = get_num_layers_to_build(tf_config, vp_stage=vp_rank)
        # offset = get_transformer_layer_offset(tf_config, vp_stage=vp_rank)
        local_rank,_ = get_current_rank_layer_info(tf_config,vp_rank)
        offset, _ = local_rank["start"], local_rank["end"]
        router_instances_list = RouterReplayHelper.get_micro_batch_router_list(tf_config, vp_rank)
        for i,router in enumerate(router_instances_list):
            router.set_target_indices(layers_topk_idx_reshape[i+offset].to(torch.int64))

def reorder_and_merge_vpp_layers(
    t: torch.Tensor,
    num_microbatches: int,
    vpp_size: int,
    microbatch_group_size_per_vp_stage: int,
) -> torch.Tensor:
    """
    Reorder and merge per-VPP layer blocks into a contiguous layer dimension.

    Given a tensor shaped as [bs*vpp_size, max_token_len, layer_num_per_vpp, topk], this function:
    1) Builds the schedule table for virtual microbatches and reorders the first dimension so that entries
       belonging to the same model chunk (VPP stage) become contiguous.
    2) Reshapes and merges the (vpp_size, layer_num_per_vpp) into a single layer dimension, producing
       [bs, max_token_len, layer_num, topk].

    Args:
        t (torch.Tensor): Input tensor of shape [bs*vpp_size, max_token_len, layer_num_per_vpp, topk].
        num_microbatches (int): Number of microbatches per pipeline stage (bs).
        vpp_size (int): Virtual pipeline parallel size (number of model chunks).
        microbatch_group_size_per_vp_stage (int): Number of consecutive microbatches processed per VPP stage.

    Returns:
        torch.Tensor: Output tensor of shape [bs, max_token_len, layer_num, topk].

    Raises:
        ValueError: If input tensor dimensionality or expected sizes do not match.
        RuntimeError: If the computed output shape is unexpected or the schedule length mismatches.
    """
    if t.dim() != 4:
        raise ValueError(f"expect a 4D tensor, got dim={t.dim()} and shape={tuple(t.shape)}")
    bs_vpp, max_token_len, layer_per_vpp, topk = t.shape

    expected_bs_vpp = num_microbatches * vpp_size
    if bs_vpp != expected_bs_vpp:
        raise ValueError(
            f"first dim (bs*vpp_size={bs_vpp}) must equal num_microbatches*vpp_size={expected_bs_vpp}"
        )
    if vpp_size <= 0:
        raise ValueError(f"vpp_size must be positive, got {vpp_size}")

    # 1) Build schedule table: map each virtual_microbatch_id -> (microbatch_id, model_chunk_id)
    schedule_table = get_schedule_table(num_microbatches, vpp_size, microbatch_group_size_per_vp_stage)
    if len(schedule_table) != expected_bs_vpp:
        raise RuntimeError(
            f"schedule_table length {len(schedule_table)} mismatch total virtual microbatches {expected_bs_vpp}"
        )

    # 2) Group by model_chunk_id to build reorder indices so entries of the same chunk become contiguous along dim 0
    indices_by_chunk = [[] for _ in range(vpp_size)]
    for vidx, (_mb, chunk_id) in enumerate(schedule_table):
        indices_by_chunk[chunk_id].append(vidx)
    reorder_indices = [idx for chunk_id in range(vpp_size) for idx in indices_by_chunk[chunk_id]]

    index_tensor = torch.tensor(reorder_indices, dtype=torch.long, device=t.device)
    t_reordered = torch.index_select(t, dim=0, index=index_tensor)

    # 3) After reordering, reshape and merge along the layer dimension
    bs = num_microbatches
    # View: [vpp_size, bs, max_token_len, layer_per_vpp, topk]
    t_view = t_reordered.contiguous().view(vpp_size, bs, max_token_len, layer_per_vpp, topk)
    # Permute dims -> [bs, max_token_len, vpp_size, layer_per_vpp, topk]
    t_perm = t_view.permute(1, 2, 0, 3, 4).contiguous()
    # Merge (vpp_size, layer_per_vpp) -> layer_num
    out = t_perm.view(bs, max_token_len, vpp_size * layer_per_vpp, topk)

    # Shape check
    if out.shape != (bs, max_token_len, vpp_size * layer_per_vpp, topk):
        raise RuntimeError(
            f"unexpected output shape {tuple(out.shape)}; "
            f"expected ({bs}, {max_token_len}, {vpp_size * layer_per_vpp}, {topk})"
        )
    return out


def compute_pipeline_layer_assignment(tf_config):
    # TODO: Consider using Megatron's official API if available.
    """
    Compute the global layer index ranges and counts assigned to each pipeline-parallel rank and
    (optionally) each virtual pipeline stage.

    The function supports both uniform splitting (equal layers per PP rank) and non-uniform splitting
    when first/last stage layer counts are provided. If virtual pipeline is enabled, each physical PP
    rank's layers are further evenly divided into "vp_size" virtual stages.

    Args:
        tf_config: Configuration object that provides:
            - num_layers (int): Total number of transformer layers.
            - pipeline_model_parallel_size (int): Number of pipeline-parallel ranks (PP size).
            - virtual_pipeline_model_parallel_size (Optional[int]): Virtual pipeline size (VP size).
            - num_layers_in_first_pipeline_stage (Optional[int]): Layers in the first PP stage.
            - num_layers_in_last_pipeline_stage (Optional[int]): Layers in the last PP stage.

    Returns:
        dict[tuple[int, int], dict]: A mapping from (pp_rank, vp_stage) to a dict with keys:
            - "start" (int): Inclusive start layer index in the global model.
            - "end" (int): Exclusive end layer index.
            - "count" (int): Number of layers in the range.
    """
    num_layers = tf_config.num_layers
    pp_size = tf_config.pipeline_model_parallel_size
    vp_size = tf_config.virtual_pipeline_model_parallel_size
    first_stage_layers = tf_config.num_layers_in_first_pipeline_stage
    last_stage_layers = tf_config.num_layers_in_last_pipeline_stage

    assignments = {}

    if pp_size is None or pp_size <= 1:
        # Single pipeline stage; no splitting required.
        if vp_size is not None and vp_size > 1:
            # Virtual pipeline requires even splitting within each physical stage.
            assert num_layers % vp_size == 0, "num_layers must be divisible by vp_size"
            per_vp = num_layers // vp_size
            offset = 0
            for j in range(vp_size):
                assignments[(0, j)] = {"start": offset, "end": offset + per_vp, "count": per_vp}
                offset += per_vp
        else:
            assignments[(0, 0)] = {"start": 0, "end": num_layers, "count": num_layers}
        return assignments

    # Uniform/non-uniform splitting without layout-specific configuration
    if first_stage_layers is None and last_stage_layers is None:
        # Evenly split layers across physical PP ranks
        assert num_layers % pp_size == 0, "num_layers must be divisible by pp_size"
        per_rank_total = num_layers // pp_size
        if vp_size is None:
            # No virtual pipeline
            for i in range(pp_size):
                start_i = i * per_rank_total
                end_i = start_i + per_rank_total
                assignments[(i, 0)] = {"start": start_i, "end": end_i, "count": per_rank_total}
        else:
            # With virtual pipeline: evenly split each physical stage into vp_size sub-stages
            assert per_rank_total % vp_size == 0, "Layers per rank must be divisible by vp_size"
            per_vp = per_rank_total // vp_size
            for i in range(pp_size):
                base = i * per_rank_total
                for j in range(vp_size):
                    start_ij = base + j * per_vp
                    assignments[(i, j)] = {"start": start_ij, "end": start_ij + per_vp, "count": per_vp}
        return assignments
    else:
        # Non-uniform first/last stage configuration
        assert first_stage_layers is not None and last_stage_layers is not None, "Both first and last stage layer counts must be provided"
        assert pp_size >= 2, "Non-uniform splitting requires at least two physical stages"
        mid_ranks = pp_size - 2
        remaining = num_layers - first_stage_layers - last_stage_layers
        if mid_ranks > 0:
            assert remaining % mid_ranks == 0, "Total layers in middle stages must be evenly distributed across middle ranks"
            per_mid_total = remaining // mid_ranks
        else:
            per_mid_total = 0

        # Compute total layer range for each physical stage
        start = 0
        phys_ranges = []  # [(start, end)] for i in [0..pp_size-1]
        for i in range(pp_size):
            if i == 0:
                total_i = first_stage_layers
            elif i == pp_size - 1:
                total_i = last_stage_layers
            else:
                total_i = per_mid_total
            phys_ranges.append((start, start + total_i))
            start += total_i

        # Virtual pipeline splits each physical stage by VP size
        if vp_size is None:
            for i in range(pp_size):
                s, e = phys_ranges[i]
                assignments[(i, 0)] = {"start": s, "end": e, "count": e - s}
        else:
            for i in range(pp_size):
                s, e = phys_ranges[i]
                total_i = e - s
                assert total_i % vp_size == 0, "Layers in a physical stage must be divisible by vp_size"
                per_vp = total_i // vp_size
                for j in range(vp_size):
                    start_ij = s + j * per_vp
                    assignments[(i, j)] = {"start": start_ij, "end": start_ij + per_vp, "count": per_vp}
        return assignments


def get_current_rank_layer_info(tf_config, vp_rank = None):
    # When vp_rank is None, default to the current VP rank (or 0 if VP is disabled).
    """Return the local layer range/count for the current process and the full assignment table.

    Args:
        tf_config: Configuration object used by compute_pipeline_layer_assignment.
        vp_rank (Optional[int]): Explicit virtual pipeline stage rank to query. If None, uses
            mpu.get_virtual_pipeline_model_parallel_rank() when VP is enabled; otherwise 0.

    Returns:
        Tuple[dict, dict]: A tuple of (local_assignment, all_assignments) where local_assignment contains
        keys {"start", "end", "count"} for the current (pp_rank, vp_stage).
    """
    assignments = compute_pipeline_layer_assignment(tf_config)
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    vp_size = tf_config.virtual_pipeline_model_parallel_size
    if vp_rank == None:
        vp_rank = mpu.get_virtual_pipeline_model_parallel_rank() if vp_size is not None else 0

    local = assignments[(pp_rank, vp_rank)]
    return local, assignments

def pp_gather(local_layers_router_map, tf_config):
        # TODO: Consider non-uniform layer allocation cases.
        """
        Gather local router maps from all PP ranks into a global router map.

        Args:
            local_layers_router_map (torch.Tensor): Local router map of shape
                [bs, max_seq_len, local_num_layers, topk].
            tf_config: Configuration providing pipeline_model_parallel_size.

        Returns:
            torch.Tensor: Global router map of shape [bs, max_seq_len, num_layers, topk] placed on CPU.
        """
        pp_size = tf_config.pipeline_model_parallel_size
        if pp_size <= 1:
            return local_layers_router_map

        pp_group = mpu.get_pipeline_model_parallel_group()
        world_size = torch.distributed.get_world_size(pp_group)
        local_layers_router_map = local_layers_router_map.cuda()
        layers_topk_idx_global_list = [torch.empty(size=local_layers_router_map.shape, \
                                                    dtype=local_layers_router_map.dtype,   \
                                                    device=local_layers_router_map.device ) \
                                                    for _ in range(world_size) ]
        torch.distributed.all_gather(
            tensor=local_layers_router_map,
            tensor_list=layers_topk_idx_global_list,
            group=pp_group,
            async_op=False,
        )
        global_router_map = torch.cat(layers_topk_idx_global_list, dim=2).to("cpu")
        return global_router_map

def pp_dispatch(global_layers_router_map, tf_config):
    """
    Dispatch a global router map to the current PP rank, returning only the local layer slice.

    If virtual pipeline is disabled, this simply slices the global map according to the local PP range.
    If virtual pipeline is enabled, the function aggregates the VP stage ranges for the current PP rank
    by multiplying the per-VP count by vp_size, then slices accordingly.

    Args:
        global_layers_router_map (torch.Tensor): Global router map of shape [bs, max_seq_len, num_layers, topk].
        tf_config: Configuration object that includes pipeline_model_parallel_size and optionally
            virtual_pipeline_model_parallel_size.

    Returns:
        torch.Tensor: Local router map for the current PP rank of shape
        [bs, max_seq_len, pp_rank_layers, topk].
    """
    pp_size = tf_config.pipeline_model_parallel_size
    vp_size = tf_config.virtual_pipeline_model_parallel_size
    if pp_size <= 1:
        return global_layers_router_map
    
    # Enable VPP
    local_info, all_assignments = get_current_rank_layer_info(tf_config)
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    if  vp_size is None or vp_size<=1:
        sta = local_info["start"]
        end = local_info["end"]
    else:
        local_info = all_assignments[(pp_rank, 0)]
        count = local_info["count"]
        sta = local_info["start"]
        pp_layer_count = count * vp_size
        end = sta + pp_layer_count
    local_router_map =  global_layers_router_map[:,:,sta:end,:]

    return local_router_map



class RouterReplayHelper:
    """Helper class to query router replay state and locate local RouterReplay instances."""

    @staticmethod
    def get_micro_batch_router_list(tf_config, vp_rank=None):
        """
        Return the list of RouterReplay instances corresponding to the current micro-batch and local
        (pp_rank, vp_stage) layer range.

        When virtual pipeline (VPP) is enabled, the local range for the PP rank is expanded to include
        all VP stages by multiplying the per-VP count by vp_size. The returned slice is taken from the
        global RouterReplay.router_instances list.

        Args:
            tf_config: Configuration object used to compute layer assignments.
            vp_rank (Optional[int]): Explicit virtual pipeline stage to query. If None, the current VP
                rank from Megatron parallel state is used when available.

        Returns:
            list: A contiguous sublist of RouterReplay.router_instances for the local layer range.
        """
        local, all_assignments = get_current_rank_layer_info(tf_config, vp_rank)
        vp_size = tf_config.virtual_pipeline_model_parallel_size
        pp_size = tf_config.pipeline_model_parallel_size
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        pp_local = deepcopy(local)
        # Enable VPP
        if not (pp_size <= 1 or vp_size is None or vp_size<=1):
            local_t = all_assignments[(pp_rank, 0)]
            count = local_t["count"]
            sta = local_t["start"]
            pp_layer_count = count * vp_size

            pp_local["start"] = sta
            pp_local["end"] = sta + pp_layer_count
            pp_local["count"] = pp_layer_count
        sta, end = local["start"] - pp_local["start"], local["end"] - pp_local["start"]
        router_instances_list = RouterReplay.router_instances[sta:end]
        return router_instances_list

    @staticmethod
    def is_r2_record_mode(tf_config, vp_rank=None) -> bool:
        """Return True if the current mode is RECORD (R2) for the local router instances.

        This inspects the first local RouterReplay instance's routing_mode and compares it to
        RoutingMode.RECORD.
        """
        router_instances_list = RouterReplayHelper.get_micro_batch_router_list(tf_config, vp_rank)
        return (
            router_instances_list
            and router_instances_list[0].routing_mode == RoutingMode.RECORD
        )

    @staticmethod
    def is_replay_forward_mode(tf_config, vp_rank=None) -> bool:
        """Return True if the current mode is REPLAY_FORWARD for the local router instances.

        This inspects the first local RouterReplay instance's routing_mode and compares it to
        RoutingMode.REPLAY_FORWARD.
        """
        router_instances_list = RouterReplayHelper.get_micro_batch_router_list(tf_config, vp_rank)
        return (
            router_instances_list
            and router_instances_list[0].routing_mode == RoutingMode.REPLAY_FORWARD
        )

    @staticmethod
    def is_replay_backward_mode(tf_config, vp_rank=None) -> bool:
        """Return True if the current mode is REPLAY_BACKWARD for the local router instances.

        This inspects the first local RouterReplay instance's routing_mode and compares it to
        RoutingMode.REPLAY_BACKWARD.
        """
        router_instances_list = RouterReplayHelper.get_micro_batch_router_list(tf_config, vp_rank)
        return (
            router_instances_list
            and router_instances_list[0].routing_mode == RoutingMode.REPLAY_BACKWARD
        )

    @staticmethod
    def is_r2_or_r3_mode(router_replay) -> bool:
        """Return True if the router replay mode string is either "R2" or "R3"."""
        return router_replay.mode in ["R2", "R3"]
