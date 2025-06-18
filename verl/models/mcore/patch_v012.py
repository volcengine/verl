# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

# there is some bug in mcore 0.12, so we need to patch it
# 1. `get_query_key_value_tensors` in `multi_latent_attention.py` works wrong when packed_seq_params is not None


def apply_patch():
    import torch
    from megatron.core import parallel_state, tensor_parallel
    from megatron.core.transformer.multi_latent_attention import MLASelfAttention, apply_rotary_pos_emb, deprecate_inference_params, gather_from_sequence_parallel_region, gather_from_tensor_model_parallel_region, scatter_to_sequence_parallel_region

    def patch_get_query_key_value_tensors(
        self,
        hidden_states,
        key_value_states=None,
        position_ids=None,
        packed_seq_params=None,
        inference_context=None,
        *,
        inference_params=None,
    ):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        # s = sequence length, b = batch size, h = hidden size, n = num attention heads
        # Attention heads [s, b, n*h]
        assert hidden_states.ndim == 3, f"hidden_states should be 3D, [s, b, n*h], got {hidden_states.ndim}D"

        inference_context = deprecate_inference_params(inference_context, inference_params)

        # =========================================
        # Prepare RoPE and seqlen related params
        # =========================================
        rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(inference_context, None, hidden_states, self.config, packed_seq_params)

        # rotary_pos_emb:[s, b, 1, 64]
        mscale = 1.0
        if self.config.rope_type == "rope":
            packed_seq = packed_seq_params is not None and packed_seq_params.qkv_format == "thd"
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len, packed_seq=packed_seq)
        else:
            rotary_pos_emb, mscale = self.rotary_pos_emb(rotary_seq_len)

        # =========================================
        # QKV down projection and layernorm
        # =========================================
        if self.config.q_lora_rank is not None:
            # if linear_q_down_proj is ColumnParallelLinear:
            #     q_compressed: [s, b, q_lora_rank / TP]
            # elif linear_q_down_proj is Linear:
            #     q_compressed: [s / TP, b, q_lora_rank]
            q_compressed, _ = self.linear_q_down_proj(hidden_states)

            # When output is sharded (ColumnParallelLinear), two things are needed to be
            # identical to a normal Linear.
            #   1. Manually gather output to restore output dim q_lora_rank;
            #   2. Scatter sequence back to s / TP if sequence-parallel since it was
            #      gathered by ColumnParallelLinear.
            if q_compressed.size(-1) != self.config.q_lora_rank:
                q_compressed = gather_from_tensor_model_parallel_region(q_compressed)
                if self.config.sequence_parallel:
                    q_compressed = scatter_to_sequence_parallel_region(q_compressed)

            q_compressed = self.q_layernorm(q_compressed)
        else:
            q_compressed = hidden_states

        # if linear_kv_down_proj is ColumnParallelLinear:
        #     kv_combined: [s, b, (kv_lora_rank + qk_pos_emb_head_dim) / TP]
        # elif linear_kv_down_proj is Linear:
        #     kv_combined: [s / TP, b, (kv_lora_rank + qk_pos_emb_head_dim)]
        kv_combined, _ = self.linear_kv_down_proj(hidden_states)
        if kv_combined.size(-1) != self.config.kv_lora_rank + self.config.qk_pos_emb_head_dim:
            # kv_combined: [s, b, (kv_lora_rank + qk_pos_emb_head_dim)]
            kv_combined = gather_from_tensor_model_parallel_region(kv_combined)
            # kv_compressed:[s, b, kv_lora_rank], k_pos_emb: [s, b, qk_pos_emb_head_dim]
            kv_compressed, k_pos_emb = torch.split(kv_combined, [self.config.kv_lora_rank, self.config.qk_pos_emb_head_dim], dim=-1)
            if self.config.sequence_parallel:
                # kv_compressed:[s / TP, b, kv_lora_rank]
                kv_compressed = scatter_to_sequence_parallel_region(kv_compressed)
        else:
            # kv_compressed:[s / TP, b, kv_lora_rank], k_pos_emb: [s / TP, b, qk_pos_emb_head_dim]
            kv_compressed, k_pos_emb = torch.split(kv_combined, [self.config.kv_lora_rank, self.config.qk_pos_emb_head_dim], dim=-1)
            if parallel_state.get_tensor_model_parallel_world_size() > 1:
                # k_pos_emb: [s, b, qk_pos_emb_head_dim]
                k_pos_emb = gather_from_sequence_parallel_region(k_pos_emb)

        kv_compressed = self.kv_layernorm(kv_compressed)

        # =========================================
        # QKV up projection and RoPE apply
        # =========================================
        def qkv_up_proj_and_rope_apply(q_compressed, kv_compressed, k_pos_emb, rotary_pos_emb):
            if self.config.q_lora_rank is not None:
                q, _ = self.linear_q_up_proj(q_compressed)
            else:
                # hidden_states:[s, b, 2048], q: [s, b, n * 192]
                q, _ = self.linear_q_proj(q_compressed)

            q_len, bsz, _ = q.size()

            # q: [s, b, n, 192]
            q = q.view(q_len, bsz, self.num_attention_heads_per_partition, self.q_head_dim)

            # kv: [s, b, 2048]
            kv, _ = self.linear_kv_up_proj(kv_compressed)

            # kv: [s, b, n, 256]
            kv = kv.view(
                q_len,
                bsz,
                self.num_attention_heads_per_partition,
                self.config.qk_head_dim + self.config.v_head_dim,
            )

            if inference_context is not None:
                # add offset to the sequence start for inference
                sequence_start = inference_context.sequence_len_offset
                sequence_end = sequence_start + q_len
                rotary_pos_emb = rotary_pos_emb[sequence_start:sequence_end]
            else:
                # Shorten rotary_pos_emb to the sequence length when inference_params
                # is not provided. This makes sure we can run forward directly with
                # any sequence length. During training, the sequence length is always
                # the full rotary_pos_emb length.
                rotary_pos_emb = rotary_pos_emb[0:q_len]

            # [s, b, 64] -> [s, b, 1, 64]
            k_pos_emb = torch.unsqueeze(k_pos_emb, 2)

            # q: [s, b, n, 128], q_pos_emb: [s, b, n, 64]
            q_no_pe, q_pos_emb = torch.split(q, [self.config.qk_head_dim, self.config.qk_pos_emb_head_dim], dim=-1)

            # k_no_pe: [s, b, n, 128], value: [s, b, n, 128]
            k_no_pe, value = torch.split(kv, [self.config.qk_head_dim, self.config.v_head_dim], dim=-1)

            if packed_seq_params is not None:
                cu_seqlens_q = packed_seq_params.cu_seqlens_q
                cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
                q_pos_emb = q_pos_emb.squeeze(1)
                k_pos_emb = k_pos_emb.squeeze(1)
                q_no_pe = q_no_pe.squeeze(1)
                k_no_pe = k_no_pe.squeeze(1)
                value = value.squeeze(1)
            else:
                cu_seqlens_q = cu_seqlens_kv = None

            # q_pos_emb: [s, b, n, 64], k_pos_emb:[s, b, 1, 64]
            q_pos_emb = apply_rotary_pos_emb(
                q_pos_emb,
                rotary_pos_emb,
                config=self.config,
                cu_seqlens=cu_seqlens_q,
                mscale=mscale,
            )
            k_pos_emb = apply_rotary_pos_emb(
                k_pos_emb,
                rotary_pos_emb,
                config=self.config,
                cu_seqlens=cu_seqlens_kv,
                mscale=mscale,
            )

            # query: [s, b, n, 192]
            query = torch.cat([q_no_pe, q_pos_emb], dim=-1)
            if packed_seq_params is not None:
                k_pos_emb = k_pos_emb.expand(-1, self.num_attention_heads_per_partition, -1)
                key = torch.cat([k_no_pe, k_pos_emb], dim=-1)
            else:
                # key: [s, b, n, 192]
                k_pos_emb = k_pos_emb.expand(-1, -1, self.num_attention_heads_per_partition, -1)
                key = torch.cat([k_no_pe, k_pos_emb], dim=-1)

            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()
            return query, key, value

        if self.recompute_up_proj:
            self.qkv_up_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            query, key, value = self.qkv_up_checkpoint.checkpoint(qkv_up_proj_and_rope_apply, q_compressed, kv_compressed, k_pos_emb, rotary_pos_emb)
        else:
            query, key, value = qkv_up_proj_and_rope_apply(q_compressed, kv_compressed, k_pos_emb, rotary_pos_emb)

        return query, key, value

    MLASelfAttention.get_query_key_value_tensors = patch_get_query_key_value_tensors


def apply_optimizer_sharded_save_load_patches():
    """
    Apply patch to ChainedOptimizer and DistributedOptimizer in mcore 0.12.0 to enable
    saving sharded optimizer state from GPU to disk directly without DP gathering to CPU memory.
    """
    from pathlib import Path

    import torch
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
    from megatron.core.optimizer.optimizer import ChainedOptimizer

    def get_parameter_state_local(self):
        """
        Get parameter state (i.e., parameter & optimizer tensors).

        Each rank returns its own sharded state directly from GPU.
        The state is already sharded in the buffer.
        """
        if self.ddp_config.use_custom_fsdp:
            state = {"buckets_coalesced": True}
            for model_chunk in self.model_chunks:
                pg_buffer = model_chunk.param_and_grad_buffer
                for group_id, group in enumerate(pg_buffer.parameter_groups):
                    this_group_state = {}
                    mbuf = group.master_weight_buffer
                    for item_id, _ in enumerate(group.params):
                        main_param = mbuf.get_item(item_id)
                        optim_state = self.optimizer.state[main_param]
                        # Keep tensors on GPU
                        for name, value in optim_state.items():
                            assert torch.is_tensor(value), f"Expected tensor, got {type(value)}"
                            this_group_state.setdefault(name, []).append(value)

                    state[f"group_{group_id}"] = this_group_state
            return state

        state = {"buckets_coalesced": True}
        for gbuf_idx, gbuf_range_maps in enumerate(self.gbuf_ranges):
            dtype_state = {}
            assert len(gbuf_range_maps) == 1, "single dtype supported, for now."
            for dtype, gbuf_range_map_for_all_buckets in gbuf_range_maps.items():
                local_tensors = {}
                for key in ("param", "exp_avg", "exp_avg_sq"):
                    local_tensors[key] = []

                for gbuf_range_map in gbuf_range_map_for_all_buckets:
                    # Build contiguous DP rank shards (for param + optim states)
                    for model_param, param_range_map in gbuf_range_map["param_map"].items():
                        tensors = self._get_main_param_and_optimizer_states(model_param)

                        # Store references to existing tensors
                        gbuf_local_start = param_range_map["gbuf_local"].start
                        gbuf_local_end = param_range_map["gbuf_local"].end
                        for key in local_tensors:
                            local_tensors[key].append({"tensor": tensors[key], "start": gbuf_local_start, "end": gbuf_local_end})

                dtype_state[dtype] = local_tensors
            state[gbuf_idx] = dtype_state
        return state

    def load_parameter_state_local(self, state_dict):
        """
        Load local parameter state.
        """
        if self.ddp_config.use_custom_fsdp:
            for model_chunk in self.model_chunks:
                pg_buffer = model_chunk.param_and_grad_buffer
                for group_id, group in enumerate(pg_buffer.parameter_groups):
                    if f"group_{group_id}" not in state_dict:
                        continue
                    this_group_state = state_dict[f"group_{group_id}"]
                    mbuf = group.master_weight_buffer
                    for item_id, _ in enumerate(group.params):
                        main_param = mbuf.get_item(item_id)
                        optim_state = self.optimizer.state[main_param]
                        for name, value in this_group_state.items():
                            if name in optim_state:
                                optim_state[name].copy_(value)
            return

        for gbuf_idx, gbuf_range_maps in enumerate(self.gbuf_ranges):
            assert len(gbuf_range_maps) == 1, "single dtype supported, for now."
            for dtype, gbuf_range_map_for_all_buckets in gbuf_range_maps.items():
                if gbuf_idx not in state_dict or dtype not in state_dict[gbuf_idx]:
                    continue

                local_tensors = state_dict[gbuf_idx][dtype]

                for gbuf_range_map in gbuf_range_map_for_all_buckets:
                    for model_param, param_range_map in gbuf_range_map["param_map"].items():
                        gbuf_local_start = param_range_map["gbuf_local"].start
                        gbuf_local_end = param_range_map["gbuf_local"].end

                        tensors = self._get_main_param_and_optimizer_states(model_param)

                        for key in tensors:
                            # Find the matching tensor reference in the loaded state
                            for tensor_ref in local_tensors[key]:
                                if tensor_ref["start"] == gbuf_local_start and tensor_ref["end"] == gbuf_local_end:
                                    tensors[key].copy_(tensor_ref["tensor"])
                                    # Exit early to avoid unnecessary iteration
                                    break

    def save_sharded_parameter_state(self, filename: str):
        """Save the distributed parameter state directly from GPU to disk in a sharded format.

        Args:
            filename (str): path to save parameter state to.
        """
        if self.is_stub_optimizer:
            return

        state_dict = self.get_parameter_state_local()
        torch.save(state_dict, filename)

    def load_sharded_parameter_state(self, filename: str):
        """Load the distributed parameter state from sharded checkpoints.

        Args:
            filename (str): path to load parameter state from.
        """
        if self.is_stub_optimizer:
            return

        try:
            state_dict = torch.load(filename)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find checkpoint file {filename}.")

        self.load_parameter_state_local(state_dict)

    def save_chained_sharded_parameter_state(self, filename: str):
        """
        Save the distributed parameter states of all optimizers in a sharded format.

        Each optimizer's state is saved directly to disk without gathering to DP 0.
        The checkpoint files are saved in separate folders for each optimizer.

        Args:
            filename (str): path to save parameter state to.
        """
        if len(self.chained_optimizers) == 1:
            self.chained_optimizers[0].save_sharded_parameter_state(filename)
            return

        for idx, optimizer in enumerate(self.chained_optimizers):
            if hasattr(optimizer, "save_sharded_parameter_state"):
                optimizer_dir = Path(filename).parent / f"optimizer_{idx}"
                if torch.distributed.get_rank() == 0:
                    optimizer_dir.mkdir(exist_ok=True)

                # Ensure directory exists before saving
                torch.distributed.barrier()

                optimizer_filename = optimizer_dir / Path(filename).name
                optimizer.save_sharded_parameter_state(str(optimizer_filename))

    def load_chained_sharded_parameter_state(self, filename: str):
        """Load the distributed parameter states of all optimizers from sharded checkpoints.

        Each optimizer's state is loaded directly without gathering to DP 0.
        The checkpoint files are expected to be in optimizer-specific folders.

        Args:
            filename (str): Base filename for the checkpoint. The actual filename will include
                          the rank information and be loaded from optimizer-specific folders.
        """
        if len(self.chained_optimizers) == 1:
            self.chained_optimizers[0].load_sharded_parameter_state(filename)
            return

        for idx, optimizer in enumerate(self.chained_optimizers):
            if not hasattr(optimizer, "load_sharded_parameter_state"):
                continue
            optimizer_dir = Path(filename).parent / f"optimizer_{idx}"
            optimizer_filename = optimizer_dir / Path(filename).name
            optimizer.load_sharded_parameter_state(str(optimizer_filename))

    DistributedOptimizer.get_parameter_state_local = get_parameter_state_local
    DistributedOptimizer.load_parameter_state_local = load_parameter_state_local
    DistributedOptimizer.save_sharded_parameter_state = save_sharded_parameter_state
    DistributedOptimizer.load_sharded_parameter_state = load_sharded_parameter_state
    ChainedOptimizer.save_sharded_parameter_state = save_chained_sharded_parameter_state
    ChainedOptimizer.load_sharded_parameter_state = load_chained_sharded_parameter_state
