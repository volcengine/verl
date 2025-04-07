# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

from verl.utils.megatron import sequence_parallel as sp_utils
from verl.utils.megatron import tensor_parallel as tp_utils
import torch
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core import parallel_state as mpu
from verl.utils.megatron_utils import unwrap_model


def gptmodel_forward(model,
                     input_ids,
                     attention_mask,
                     position_ids,
                     sequence_parallel,
                     value_model=False,
                     pack_seqs=True):
    pre_process = unwrap_model(model).pre_process
    post_process = unwrap_model(model).post_process
    if pack_seqs:
        batch_size, seq_len = attention_mask.shape[:2]
        input_ids_rmpad, packed_seq_params = preprocess_packed_seqs(input_ids, attention_mask, pre_process=pre_process)
        input_ids_rmpad = input_ids_rmpad.contiguous()
        output_orig = model(input_ids=input_ids_rmpad,
                            attention_mask=None,
                            position_ids=position_ids,
                            packed_seq_params=packed_seq_params)

        output = postprocess_packed_seqs(output_orig,
                                         packed_seq_params,
                                         attention_mask,
                                         batch_size,
                                         seq_len,
                                         post_process=post_process)
    else:
        batch_size, sequence_length = attention_mask.shape
        new_input_ids, new_attention_mask, new_position_ids = remove_left_padding(input_ids,
                                                                                  attention_mask,
                                                                                  position_ids,
                                                                                  sequence_parallel,
                                                                                  pre_process=pre_process)
        output = model(input_ids=new_input_ids, attention_mask=new_attention_mask, position_ids=new_position_ids)
        output = recover_left_padding(output,
                                      new_attention_mask,
                                      attention_mask,
                                      sequence_length,
                                      post_process=post_process)
    if value_model and post_process:
        output = output[..., 0]
    return output


def preprocess_packed_seqs(input_ids: torch.Tensor,
                           attention_mask: torch.Tensor,
                           pre_process: bool = True) -> tuple[torch.Tensor, PackedSeqParams]:
    """
    Preprocess packed sequences
    """
    batch_size = input_ids.shape[0]

    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    tp_size = mpu.get_tensor_model_parallel_world_size()
    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()
    if cp_size > 1:
        align_size = tp_size * cp_size * 2
    else:
        align_size = tp_size

    pad_size = (align_size - seqlens_in_batch % align_size) % align_size
    seqlens_in_batch_padded = seqlens_in_batch + pad_size
    cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=input_ids.device)
    cu_seqlens[1:] = torch.cumsum(seqlens_in_batch, dim=0)
    cu_seqlens_padded = torch.zeros(batch_size + 1, dtype=torch.int32, device=input_ids.device)
    cu_seqlens_padded[1:] = torch.cumsum(seqlens_in_batch_padded, dim=0)
    max_seqlen_in_batch = seqlens_in_batch_padded.max().item()

    shape = list(input_ids.shape[1:])
    shape[0] = seqlens_in_batch_padded.sum().item() // cp_size
    if pre_process:
        input_ids_rmpad = torch.zeros(shape, dtype=input_ids.dtype, device=input_ids.device)
        for i in range(batch_size):
            seqlen = seqlens_in_batch[i] // cp_size
            input_ids_rmpad[cu_seqlens_padded[i] // cp_size:cu_seqlens_padded[i] // cp_size +
                            seqlen] = input_ids[i, attention_mask[i]][seqlen * cp_rank:seqlen * (cp_rank + 1)]

    packed_seq_params = PackedSeqParams(qkv_format='thd',
                                        cu_seqlens_q=cu_seqlens_padded,
                                        max_seqlen_q=max_seqlen_in_batch,
                                        cu_seqlens_kv=cu_seqlens_padded,
                                        max_seqlen_kv=max_seqlen_in_batch,
                                        cu_seqlens_q_padded=cu_seqlens_padded,
                                        cu_seqlens_kv_padded=cu_seqlens_padded)
    if pre_process:
        return input_ids_rmpad.unsqueeze(0), packed_seq_params
    else:
        return input_ids, packed_seq_params


def postprocess_packed_seqs(output: torch.Tensor,
                            packed_seq_params: PackedSeqParams,
                            attention_mask: torch.Tensor,
                            batch_size: int,
                            seq_len: int,
                            post_process: bool = True) -> torch.Tensor:
    """
    Postprocess packed sequences
    """
    if not post_process:
        return output
    shape = [batch_size, seq_len] + list(output.shape[2:])  # 1,packed, dim -> batch_size, seq_len, dim
    output_new = torch.zeros(shape, dtype=output.dtype, device=output.device)

    cp_size = mpu.get_context_parallel_world_size()
    # all gather output across context parallel group
    if cp_size > 1:
        # output shape: [1, packed_len, hidden_dim]
        # need to gather across cp group and concatenate in sequence dimension
        output_list = [torch.empty_like(output) for _ in range(cp_size)]
        torch.distributed.all_gather(output_list, output.detach(), group=mpu.get_context_parallel_group())
        output_list[mpu.get_context_parallel_rank()] = output
    else:
        output_list = [output]
    for i in range(batch_size):
        s_len_padded_chunk = (packed_seq_params.cu_seqlens_q_padded[i + 1] -
                              packed_seq_params.cu_seqlens_q_padded[i]) // cp_size
        s_len = attention_mask[i].sum().item()
        tmp = []
        for j in range(cp_size):
            o = output_list[j][0]
            seq_chunk_len_if_last = min(s_len, s_len_padded_chunk * (j + 1)) - s_len_padded_chunk * j
            if seq_chunk_len_if_last <= 0:
                break
            packed_start_idx = packed_seq_params.cu_seqlens_q_padded[i] // cp_size
            packed_end_idx = packed_start_idx + seq_chunk_len_if_last
            tmp.append(o[packed_start_idx:packed_end_idx])
        output_new[i, attention_mask[i]] = torch.cat(tmp, dim=0)

    return output_new


def remove_left_padding(input_ids: torch.Tensor,
                        attention_mask: torch.Tensor,
                        position_ids: torch.Tensor,
                        sequence_parallel: bool = False,
                        pre_process: bool = True):
    """
    Remove left padding from input_ids, attention_mask and position_ids
    return new_input_ids, new_attention_mask, new_position_ids
    """
    assert attention_mask.ndim == 2
    assert position_ids.ndim == 2
    cp_size = mpu.get_context_parallel_world_size()
    assert cp_size == 1, 'Context parallel size without seq_pack is not supported'
    batch_size = input_ids.shape[0]
    shape = list(input_ids.shape)  # batch_size, seq_len,...
    seq_lens = attention_mask.sum(dim=1)
    seq_len = seq_lens.max().item()
    if sequence_parallel:
        from megatron.core import parallel_state as mpu
        sp_world_size = mpu.get_tensor_model_parallel_world_size()
        pad_size = (sp_world_size - seq_len % sp_world_size) % sp_world_size
        seq_len = seq_len + pad_size
    shape[1] = seq_len
    if pre_process:
        new_input_ids = torch.zeros(dtype=input_ids.dtype, device=input_ids.device, size=shape)
    new_attention_mask = torch.zeros(dtype=attention_mask.dtype,
                                     device=attention_mask.device,
                                     size=(batch_size, seq_len))
    new_position_ids = torch.zeros(dtype=position_ids.dtype, device=position_ids.device, size=(batch_size, seq_len))
    for i in range(batch_size):
        if pre_process:
            new_input_ids[i, :seq_lens[i]] = input_ids[i, attention_mask[i]]
        new_attention_mask[i, :seq_lens[i]] = attention_mask[i, attention_mask[i]]
        new_position_ids[i, :seq_lens[i]] = position_ids[i, attention_mask[i]]
    if pre_process:
        return new_input_ids, new_attention_mask, new_position_ids
    else:
        return input_ids, new_attention_mask, new_position_ids


def recover_left_padding(result,
                         attention_mask: torch.Tensor,
                         original_attention_mask: torch.Tensor,
                         origin_seqlen: int,
                         post_process: bool = True):
    """
    Recover left padding from result
    return result
    """
    if not post_process:
        return result
    shape = list(result.shape)
    batch_size = shape[0]
    shape[1] = origin_seqlen
    new_result = torch.zeros(dtype=result.dtype, device=result.device, size=shape)
    for i in range(batch_size):
        new_result[i, original_attention_mask[i]] = result[i, attention_mask[i]]
    return new_result
