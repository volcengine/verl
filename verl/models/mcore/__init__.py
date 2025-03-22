from verl.utils.megatron import sequence_parallel as sp_utils
from verl.utils.megatron import tensor_parallel as tp_utils
import torch


def gptmodel_forward(model, input_ids, attention_mask, position_ids, sequence_parallel, pack_seqs=False):
    if pack_seqs:
        from flash_attn.bert_padding import pad_input, unpad_input  # noqa
        from megatron.core.packed_seq_params import PackedSeqParams
        import copy
        batch_size, sequence_length = input_ids.shape
        input_ids_rmpad, indices, cu_seqlens, max_seqlen_in_batch, *_ = unpad_input(input_ids.unsqueeze(dim=-1),
                                                                                    attention_mask)
        cu_seqlens_padded = None
        if sequence_parallel:
            original_total_nnz = input_ids_rmpad.shape[0]
            input_ids_rmpad = sp_utils.pad_to_sequence_parallel(input_ids_rmpad)
            total_nnz_new = input_ids_rmpad.shape[0]
            pad_size = total_nnz_new - original_total_nnz
            if pad_size > 0:
                cu_seqlens_padded = copy.deepcopy(cu_seqlens)
                cu_seqlens_padded[-1] = cu_seqlens_padded[-1] + pad_size
                seqlens = cu_seqlens_padded[1:] - cu_seqlens_padded[:-1]
                max_seqlen_in_batch = seqlens.max().cpu().item()

        packed_seq_params = PackedSeqParams(qkv_format='thd',
                                            cu_seqlens_q=cu_seqlens,
                                            max_seqlen_q=max_seqlen_in_batch,
                                            cu_seqlens_kv=cu_seqlens,
                                            max_seqlen_kv=max_seqlen_in_batch,
                                            cu_seqlens_q_padded=cu_seqlens_padded,
                                            cu_seqlens_kv_padded=cu_seqlens_padded)
        output = model(input_ids=input_ids_rmpad.T,
                       attention_mask=None,
                       position_ids=position_ids,
                       packed_seq_params=packed_seq_params)
        output = torch.squeeze(output, dim=0)  # (1,seq,1)->(seq,1)

        if sequence_parallel and pad_size > 0:
            output = output[:cu_seqlens[-1]]

        output = pad_input(output, indices, batch_size, seqlen=sequence_length)
    else:
        # output = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
        batch_size, sequence_length = input_ids.shape
        new_input_ids, new_attention_mask, new_position_ids = remove_left_padding(input_ids, attention_mask,
                                                                                  position_ids, sequence_parallel)
        output = model(input_ids=new_input_ids, attention_mask=new_attention_mask, position_ids=new_position_ids)
        output = recover_left_padding(output, new_attention_mask, attention_mask, sequence_length)

    return output


def remove_left_padding(input_ids: torch.Tensor,
                        attention_mask: torch.Tensor,
                        position_ids: torch.Tensor,
                        sequence_parallel: bool = False):
    """
    Remove left padding from input_ids, attention_mask and position_ids
    return new_input_ids, new_attention_mask, new_position_ids
    """
    assert attention_mask.ndim == 2
    assert position_ids.ndim == 2

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

    new_input_ids = torch.zeros(dtype=input_ids.dtype, device=input_ids.device, size=shape)
    new_attention_mask = torch.zeros(dtype=attention_mask.dtype,
                                     device=attention_mask.device,
                                     size=(batch_size, seq_len))
    new_position_ids = torch.zeros(dtype=position_ids.dtype, device=position_ids.device, size=(batch_size, seq_len))
    for i in range(batch_size):
        new_input_ids[i, :seq_lens[i]] = input_ids[i, attention_mask[i]]
        new_attention_mask[i, :seq_lens[i]] = attention_mask[i, attention_mask[i]]
        new_position_ids[i, :seq_lens[i]] = position_ids[i, attention_mask[i]]
    return new_input_ids, new_attention_mask, new_position_ids


def recover_left_padding(result, attention_mask: torch.Tensor, original_attention_mask: torch.Tensor,
                         origin_seqlen: int):
    """
    Recover left padding from result
    return result
    """
    shape = list(result.shape)
    batch_size = shape[0]
    shape[1] = origin_seqlen
    new_result = torch.zeros(dtype=result.dtype, device=result.device, size=shape)
    for i in range(batch_size):
        new_result[i, original_attention_mask[i]] = result[i, attention_mask[i]]
    return new_result
