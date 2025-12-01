from __future__ import annotations
import torch
from typing import List
from prefix_grouper import PrefixGrouper
from verl.utils.torch_functional import logprobs_from_logits


# -------------- 工具：为 PrefixGrouper 生成 position_ids ------------------

def build_position_ids_for_prefix_grouper(prefix_grouper: PrefixGrouper) -> torch.Tensor:
    """
    为 PrefixGrouper 构造正确的 position_ids。
    
    根据官方文档，每个 response 的 position_ids 应该从 prefix_len 重新开始，而不是连续递增。
    例如：[0,1,2,3, 4,5, 4,5] 而不是 [0,1,2,3, 4,5, 6,7]
    
    Args:
        prefix_grouper: PrefixGrouper 实例
    
    Returns:
        position_ids: [num_samples, seq_len]
    """
    num_samples = len(prefix_grouper.group_info)
    max_len = prefix_grouper.padding_mask.size(1)
    device = prefix_grouper.padding_mask.device
    
    position_ids = torch.zeros(num_samples, max_len, dtype=torch.long, device=device)
    
    for i, group in enumerate(prefix_grouper.group_info):
        prefix_len = group.prefix_len
        
        # 前缀部分：0, 1, 2, ..., prefix_len-1
        position_ids[i, :prefix_len] = torch.arange(prefix_len, device=device)
        
        # 后缀部分：每个 response 都从 prefix_len 开始
        cur_pos = prefix_len
        for suffix_len in group.suffix_lens:
            if suffix_len > 0:
                position_ids[i, cur_pos:cur_pos + suffix_len] = torch.arange(
                    prefix_len, prefix_len + suffix_len, device=device
                )
                cur_pos += suffix_len
    
    return position_ids


# -------------- 工具：对 micro_batch 按 uid 排序 ------------------

def sort_batch_by_uid(data):
    uids = data.non_tensor_batch.get("uid")
    if uids is None:
        return data

    if torch.is_tensor(uids):
        order = torch.argsort(uids, stable=True)
    else:
        import numpy as np
        order = torch.as_tensor(np.argsort(uids, kind="stable"))

    if torch.equal(order, torch.arange(order.numel())):
        return data

    data.reorder(order)
    return data


def build_pg_from_micro_batch(
    micro_batch,
    pad_token_id: int,
    padding_mode: str = "right",
):
    batch = getattr(micro_batch, "batch", micro_batch)
    non_tensor_batch = getattr(micro_batch, "non_tensor_batch", {})

    prompts = batch["prompts"]
    responses = batch["responses"]
    response_mask = batch["response_mask"]
    uids = non_tensor_batch["uid"]

    bs = responses.size(0)

    # -------- group_sizes --------
    group_sizes = []
    cur = 1
    for i in range(1, bs):
        if uids[i] == uids[i-1]:
            cur += 1
        else:
            group_sizes.append(cur)
            cur = 1
    group_sizes.append(cur)

    # 每组第一条 sample 作为 prefix
    prefix_indices = []
    cursor = 0
    for gs in group_sizes:
        prefix_indices.append(cursor)
        cursor += gs
    prefix_indices = torch.tensor(prefix_indices, device=prompts.device)

    prefix_ids = prompts.index_select(0, prefix_indices)
    prefix_mask = prefix_ids.ne(pad_token_id)

    # -------- PG 初始化：suffix_mask --------
    prefix_grouper = PrefixGrouper.from_ungrouped_masks(
        prefix_mask=prefix_mask,
        suffix_mask=response_mask,
        group_sizes=group_sizes,
        padding_mode=padding_mode,
        device=prompts.device,
    )

    # -------- concat_input：completion_ids--------
    concat_input_ids = prefix_grouper.concat_input(
        prefix_ids, prefix_mask, responses, response_mask
    )

    attention_mask = prefix_grouper.padding_mask
    
    position_ids = build_position_ids_for_prefix_grouper(prefix_grouper)

    return (
        prefix_grouper,
        concat_input_ids,
        attention_mask,
        position_ids,
        responses,
        response_mask,
    )


def pg_forward(
    model,
    prefix_grouper,
    concat_input_ids,
    attention_mask,
    position_ids,
    completion_ids,
    completion_mask,
    *,
    temperature=1.0,
    padding_mode="right",
    include_prefix_last=1,
    calculate_entropy=False,
    entropy_fn=None,
):
    # ------- 模型 forward -------
    logits = model(
        input_ids=concat_input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
        prefix_grouper=prefix_grouper,
    ).logits

    # ------- split_output(include_prefix_last=1) -------
    prefix_out, prefix_mask, suffix_out_raw, suffix_mask_raw = (
        prefix_grouper.split_output(
            logits, include_prefix_last=include_prefix_last)
    )

    # ------- convert_padding -------
    completion_ids_right = prefix_grouper.convert_padding(
        completion_ids,
        completion_mask,
        padding_mode=padding_mode,
    )

    # ------- alignment -------
    suffix_out = suffix_out_raw[:, :-1].float()
    suffix_mask = suffix_mask_raw[:, 1:]

    suffix_out /= temperature

    log_probs = logprobs_from_logits(suffix_out, completion_ids_right)

    entropy = None
    if calculate_entropy and entropy_fn is not None:
        entropy = entropy_fn(suffix_out)

    return log_probs, entropy, suffix_mask
