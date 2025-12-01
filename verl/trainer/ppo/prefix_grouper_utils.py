from __future__ import annotations
import torch
from typing import List
from prefix_grouper import PrefixGrouper
from verl.utils.torch_functional import logprobs_from_logits


def build_position_ids_for_prefix_grouper(prefix_grouper: PrefixGrouper) -> torch.Tensor:
    """Build position_ids for PrefixGrouper where each response restarts from prefix_len."""
    num_samples = len(prefix_grouper.group_info)
    max_len = prefix_grouper.padding_mask.size(1)
    device = prefix_grouper.padding_mask.device
    
    position_ids = torch.zeros(num_samples, max_len, dtype=torch.long, device=device)
    
    for i, group in enumerate(prefix_grouper.group_info):
        prefix_len = group.prefix_len
        
        position_ids[i, :prefix_len] = torch.arange(prefix_len, device=device)
        cur_pos = prefix_len
        for suffix_len in group.suffix_lens:
            if suffix_len > 0:
                position_ids[i, cur_pos:cur_pos + suffix_len] = torch.arange(
                    prefix_len, prefix_len + suffix_len, device=device
                )
                cur_pos += suffix_len
    
    return position_ids


def build_pg_from_micro_batch(
    micro_batch: dict,
    pad_token_id: int,
    padding_mode: str = "right",
):
    """Build PrefixGrouper from micro_batch dict containing prompts, responses, response_mask, uid."""
    prompts = micro_batch["prompts"]
    responses = micro_batch["responses"]
    response_mask = micro_batch["response_mask"]
    uids = micro_batch["uid"]

    bs = responses.size(0)

    group_sizes = []
    cur = 1
    for i in range(1, bs):
        if uids[i] == uids[i-1]:
            cur += 1
        else:
            group_sizes.append(cur)
            cur = 1
    group_sizes.append(cur)

    prefix_indices = []
    cursor = 0
    for gs in group_sizes:
        prefix_indices.append(cursor)
        cursor += gs
    prefix_indices = torch.tensor(prefix_indices, device=prompts.device)

    prefix_ids = prompts.index_select(0, prefix_indices)
    prefix_mask = prefix_ids.ne(pad_token_id)

    prefix_grouper = PrefixGrouper.from_ungrouped_masks(
        prefix_mask=prefix_mask,
        suffix_mask=response_mask,
        group_sizes=group_sizes,
        padding_mode=padding_mode,
        device=prompts.device,
    )

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
    logits = model(
        input_ids=concat_input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
        prefix_grouper=prefix_grouper,
    ).logits

    prefix_out, prefix_mask, suffix_out_raw, suffix_mask_raw = (
        prefix_grouper.split_output(
            logits, include_prefix_last=include_prefix_last)
    )

    completion_ids_right = prefix_grouper.convert_padding(
        completion_ids,
        completion_mask,
        padding_mode=padding_mode,
    )

    suffix_out = suffix_out_raw[:, :-1].float()
    suffix_mask = suffix_mask_raw[:, 1:]

    suffix_out /= temperature

    log_probs = logprobs_from_logits(suffix_out, completion_ids_right)

    entropy = None
    if calculate_entropy and entropy_fn is not None:
        entropy = entropy_fn(suffix_out)

    return log_probs, entropy, suffix_mask
