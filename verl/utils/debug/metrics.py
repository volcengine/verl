import torch

from verl.protocol import DataProto
import logging

logger = logging.getLogger(__file__)

def calculate_token_list_diff(tensor1: torch.Tensor, tensor2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # verify inputs
    if tensor1.numel() == 0 or tensor2.numel() == 0:
        return torch.zeros(tensor1.shape[0], dtype=torch.long, device=tensor1.device)
    if tensor1.shape!=tensor2.shape or mask.shape!=tensor1.shape or mask.shape!=tensor2.shape:
        print(f"<WARN> dim of tensor1, tensor2, mask is not equal, {(tensor1.shape)=},{(tensor2.shape)=}, {(mask.shape)=}")
        return torch.ones_like(tensor1)
    # transfer to same device
    if tensor2.device != tensor1.device:
        tensor2 = tensor2.to(tensor1.device)
    if mask.device != tensor1.device:
        mask = mask.to(tensor1.device)
    
    # calculate diff
    diff_mask = (tensor1 != tensor2)
    
    valid_diff_mask = diff_mask & (mask == 1)
    
    diff_counts = valid_diff_mask.sum(dim=1)
    
    return diff_counts
def pearson_correlation_coefficient(tensor1: torch.Tensor, tensor2: torch.Tensor, mask: torch.Tensor)->torch.Tensor:
    # implemention of https://arxiv.org/pdf/2506.13585
    if tensor1.shape!=tensor2.shape or mask.shape!=tensor1.shape or mask.shape!=tensor2.shape:
        return 0
    mt1 = torch.masked_select(tensor1, mask)
    mt2 = torch.masked_select(tensor2, mask)
    result = torch.corrcoef(torch.stack([mt1, mt2], dim=0))
    return result[0][1].detach().item()
def calculate_log_prob_diff(log_probs1: torch.Tensor, log_probs2: torch.Tensor, mask: torch.Tensor)->torch.Tensor: 
    full_diff = torch.abs(log_probs1 - log_probs2)
    return torch.masked_select(full_diff, mask)
    
def calculate_debug_metrics(data: DataProto)->dict:
    rollout_old_log_probs = data.batch["rollout_log_probs"]
    actor_old_log_probs = data.batch["old_log_probs"]
    if "loss_mask" in data.batch:
        logger.debug('loss mask found, use it to mask log probs')
        log_prob_mask = data.batch["loss_mask"]
    elif 'attention_mask' in data.batch:
        log_prob_mask = data.batch["attention_mask"]
    else:
        logger.warning(f'no loss mask found, use all log probs, {(data.batch.keys())=}')
    responses = data.batch["responses"]
    response_length = responses.size(1)

    response_mask = log_prob_mask[:, -response_length:]
    # calculate pearson corrcoef
    actor_probs = torch.exp(actor_old_log_probs)
    rollout_probs = torch.exp(rollout_old_log_probs)
    response_mask_bool = response_mask.bool()
    pearson_corrcoef = pearson_correlation_coefficient(actor_probs, rollout_probs, response_mask_bool)
    rollout_probs_diff = calculate_log_prob_diff(actor_probs, rollout_probs, response_mask_bool)
    return {
        "training/rollout_probs_diff_valid": 1,
        "training/rollout_probs_diff_max": torch.max(rollout_probs_diff).detach().item(),
        "training/rollout_probs_diff_mean": torch.mean(rollout_probs_diff).detach().item(),
        "training/rollout_probs_diff_std": torch.std(rollout_probs_diff).detach().item(),
        "training/rollout_actor_probs_pearson_corr": pearson_corrcoef
    }
