import torch
import verl.utils.torch_functional as verl_F
from verl.trainer.ppo import core_algos

def compute_truncate_gae_advantage_return(token_level_rewards: torch.Tensor, single_token_level_rewards: torch.Tensor, values: torch.Tensor, eos_mask: torch.Tensor,
                                 gamma: torch.Tensor, lam: torch.Tensor, use_variable_lambda: torch.Tensor,
                                 variable_lambda_scalar: torch.Tensor, adv_whiten: bool, use_separate_critic_lam: bool,
                                 critic_lam: torch.Tensor, is_finished: torch.Tensor, ignore_token_num: int, rounds_eos_mask: torch.Tensor, seq_len_per_sample: torch.Tensor, is_clamp: bool):
    window_mask = torch.ones_like(eos_mask)
    window_mask[:, -ignore_token_num:] = 0
    window_mask = torch.maximum(window_mask, torch.ones_like(eos_mask) * is_finished.unsqueeze(-1))
    values = values * eos_mask
    token_level_rewards = token_level_rewards * rounds_eos_mask[:, :token_level_rewards.size(-1)]
    if use_variable_lambda:
        # seq_len_per_sample = torch.clamp(torch.sum(rounds_eos_mask, dim=1), min=1.0)
        seq_len_per_sample += (1 - is_finished) * values.shape[-1] // 2 
        lam = torch.clamp(1 - 1 / (variable_lambda_scalar * seq_len_per_sample), min=lam)
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = values.shape[-1]
        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = single_token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            if t == gen_len - 1:
                delta = delta * is_finished + ~is_finished * (gamma-1) * values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        advantages *= window_mask
        if use_separate_critic_lam and critic_lam == 1:
            cumsum_rewards = torch.cumsum(token_level_rewards, dim=1)
            returns = token_level_rewards - cumsum_rewards + cumsum_rewards[:, -1:None]
            returns *= is_finished.unsqueeze(-1)
        origin_advantages = advantages
        if adv_whiten:
            advantages = verl_F.masked_whiten(origin_advantages, eos_mask*window_mask)
            advantages = advantages * eos_mask * window_mask
        else:
            advantages = torch.clone(origin_advantages)
        if is_clamp:
            advantages = torch.clamp(advantages, max=10.0, min=-10.0)
    return origin_advantages, advantages, returns


def compute_lm_loss(log_prob, raw_scores, eos_ids, rounds_eos_mask=None, loss_average_method='token'):
    if rounds_eos_mask is not None:
        log_prob = log_prob[:, :rounds_eos_mask.shape[-1]]
    eos_ids = eos_ids.unsqueeze(1)
    scores = torch.gather(raw_scores, 1, eos_ids)
    ids = torch.arange(log_prob.shape[1], device=eos_ids.device).unsqueeze(0).repeat(log_prob.shape[0], 1)
    mask0 = ids <= eos_ids
    mask1 = (scores > 0).repeat(1, log_prob.shape[1])
    mask = mask0 & mask1
    if rounds_eos_mask is not None:
        mask &= rounds_eos_mask.bool()
    lm_loss = torch.masked_select(log_prob, mask)
    if loss_average_method in ['sample', 'token']:
        lm_loss = -torch.sum(lm_loss) / max(lm_loss.numel(), 1) 
    elif loss_average_method in ['minibatch', 'batch']:
        lm_loss = -torch.sum(lm_loss)
    else:
        raise NotImplementedError(f"loss_average_method {loss_average_method} not implemented") 
    return lm_loss


def compute_kl_loss(log_prob, ref_log_prob, eos_mask, kl_penalty_, loss_average_method='token'):
    if kl_penalty_ in ("abs", "mse", "low_var_kl"):
        kl = core_algos.kl_penalty(log_prob, ref_log_prob, kl_penalty_)
    elif kl_penalty_ in ("kl"):
        kl = core_algos.kl_penalty(log_prob, ref_log_prob, kl_penalty_).square()
    else:
        raise NotImplementedError
    if loss_average_method == 'sample':
        seq_len_per_sample = torch.clamp(torch.sum(eos_mask, dim=1), min=1.0)
        kl_loss = torch.mean(torch.sum(kl * eos_mask, dim=1) / seq_len_per_sample)
    elif loss_average_method == 'token':
        kl_loss = (kl * eos_mask).sum() / (eos_mask.sum() + 1e-6)
    elif loss_average_method in ['minibatch', 'batch']:
        kl_loss = (kl * eos_mask).sum()
    return kl_loss


def compute_value_loss(vpreds, returns, values, eos_mask, cliprange_value_low, cliprange_value_high, overlong_mask):
    """Compute the value loss. Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151
    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)
    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped
    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value_low, values + cliprange_value_high)
    vf_losses1 = (vpreds - returns)**2
    vf_losses2 = (vpredclipped - returns)**2
    seq_len_per_sample = torch.clamp(torch.sum(eos_mask, dim=1), min=1.0)
    if overlong_mask is not None:
        vf_loss = 0.5 * torch.mean(
            torch.sum(torch.max(vf_losses1, vf_losses2) * eos_mask, dim=1) / seq_len_per_sample * overlong_mask)
    else:
        vf_loss = 0.5 * torch.mean(torch.sum(torch.max(vf_losses1, vf_losses2) * eos_mask, dim=1) / seq_len_per_sample)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), eos_mask)
    vf_loss = vf_loss
    return vf_loss, vf_clipfrac


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.

    Args:
        loss_mat: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_agg_mode: (str) choices:
            method to aggregate the loss matrix into a scalar.
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)  # token-mean
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]  # The divisor
        # (loss_mask.shape[-1]) should ideally be constant
        # throughout training to well-replicate the DrGRPO paper.
        # TODO: Perhaps add user-defined normalizer argument to
        # agg_loss to ensure divisor stays constant throughout.
    elif loss_agg_mode in ['minibatch-sum', 'batch-sum']:
        # NOTE(HanlinDu): this case is the only incremental difference from ppo.core_algos.agg_loss
        # Maybe we can replace the original one with this one in the TPPO usages,
        # or even in all of the Verl PPO usages.
        loss = (loss_mat * loss_mask).sum()
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=10.0, # cliprange2
    loss_agg_mode: str = "batch-sum",
    eos_ids=None,
    overlong_mask=None,
    acc_mask=None,
    kl_penalty='low_var_kl',
):
    negative_approx_kl = core_algos.kl_penalty(log_prob, old_log_prob, kl_penalty=kl_penalty)
    # Clamp negative_approx_kl for stability
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    
    pg_losses1 = -advantages * ratio
    if cliprange_high is None:
        assert cliprange is not None, "cliprange_high is None, but cliprange is also None, please set one of them"
        cliprange_high = cliprange

    if cliprange_low is None:
        assert cliprange is not None, "cliprange_low is None, but cliprange is also None, please set one of them"
        cliprange_low = cliprange

    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange_low, 1.0 + cliprange_high)
    pg_losses3 = torch.abs(-advantages * clip_ratio_c)
    pg_losses_clip = torch.maximum(pg_losses1, pg_losses2)
    pg_losses = torch.minimum(pg_losses_clip, pg_losses3)

    if overlong_mask is not None:
        response_mask = response_mask * overlong_mask.unsqueeze(-1)
    if acc_mask is not None:
        response_mask = response_mask * acc_mask.unsqueeze(-1)

    pg_loss = core_algos.agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    pg_clipfrac_lower = verl_F.masked_mean(torch.gt(pg_losses_clip, pg_losses3) * (advantages < 0).float(), response_mask)
    
    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower






