# Copyright 2025 Bytedance Ltd. and/or its affiliates
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


import torch
from tensordict import TensorDict

from verl.trainer.ppo.core_algos import agg_loss, compute_value_loss, get_policy_loss_fn, kl_penalty
from verl.utils.torch_functional import masked_mean
from verl.workers.config import ActorConfig, CriticConfig


def sft_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    log_prob = model_output["log_probs"]  # [bsz, response_length]
    response_mask = data["response_mask"].to(bool)
    loss = -torch.mean(log_prob * response_mask)
    return loss, {"loss": loss.detach().item()}


def ppo_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    log_prob = model_output["log_probs"]
    entropy = model_output.get("entropy", None)

    metrics = {}

    response_mask = data["response_mask"].to(bool)
    # compute policy loss
    old_log_prob = data["old_log_probs"]
    advantages = data["advantages"]

    loss_agg_mode = config.loss_agg_mode

    loss_mode = config.policy_loss.get("loss_mode", "vanilla")

    policy_loss_fn = get_policy_loss_fn(loss_mode)
    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
    )

    metrics.update(
        {
            "pg_loss": pg_loss.detach().item(),
            "pg_clipfrac": pg_clipfrac.detach().item(),
            "ppo_kl": ppo_kl.detach().item(),
            "pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
        }
    )
    policy_loss = pg_loss

    # add entropy loss
    if entropy is not None:
        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
        entropy_coeff = config.entropy_coeff
        policy_loss -= entropy_coeff * entropy_loss

    # add kl loss
    if config.use_kl_loss:
        ref_log_prob = data["ref_log_prob"]
        # compute kl loss
        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=config.kl_loss_type)
        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=config.loss_agg_mode)

        policy_loss += kl_loss * config.kl_loss_coef
        metrics["kl_loss"] = kl_loss.detach().item()
        metrics["kl_coef"] = config.kl_loss_coef

    return policy_loss, metrics


def value_loss(config: CriticConfig, model_output, data: TensorDict, dp_group=None):
    vpreds = model_output["values"]
    values = data["values"]

    values = data["values"]
    returns = data["returns"]
    response_mask = data["response_mask"].to(bool)

    vf_loss, vf_clipfrac = compute_value_loss(
        vpreds=vpreds,
        values=values,
        returns=returns,
        response_mask=response_mask,
        cliprange_value=config.cliprange_value,
        loss_agg_mode=config.loss_agg_mode,
    )

    metrics = {}

    metrics.update(
        {
            "critic/vf_loss": vf_loss.detach().item(),
            "critic/vf_clipfrac": vf_clipfrac.detach().item(),
            "critic/vpred_mean": masked_mean(vpreds, response_mask).detach().item(),
        }
    )

    return vf_loss, metrics
