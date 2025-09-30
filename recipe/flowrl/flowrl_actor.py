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

import torch
from typing import Tuple, Dict, Any
from verl.workers.actor.dp_actor import DPActor
import verl.utils.torch_functional as verl_F


class ProjZModule(torch.nn.Module):
    """Projection network for estimating log partition function Z in FlowRL."""

    def __init__(self, hidden_size: int, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        layers = []

        for i in range(num_layers - 1):
            layers.extend([
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.GELU(),
                torch.nn.LayerNorm(hidden_size),
                torch.nn.Dropout(dropout)
            ])

        layers.append(torch.nn.Linear(hidden_size, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class FlowRLActor(DPActor):
    """FlowRL Actor that extends DPActor with partition function estimation."""

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.tb_coef = getattr(config.algorithm, 'tb_coef', 15.0)

    def _post_init_model(self):
        """Initialize the projection network after model loading."""
        super()._post_init_model()

        # Add projection network for log Z estimation
        if hasattr(self.actor_module.config, 'hidden_size'):
            hidden_size = self.actor_module.config.hidden_size
        else:
            # Fallback for different model architectures
            hidden_size = getattr(self.actor_module.config, 'd_model', 4096)

        proj_layers = getattr(self.config.actor, 'proj_layer', 3)
        proj_dropout = getattr(self.config.actor, 'proj_dropout', 0.1)

        self.actor_module.proj_z = ProjZModule(
            hidden_size=hidden_size,
            num_layers=proj_layers,
            dropout=proj_dropout
        )

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False, return_log_z=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Enhanced forward pass that can return log Z values."""

        # Standard forward pass
        logits, attention_mask, packed_input_ids = self._prepare_inputs(micro_batch, temperature)

        # Extract batch and sequence information
        batch_size = micro_batch['batch_size']
        response_length = micro_batch['response_length']

        # Compute log probabilities and entropy
        log_probs = self._compute_log_probs(logits, packed_input_ids, attention_mask)

        entropy = None
        if calculate_entropy:
            entropy = self._compute_entropy(logits, attention_mask)

        if not return_log_z:
            return entropy, log_probs

        # FlowRL specific: compute log Z
        # Get hidden states from the model output
        with torch.no_grad():
            outputs = self.actor_module(
                input_ids=packed_input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False
            )

        last_hidden = outputs.hidden_states[-1]  # Get last layer hidden states

        # Handle sequence packing if used
        if hasattr(self, 'use_ulysses_sp') and self.use_ulysses_sp:
            # Implementation depends on your specific setup
            pass

        # Reshape and extract prompt hidden states
        seqlen = attention_mask.size(1)
        full_last_hidden = last_hidden.view(batch_size, seqlen, -1)

        # Extract prompt hidden states (excluding response)
        prompts_last_hidden = full_last_hidden[:, :-response_length-1, :]
        prompt_attention_mask = attention_mask[:, :-response_length-1]

        # Compute average hidden state over prompt tokens
        avg_hidden = verl_F.masked_mean(
            prompts_last_hidden,
            prompt_attention_mask.unsqueeze(-1),
            axis=1
        )

        # Compute log Z using projection network
        log_z = self.actor_module.proj_z(avg_hidden)

        return entropy, log_probs, log_z

    def compute_flowrl_objective(self, logpf=None, logf_ref=None, logpf_old=None,
                                log_z=None, reward=None, response_mask=None, clip_ratio=None):
        """Compute FlowRL trajectory balance objective."""

        # Squeeze log_z to (B,)
        log_z = log_z.squeeze(-1)
        B = log_z.shape[0]

        # Mean of log p_f / log p_ref over valid tokens
        avg_logpf = verl_F.masked_mean(logpf, response_mask, axis=1)
        avg_logp_ref = verl_F.masked_mean(logf_ref, response_mask, axis=1)

        # Mean of token-level reward → log
        # We set R = exp(advantage); then log_reward = advantage
        seq_log_reward = verl_F.masked_mean(reward, response_mask, axis=1)

        # TB loss residual: log Z + log p_f - β * reward - log p_ref
        delta = log_z + avg_logpf - self.tb_coef * seq_log_reward - avg_logp_ref

        # Importance sampling
        log_w = verl_F.masked_sum(logpf - logpf_old, response_mask, axis=1)
        importance_weight = torch.exp(log_w).detach()
        clip_importance_weight = torch.clamp(importance_weight, 1 - clip_ratio, 1 + clip_ratio)

        # Weighted trajectory balance loss
        weighted_losses = clip_importance_weight * (delta ** 2)
        avg_loss = torch.mean(weighted_losses)

        # Loss statistics for monitoring
        loss_term_dict = {
            "actor/logpf": verl_F.masked_mean(logpf, response_mask).detach().item(),
            "actor/logp_ref": verl_F.masked_mean(logf_ref, response_mask).detach().item(),
            "actor/log_z": log_z.mean().detach().item(),
            "actor/log_reward": verl_F.masked_mean(reward, response_mask).detach().item(),
            "actor/tb_loss": avg_loss.detach().item(),
            "actor/delta_mean": delta.mean().detach().item(),
            "actor/delta_std": delta.std().detach().item(),
            "actor/importance_weight_mean": importance_weight.mean().detach().item(),
        }

        return avg_loss, loss_term_dict

    def _compute_actor_loss(self, data):
        """Override to use FlowRL objective instead of PPO loss."""

        micro_batch = data
        temperature = self.config.actor_rollout_ref.actor.temperature
        calculate_entropy = self.entropy_bonus > 0

        # FlowRL forward pass with log Z
        entropy, log_prob, log_z = self._forward_micro_batch(
            micro_batch=micro_batch,
            temperature=temperature,
            calculate_entropy=calculate_entropy,
            return_log_z=True
        )

        # Extract data for FlowRL objective
        old_log_prob = data['old_log_prob']
        ref_log_prob = data['ref_log_prob']
        advantages = data['advantages']
        response_mask = data['response_mask']

        # Compute FlowRL trajectory balance loss
        policy_loss, loss_stats = self.compute_flowrl_objective(
            logpf=log_prob,
            logf_ref=ref_log_prob,
            logpf_old=old_log_prob,
            log_z=log_z,
            reward=advantages,
            response_mask=response_mask,
            clip_ratio=self.config.clip_ratio
        )

        # Add entropy bonus if configured
        total_loss = policy_loss
        if calculate_entropy and entropy is not None:
            entropy_loss = -self.entropy_bonus * verl_F.masked_mean(entropy, response_mask)
            total_loss += entropy_loss
            loss_stats['actor/entropy'] = verl_F.masked_mean(entropy, response_mask).detach().item()
            loss_stats['actor/entropy_loss'] = entropy_loss.detach().item()

        loss_stats['actor/total_loss'] = total_loss.detach().item()

        return total_loss, loss_stats