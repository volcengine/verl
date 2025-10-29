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

import logging
import os

import torch

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor.dp_actor import DataParallelPPOActor

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ProjZModule(torch.nn.Module):
    """Projection network for estimating log partition function Z in FlowRL."""

    def __init__(self, hidden_size: int, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        layers = []

        for i in range(num_layers - 1):
            layers.extend(
                [
                    torch.nn.Linear(hidden_size, hidden_size),
                    torch.nn.GELU(),
                    torch.nn.LayerNorm(hidden_size),
                    torch.nn.Dropout(dropout),
                ]
            )

        layers.append(torch.nn.Linear(hidden_size, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class FlowRLActor(DataParallelPPOActor):
    """FlowRL Actor that extends DataParallelPPOActor with partition function estimation."""

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        # FlowRL hyperparameters (hardcoded as per paper)
        self.flowrl_beta_coef = 15.0  # β coefficient for reward scaling in flowrl loss

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False, return_log_z=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    output_hidden_states=True if return_log_z else False,  # FlowRL: for log_z estimation
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    output_hidden_states=True if return_log_z else False,  # FlowRL: for log_z estimation
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            # ==== FlowRL: use proj_z to estimate log Z ====
            if return_log_z:
                last_hidden = output.hidden_states[-1].squeeze(0)  # (total_nnz, hidden size)
                if self.use_ulysses_sp:
                    last_hidden = gather_outputs_and_unpad(
                        last_hidden,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                full_last_hidden = pad_input(
                    hidden_states=last_hidden, indices=indices, batch=batch_size, seqlen=seqlen
                )
                # extract pormpt hiddenstate for log z
                prompts_last_hidden = full_last_hidden[:, : -response_length - 1]
                prompt_attention_mask = attention_mask[:, : -response_length - 1]
                avg_hidden = verl_F.masked_mean(prompts_last_hidden, prompt_attention_mask.unsqueeze(-1), axis=1)

                log_z = self.actor_module.proj_z(avg_hidden)

                return entropy, log_probs, log_z
            else:
                return entropy, log_probs

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # if self.config.tis_imp_ratio_cap > 0:
        #     assert "rollout_log_probs" in data.batch.keys(), (
        #         "Truncated Importance Sampling (TIS) requires to configure "
        #         "`actor_rollout_ref.rollout.calculate_log_probs=True` "
        #         "and is not currently supported in Server mode (agent loop)."
        #     )
        #     select_keys.append("rollout_log_probs")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]
                    ref_log_prob = model_inputs["ref_log_prob"]

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # FlowRL: always compute log_z, no entropy needed
                    _, log_prob, log_z = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=False, return_log_z=True
                    )

                    if on_policy:
                        old_log_prob = log_prob.detach()
                    else:
                        old_log_prob = model_inputs["old_log_probs"]

                    # loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla
                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    # policy_loss_fn = get_policy_loss_fn(loss_mode)
                    # pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                    #     old_log_prob=old_log_prob,
                    #     log_prob=log_prob,
                    #     advantages=advantages,
                    #     response_mask=response_mask,
                    #     loss_agg_mode=loss_agg_mode,
                    #     config=self.config,
                    #     rollout_log_probs=rollout_log_probs,
                    # )
                    # Compute FlowRL trajectory balance loss
                    policy_loss, flowrl_metrics = self.compute_flowrl_objective(
                        log_prob=log_prob,
                        ref_log_prob=ref_log_prob,
                        old_log_prob=old_log_prob,
                        log_z=log_z,
                        reward=advantages,
                        response_mask=response_mask,
                        clip_ratio=self.config.clip_ratio,
                        # rollout_log_probs=rollout_log_probs,
                    )

                    # if entropy_coeff != 0:
                    #     entropy_loss = agg_loss(
                    #         loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode
                    #     )

                    #     # compute policy loss
                    #     policy_loss = pg_loss - entropy_loss * entropy_coeff
                    # else:
                    #     policy_loss = pg_loss

                    # if self.config.use_kl_loss:
                    #     ref_log_prob = model_inputs["ref_log_prob"]
                    #     # compute kl loss
                    #     kld = kl_penalty(
                    #         logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                    #     )
                    #     kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                    #     policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                    #     micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                    #     micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    loss.backward()

                    micro_batch_metrics.update(flowrl_metrics)
                    # micro_batch_metrics.update(
                    #     {
                    #         "actor/pg_loss": pg_loss.detach().item() * loss_scale_factor,
                    #         "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                    #         "actor/ppo_kl": ppo_kl.detach().item(),
                    #         "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                    #     }
                    # )
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics

    def compute_flowrl_objective(
        self,
        log_prob=None,
        ref_log_prob=None,
        old_log_prob=None,
        log_z=None,
        reward=None,
        response_mask=None,
        clip_ratio=None,
        rollout_log_probs=None,
    ):
        log_ratio = log_prob - old_log_prob  # (B, T)
        ratio = torch.exp(log_ratio)  # (B, T)

        condition_1 = (reward > 0) & (ratio > 1.0 + self.config.clip_ratio_high)  # (B, T)
        condition_2 = (reward < 0) & (ratio < 1.0 - self.config.clip_ratio_low)  # (B, T)

        # CISPO mask
        cispo_mask = ~(condition_1 | condition_2)
        cispo_mask = cispo_mask.float()
        combined_mask = response_mask * cispo_mask

        # squeeze log_z to (B,)
        log_z = log_z.squeeze(-1)

        # Average token log-probs & rewards over valid positions
        avg_log_prob = verl_F.masked_mean(log_prob, combined_mask, axis=1)
        avg_ref_log_prob = verl_F.masked_mean(ref_log_prob, combined_mask, axis=1)
        seq_log_reward = verl_F.masked_mean(reward, combined_mask, axis=1)

        # FlowRL residual: logZ + logpf - β*R - logpref
        delta = log_z + avg_log_prob - self.flowrl_beta_coef * seq_log_reward - avg_ref_log_prob

        # Importance ratio from current vs old policy (product of token ratios)
        log_w = verl_F.masked_sum(log_prob - old_log_prob, combined_mask, axis=1)
        imp_w_raw = torch.exp(log_w).detach()

        # Clamp importance weight for numerical stability (prevent extreme values)
        # imp_w = torch.clamp(imp_w_raw, max=10.0)
        imp_w = torch.clamp(imp_w_raw, 1 - self.config.clip_ratio_low, 1 + self.config.clip_ratio_high)

        # Loss: weighted squared residual with clipped importance weights
        weighted_losses = imp_w * (delta**2)
        avg_loss = torch.mean(weighted_losses)

        # PPO KL: negative_approx_kl = log_prob - old_log_prob
        negative_approx_kl = log_prob - old_log_prob
        ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

        # Reference KL: approx_kl_ref = log_prob - ref_log_prob
        approx_kl_ref = log_prob - ref_log_prob
        ref_kl = verl_F.masked_mean(-approx_kl_ref, response_mask)

        # cispo
        total_tokens = response_mask.sum()
        cispo_dropped = (response_mask * (1 - cispo_mask)).sum()
        cispo_mask_ratio = cispo_dropped / (total_tokens + 1e-8)

        # Metrics
        loss_term_dict = {
            "actor/log_prob": verl_F.masked_mean(log_prob, response_mask).detach().item(),
            "actor/old_log_prob": verl_F.masked_mean(old_log_prob, response_mask).detach().item(),
            "actor/ref_log_prob": verl_F.masked_mean(ref_log_prob, response_mask).detach().item(),
            "actor/log_z": log_z.mean().detach().item(),
            "actor/log_reward": verl_F.masked_mean(reward, response_mask).detach().item(),
            "actor/final_loss": avg_loss.detach().item(),
            "actor/importance_weight_raw": imp_w_raw.mean().detach().item(),
            "actor/importance_weight": imp_w.mean().detach().item(),
            "actor/ppo_kl": ppo_kl.detach().item(),  # PPO-style KL (current vs old policy)
            "actor/ref_kl": ref_kl.detach().item(),  # KL with reference policy
            "actor/cispo_mask_ratio": cispo_mask_ratio.detach().item(),  # cispo
            "actor/cispo_dropped_tokens": cispo_dropped.detach().item(),  # cispo
            "actor/condition_1_count": (condition_1 * response_mask).sum().detach().item(),  # cispo
            "actor/condition_2_count": (condition_2 * response_mask).sum().detach().item(),  # cispo
        }

        return avg_loss, loss_term_dict
