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

import math
import torch
from verl.trainer.distillation.common import DistillationLossInputs
from verl.workers.config import DistillationConfig
from verl.utils.metric import Metric, AggregationType
from typing import Any, Callable 

def clamp_log_probs(log_p: torch.Tensor, log_q: torch.Tensor, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    """Clamp log probabilities to avoid numerical instability and handle inf minus inf for masked top-k probs."""
    min_log_prob = math.log(eps)
    log_p_clamped = torch.clamp(log_p, min=min_log_prob)
    log_q_clamped = torch.clamp(log_q, min=min_log_prob)
    return log_p_clamped, log_q_clamped


def kl_divergence(log_q: torch.Tensor, log_p: torch.Tensor, take_abs: bool) -> torch.Tensor:
    """Compute KL divergence between two distributions given their log probabilities."""
    kld = torch.nn.functional.kl_div(input=log_q, target=log_p, reduction="none", log_target=True)
    if take_abs:
        kld = kld.abs()
    return kld.sum(dim=-1)


def jensen_shannon_divergence(
    log_q: torch.Tensor, log_p: torch.Tensor, beta: float, take_abs: bool = False
) -> torch.Tensor:
    """
    Compute Jensen-Shannon Divergence between two distributions given their log probabilities.

    JSD(β) = β * KL(p || m) + (1 - β) * KL(q || m), where m = beta * p + (1 - beta) * q

    The gradients of JSD(β) behave similarly to forward KL and reverse KL when β is close
    to 0 and 1 respectively. See https://arxiv.org/abs/2306.13649

    Args:
        log_q (torch.Tensor):
            Student log probabilities, shape (batch_size, response_length, vocab_size) or
            (batch_size, response_length, topk).
        log_p (torch.Tensor):
            Teacher log probabilities, same shape as log_q.
        beta (float):
            JSD interpolation weight. When beta=0, behaves like forward KL.
            When beta=1, behaves like reverse KL.
        take_abs (bool):
            Whether to take the absolute value of log_p - log_q before summing.

    Returns:
        torch.Tensor: JSD loss per token, shape (batch_size, response_length).
    """
    log_p, log_q = clamp_log_probs(log_p, log_q)
    q = log_q.exp()
    p = log_p.exp()
    m = beta * p + (1 - beta) * q
    log_m = m.log()
    kl1 = kl_divergence(log_q=log_m, log_p=log_p, take_abs=take_abs)
    kl2 = kl_divergence(log_q=log_m, log_p=log_q, take_abs=take_abs)
    loss = beta * kl1 + (1 - beta) * kl2
    return loss


def kullback_leibler_divergence(
    log_q: torch.Tensor, log_p: torch.Tensor, loss_mode: str, take_abs: bool = False
) -> torch.Tensor:
    """
    Compute forward or reverse KL divergence between two distributions given their log probabilities.

    forward KL: KL(p || q) = sum(p * (log_p - log_q))
    reverse KL: KL(q || p) = sum(q * (log_q - log_p))

    Args:
        log_q (torch.Tensor):
            Student log probabilities, shape (batch_size, response_length, vocab_size) or
            (batch_size, response_length, topk).
        log_p (torch.Tensor):
            Teacher log probabilities, same shape as log_q.
        loss_mode (str):
            KL divergence direction: "forward" or "reverse".
        take_abs (bool):
            Whether to take the absolute value of log_p - log_q before summing. This can help
            when using the top-k loss, where distributions may not sum to 1 and can be negative.

    Returns:
        torch.Tensor: KL divergence loss per token, shape (batch_size, response_length).
    """
    log_p, log_q = clamp_log_probs(log_p, log_q)
    match loss_mode:
        case "forward":
            return kl_divergence(log_q=log_q, log_p=log_p, take_abs=take_abs)
        case "reverse":
            return kl_divergence(log_q=log_p, log_p=log_q, take_abs=take_abs)
        case _:
            raise ValueError(f"Unsupported loss mode: {loss_mode}. Supported modes are: ['forward', 'reverse']")


def compute_distillation_loss_topk(
    inputs: DistillationLossInputs,
    response_mask: torch.Tensor,
    config: DistillationConfig,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """topk distillation loss."""

    loss_settings = config.loss_settings
    teacher_topk_logprobs = inputs.teacher_topk_logprobs
    student_topk_logprobs = inputs.student_topk_logprobs
    if teacher_topk_logprobs is None or student_topk_logprobs is None:
        raise ValueError("Expected teacher_topk_logprobs and student_topk_logprobs to be provided in inputs.")

    teacher_topk_indices = inputs.teacher_topk_indices
    student_topk_indices = inputs.student_topk_indices
    if teacher_topk_indices is None or student_topk_indices is None:
        raise ValueError("Expected teacher_topk_indices and student_topk_indices to be provided in inputs.")
    if not teacher_topk_indices.equal(student_topk_indices):
        raise ValueError(
            "Expected teacher and student topk indices to be the same, "
            f"but got {teacher_topk_indices} and {student_topk_indices}."
        )

    topk = config.topk
    if loss_settings.use_student_topk and loss_settings.use_teacher_topk:
        expected_num_logprobs = 2 * topk
    elif loss_settings.use_student_topk or loss_settings.use_teacher_topk:
        expected_num_logprobs = topk
    else:
        raise ValueError(
            f"Expected at least one of student or teacher topk logprobs to be used, "
            f"but got {loss_settings.use_student_topk} and {loss_settings.use_teacher_topk}."
        )
    if (
        teacher_topk_logprobs.shape[-1] != expected_num_logprobs
        or student_topk_logprobs.shape[-1] != expected_num_logprobs
    ):
        raise ValueError(
            f"Expected topk logprobs to have shape (batch_size, response_length, {expected_num_logprobs}), "
            f"but got {teacher_topk_logprobs.shape=} and {student_topk_logprobs.shape=}."
        )

    # Log amount of mass in the top-k log probabilities for both student and teacher.
    student_mass = student_topk_logprobs.exp().sum(dim=-1)[response_mask]
    teacher_mass = teacher_topk_logprobs.exp().sum(dim=-1)[response_mask]
    distillation_metrics = {
        "distillation/student_mass": student_mass.mean().item(),
        "distillation/student_mass_min": Metric(AggregationType.MIN, student_mass.min()),
        "distillation/student_mass_max": Metric(AggregationType.MAX, student_mass.max()),
        "distillation/teacher_mass": teacher_mass.mean().item(),
        "distillation/teacher_mass_min": Metric(AggregationType.MIN, teacher_mass.min()),
        "distillation/teacher_mass_max": Metric(AggregationType.MAX, teacher_mass.max()),
    }
    loss_mode = config.loss_mode
    match loss_mode:
        case "forward_kl_topk":
            distillation_losses = kullback_leibler_divergence(
                log_q=student_topk_logprobs, log_p=teacher_topk_logprobs, loss_mode="forward"
            )
        case "reverse_kl_topk":
            distillation_losses = kullback_leibler_divergence(
                log_q=student_topk_logprobs, log_p=teacher_topk_logprobs, loss_mode="reverse"
            )
        case "jsd_topk":
            distillation_losses = jensen_shannon_divergence(
                log_q=student_topk_logprobs, log_p=teacher_topk_logprobs, beta=config.jsd_beta
            )
        case _:
            raise NotImplementedError(f"Unsupported distillation loss mode: {config.loss_mode}")
    return distillation_losses, distillation_metrics