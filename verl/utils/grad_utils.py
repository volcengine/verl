import math
from typing import Iterable, Optional

import torch


def _collect_norm(grads: Iterable[torch.Tensor]) -> float:
    total = 0.0
    for grad in grads:
        if grad is None:
            continue
        total += float(torch.sum(grad.detach().float() ** 2))
    return math.sqrt(total)


def clip_grad_norm_fp32(
    parameters,
    max_norm: float,
    eps: float = 1e-6,
    extra_grads: Optional[Iterable[torch.Tensor]] = None,
    include_param_grads: bool = True,
):
    """Compute gradient norm in fp32, clip in-place, and return the pre-clip norm."""
    param_list = [p for p in parameters if p.grad is not None] if include_param_grads else []

    grads_for_norm = []
    if extra_grads is not None:
        grads_for_norm.extend(g for g in extra_grads if g is not None)
    if include_param_grads:
        grads_for_norm.extend(p.grad for p in param_list)

    if not grads_for_norm:
        return torch.tensor(0.0)

    total_norm = _collect_norm(grads_for_norm)
    clip_coef = max_norm / (total_norm + eps)

    if clip_coef < 1:
        device = grads_for_norm[0].device
        coef_tensor = torch.tensor(clip_coef, dtype=torch.float32, device=device)
        if extra_grads is not None:
            for grad in extra_grads:
                if grad is not None:
                    grad.mul_(coef_tensor.to(grad.dtype))
        if include_param_grads:
            for param in param_list:
                param.grad.mul_(coef_tensor.to(param.grad.dtype))

    return torch.tensor(total_norm)


def clip_grad_norm_bf16(
    parameters,
    max_norm: float,
    eps: float = 1e-6,
):
    """
    Compute gradient norm while staying in the gradient's native dtype.

    This mirrors the default torch.nn.utils.clip_grad_norm_ behaviour for
    bf16 gradients, avoiding explicit promotion to fp32 so that DeepSpeed
    matches the FSDP reference implementation.
    """
    param_list = [p for p in parameters if p.grad is not None]
    if not param_list:
        return torch.tensor(0.0)

    device = param_list[0].grad.device
    dtype = param_list[0].grad.dtype

    total = None
    for param in param_list:
        grad = param.grad.detach()
        if grad is None:
            continue
        grad_sq = torch.sum(grad * grad)
        total = grad_sq if total is None else total + grad_sq

    if total is None:
        return torch.tensor(0.0)

    total_norm = torch.sqrt(total)

    eps_tensor = torch.tensor(eps, dtype=dtype, device=device)
    max_norm_tensor = torch.tensor(max_norm, dtype=dtype, device=device)

    clip_coef = max_norm_tensor / (total_norm + eps_tensor)
    if clip_coef.item() < 1:
        coef = clip_coef.to(dtype)
        for param in param_list:
            param.grad.mul_(coef.to(param.grad.dtype))

    return total_norm.to(torch.float32)
