# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""Explicit text LoRA targets: attention projections and every MLP. Not the router, the output embeddings, or
the sparse-attention indexer (its selection is discrete, so its adapters would never receive a gradient).
"""

import copy
import re

import torch
import torch.distributed as dist
from peft import LoraConfig, TaskType, get_peft_model
from torch import nn
from torch.nn import functional as F

from .modeling_step5 import Step5PackedProjection

# Routed-expert factors next to the expert input (gate/up A) and output (down B): one copy per layer when shared.
SHARED_OUTER = {"gate_proj": "A", "up_proj": "A", "down_proj": "B"}


def text_lora_config(model, *, rank=32, alpha=64):
    modules = [
        name
        for name, child in model.named_modules()
        if isinstance(child, nn.Linear)
        and name != "lm_head"
        and not name.endswith(".moe.gate")
        and ".sparse_indexer_" not in name
    ]
    parameters = [name + ".weight" for name, child in model.named_modules() if isinstance(child, Step5PackedProjection)]
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        target_modules=modules,
        target_parameters=parameters,
        bias="none",
        lora_dropout=0,
    )


def attach_text_lora(model, *, rank=32, alpha=64, master_dtype=torch.float32, shared_outer=True):
    """Adapters are trainable master weights in FP32 (optimizer state follows); the frozen base keeps its dtype.

    shared_outer: all routed experts of a layer share gate A, up A and down B (one [r, in] / [out, r] matrix
    each); gate B, up B and down A stay per expert. The export expands them to the per-expert serving format.

    Compute stays in the base dtype and the low-rank product is added, never folded into the base weight:
    ordinary Linear adapters cast their weights in forward (_compute_in); packed experts take the factors in
    the base dtype and add s * (x A^T) B^T per expert (_unfold_expert_lora), as vLLM's LoRA kernels do.
    BF16 masters would drop Adam updates below BF16 resolution (lr 1e-5 on |A| ~ 1e-2 is a 1e-3 relative step)."""
    model = get_peft_model(model, text_lora_config(model, rank=rank, alpha=alpha), autocast_adapter_dtype=False)
    for name, child in model.named_modules():
        if hasattr(child, "lora_A") and hasattr(child, "get_base_layer"):
            if shared_outer and isinstance(child.get_base_layer(), Step5PackedProjection):
                _share_outer_factor(child, SHARED_OUTER[name.rsplit(".", 1)[-1]])
            child.lora_A.to(dtype=master_dtype)
            child.lora_B.to(dtype=master_dtype)
            base = child.get_base_layer()
            if isinstance(base, nn.Linear):
                for adapters in (child.lora_A, child.lora_B):
                    for linear in adapters.values():
                        _compute_in(linear, base.weight.dtype)
            elif isinstance(base, Step5PackedProjection):
                _unfold_expert_lora(child, base)
    return model


def _share_outer_factor(wrapper, side):
    """Replace the packed per-expert factor on `side` by one matrix shared by all experts of this projection.

    Built on real devices, A keeps the first expert's initialization and B is zero, so the delta is zero at init.
    On the meta device (the training startup) nothing is initialized here: load_sharded_text_checkpoint fills every
    lora_A (kaiming) and lora_B (zero) by name after to_empty, whatever its shape."""
    adapters = wrapper.lora_A if side == "A" else wrapper.lora_B
    for key in list(adapters):
        old = adapters[key]
        rank = wrapper.r[key]
        if side == "A":
            new = nn.Linear(old.in_features, rank, bias=False, device=old.weight.device, dtype=old.weight.dtype)
            if not old.weight.is_meta:
                new.weight.data.copy_(old.weight.data[:rank])
        else:
            new = nn.Linear(rank, old.out_features, bias=False, device=old.weight.device, dtype=old.weight.dtype)
            if not old.weight.is_meta:
                nn.init.zeros_(new.weight)
        new.weight.requires_grad_(old.weight.requires_grad)
        adapters[key] = new


def expert_factors(wrapper, adapter, dtype):
    """(lhs, rhs, scaling) with delta = scaling * lhs @ rhs: [E, out, r] / [E, r, in] per expert, or a 2-D
    [out, r] / [r, in] matrix for a factor shared by all experts. PEFT packs per-expert A as [E*r, in] and B as
    [out, r*E] (rank-major), the layout the exporter unpacks."""
    a = wrapper.lora_A[adapter].weight.to(dtype)
    b = wrapper.lora_B[adapter].weight.to(dtype)
    experts, rank = wrapper.num_experts, wrapper.r[adapter]
    rhs = a if a.shape[0] == rank and experts != 1 else a.reshape(experts, rank, a.shape[-1])
    lhs = b if b.shape[1] == rank and experts != 1 else b.reshape(b.shape[0], rank, experts).permute(2, 0, 1)
    return lhs, rhs, wrapper.scaling[adapter]


def _unfold_expert_lora(wrapper, base):
    """Replace PEFT's folded W + s*B@A expert stack by the projection's unfolded low-rank path: the factors are
    cast to the base dtype and applied as s * (x A^T) B^T, the same A/B the exporter sends to vLLM."""
    if wrapper.parameter_name != "weight" or wrapper.num_experts != base.weight.shape[0]:
        raise ValueError("Unexpected packed-expert LoRA wrapper")

    def forward(x, expert, base_x=None):
        if wrapper.merged:
            raise RuntimeError("Merged packed-expert LoRA is not supported; keep adapters separate")
        if wrapper.disable_adapters:
            return base(x, expert, base_x=base_x)
        active = wrapper.active_adapters
        if len(active) != 1:
            raise RuntimeError("Packed-expert LoRA supports exactly one active adapter")
        return base(x, expert, lora=expert_factors(wrapper, active[0], base.weight.dtype), base_x=base_x)

    wrapper.forward = forward


def _compute_in(linear, dtype):
    """Run an adapter Linear in `dtype` (as the serving engine does); gradients reach the FP32 master."""

    def forward(x):
        bias = None if linear.bias is None else linear.bias.to(dtype)
        return F.linear(x.to(dtype), linear.weight.to(dtype), bias)

    linear.forward = forward


_EXPERT = re.compile(r"^(.*\.moe\.)(gate_proj|up_proj|down_proj)\.lora_([AB])\.weight$")


def _check_uniform_lora(metadata, num_experts):
    if (
        metadata.get("rank_pattern")
        or metadata.get("alpha_pattern")
        or metadata.get("use_dora")
        or metadata.get("bias", "none") != "none"
        or metadata.get("lora_bias")
        or metadata.get("modules_to_save")
    ):
        raise ValueError("Step5 export requires uniform ordinary LoRA")
    if num_experts <= 0 or metadata["r"] <= 0:
        raise ValueError("Invalid expert count or LoRA rank")


def _per_expert(value, side, num_experts, rank, key):
    """[E, r, in] (A) or [E, out, r] (B) from the packed per-expert or the shared 2-D parameter."""
    packed = value.shape[0] if side == "A" else value.shape[1]
    if value.ndim != 2 or packed not in (rank, num_experts * rank):
        raise ValueError(f"Invalid packed-expert adapter shape: {key}")
    if packed == rank and num_experts != 1:
        return value.unsqueeze(0).expand(num_experts, *value.shape)
    if side == "A":
        return value.reshape(num_experts, rank, value.shape[-1])
    return value.reshape(value.shape[0], rank, num_experts).permute(2, 0, 1)


def _refuse_layer_selected(model):
    source_ids = getattr(getattr(model, "config", None), "_day0_source_layer_ids", None)
    if source_ids is not None and list(source_ids) != list(range(len(source_ids))):
        raise NotImplementedError(
            "adapter export of a model built from selected source layers: names would use local layer ids"
        )


def adapter_state_keys(model):
    """Inventory actual trainable adapter parameters, not PEFT's minimized targets."""
    _refuse_layer_selected(model)
    result = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if not name.endswith((".lora_A.default.weight", ".lora_B.default.weight")):
            raise ValueError(f"Unexpected trainable parameter {name}")
        result.add(name.removesuffix(".default.weight") + ".weight")
    if not result:
        raise ValueError("No trainable adapters to export")
    return result


def export_text_lora(tensors, peft_config, num_experts, expected_keys, dtype=None):
    """Preserve adapter values/scaling; only reshape expert A/B and their keys.

    dtype: serving dtype the FP32 master adapters are rounded to (round to nearest, as the receiver would)."""
    if dtype is not None:
        tensors = {name: value.to(dtype) for name, value in tensors.items()}
    metadata = copy.deepcopy(peft_config.to_dict())
    rank = metadata["r"]
    _check_uniform_lora(metadata, num_experts)
    expected_keys = set(expected_keys)
    if not expected_keys or set(tensors) != expected_keys:
        raise ValueError(f"Adapter inventory mismatch: {sorted(set(tensors) ^ expected_keys)}")
    result = {}
    groups = {}
    for name, value in tensors.items():
        match = _EXPERT.fullmatch(name)
        if match is None:
            result[name] = value
            continue
        prefix, projection, side = match.groups()
        groups.setdefault((prefix, projection), {})[side] = value
    for (prefix, projection), weights in groups.items():
        if weights.keys() != {"A", "B"}:
            raise ValueError(f"Incomplete expert LoRA for {prefix}{projection}")
        a = _per_expert(weights["A"], "A", num_experts, rank, prefix + projection)
        b = _per_expert(weights["B"], "B", num_experts, rank, prefix + projection)
        for expert in range(num_experts):
            for side, value in (("A", a[expert]), ("B", b[expert])):
                # Slices of an expanded shared factor alias one storage; a file writer needs separate tensors.
                shared = (a if side == "A" else b).stride(0) == 0
                name = f"{prefix}experts.{expert}.{projection}.lora_{side}.weight"
                if name in result:
                    raise ValueError(f"Duplicate exported adapter {name}")
                result[name] = value.clone() if shared else value.contiguous()
    metadata["target_parameters"] = None
    metadata["target_modules"] = sorted(
        {name.rsplit(".lora_", 1)[0].removeprefix("base_model.model.") for name in result}
    )
    return result, metadata


def stream_text_lora(model, peft_config, num_experts, dtype):
    """Same names, values and metadata as export_text_lora, one tensor at a time.

    Returns (generator of (name, tensor in `dtype` on the parameter's device), metadata). Adapters are
    all-gathered in groups, rounded on the device and released group by group.
    Every rank must consume the generator in step: the gathers are collective."""
    _refuse_layer_selected(model)
    metadata = copy.deepcopy(peft_config.to_dict())
    rank = metadata["r"]
    _check_uniform_lora(metadata, num_experts)
    entries = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if not name.endswith((".lora_A.default.weight", ".lora_B.default.weight")):
            raise ValueError(f"Unexpected trainable parameter {name}")
        entries.append((name.removesuffix(".default.weight") + ".weight", parameter))
    if not entries:
        raise ValueError("No trainable adapters to export")
    modules, sides = set(), {}
    for key, parameter in entries:
        match = _EXPERT.fullmatch(key)
        if match is None:
            modules.add(key.rsplit(".lora_", 1)[0])
            continue
        prefix, projection, side = match.groups()
        packed = parameter.shape[0] if side == "A" else parameter.shape[1]
        if parameter.ndim != 2 or packed not in (rank, num_experts * rank):
            raise ValueError(f"Invalid packed-expert adapter shape: {key}")
        sides.setdefault((prefix, projection), set()).add(side)
        modules.update(f"{prefix}experts.{expert}.{projection}" for expert in range(num_experts))
    for (prefix, projection), found in sides.items():
        if found != {"A", "B"}:
            raise ValueError(f"Incomplete expert LoRA for {prefix}{projection}")
    metadata["target_parameters"] = None
    metadata["target_modules"] = sorted(name.removeprefix("base_model.model.") for name in modules)

    def tensors():
        # Grouped all-gathers (one collective per group of adapters, not one per tensor); tensors stay on the
        # device until the sender copies them into its bucket.
        for key, value in iter_full_tensors(dict(entries)):
            value = value.detach().to(dtype)
            match = _EXPERT.fullmatch(key)
            if match is None:
                yield key, value
                continue
            prefix, projection, side = match.groups()
            value = _per_expert(value, side, num_experts, rank, key)
            shared = value.stride(0) == 0  # expanded shared factor: every expert slice aliases one storage
            for expert in range(num_experts):
                piece = value[expert]
                # Separate storage per expert: a file writer rejects aliased tensors.
                yield (
                    f"{prefix}experts.{expert}.{projection}.lora_{side}.weight",
                    piece.clone() if shared else piece.contiguous(),
                )
            del value, piece  # drop the gathered block before the next group is allocated

    return tensors(), metadata


def _evenly_dim_sharded(tensor) -> bool:
    """True for a device DTensor on a 1-D mesh whose single ``Shard`` dim splits evenly over the ranks.

    Depends on the global shape, mesh and placement and on the shard's device type, so ranks decide the same
    as long as they keep their shards on the same kind of device. CPU shards (CPUOffloadPolicy) are excluded:
    the mesh group may be NCCL-only.
    """
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        return False
    if not isinstance(tensor, DTensor) or tensor.device.type == "cpu":
        return False
    mesh, placements = tensor.device_mesh, tensor.placements
    if mesh.ndim != 1 or len(placements) != 1 or not placements[0].is_shard():
        return False
    return tensor.shape[placements[0].dim] % mesh.size() == 0


def iter_full_tensors(named, max_gathered_bytes: int = 1 << 28):
    """Yield ``(name, full tensor)`` on the local device for FSDP2 parameters.

    Evenly sharded ``DTensor``s are all-gathered in groups: the local shards of a
    group are concatenated, gathered with one collective, and split back. A
    per-expert MoE LoRA has tens of thousands of small adapter tensors, and one
    collective plus one host copy per tensor dominated the export (and kept a
    full host copy of the adapter on every rank). Anything else is gathered with
    ``full_tensor()``; plain tensors pass through.

    The gathers are collectives: every rank must consume the generator in the
    same order.
    """
    group: list[tuple[str, torch.Tensor]] = []
    group_bytes = 0

    def flush():
        nonlocal group, group_bytes
        if not group:
            return
        mesh = group[0][1].device_mesh
        size = mesh.size()
        local = [tensor.to_local().detach() for _, tensor in group]
        flat = torch.cat([shard.reshape(-1) for shard in local])
        gathered = torch.empty((size, flat.numel()), dtype=flat.dtype, device=flat.device)
        dist.all_gather_into_tensor(gathered, flat, group=mesh.get_group())
        del flat
        offset = 0
        for (name, tensor), shard in zip(group, local, strict=True):
            # [rank, *local shape] -> ranks concatenated along the shard dim.
            stacked = gathered[:, offset : offset + shard.numel()].reshape(size, *shard.shape)
            offset += shard.numel()
            yield name, stacked.movedim(0, tensor.placements[0].dim).reshape(tensor.shape)
        group, group_bytes = [], 0

    for name, tensor in named.items():
        if not _evenly_dim_sharded(tensor):
            yield from flush()
            yield name, (tensor.full_tensor() if hasattr(tensor, "full_tensor") else tensor).detach()
            continue
        nbytes = tensor.numel() * tensor.element_size()
        if group and (
            group_bytes + nbytes > max_gathered_bytes
            or tensor.dtype != group[0][1].dtype
            or tensor.device_mesh != group[0][1].device_mesh
        ):
            yield from flush()
        group.append((name, tensor))
        group_bytes += nbytes
    yield from flush()
