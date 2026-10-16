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
"""Text-only trainable Step5 causal LM, following the vLLM Step5 model implementation.

The config class comes from the installed vLLM (transformers has no step5 type). Plain PyTorch only: no
inference operators, KV cache or tensor parallelism. Weights are loaded by loading.py.
"""

import copy
import logging
import math
import os

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForCausalLM, GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from vllm.transformers_utils.configs.step5 import Step5Config


class Step5RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, dtype=torch.float32):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.eps = eps

    def forward(self, x, *, out_dtype=None):
        value = x.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)
        return (value * (self.weight.float() + 1)).to(out_dtype or x.dtype)


class Step5IndexerLayerNorm(nn.Module):
    def __init__(self, head_dim, eps=1e-6, dtype=torch.float32):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(head_dim, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(head_dim, dtype=dtype))
        self.head_dim = head_dim
        self.eps = eps

    def forward(self, x):
        heads = x.unflatten(-1, (-1, self.head_dim))
        return (
            F.layer_norm(heads.float(), (self.head_dim,), self.weight.float(), self.bias.float(), self.eps)
            .flatten(-2)
            .to(x.dtype)
        )


def step5_swiglu(gate, up, limit=None):
    gate = F.silu(gate)
    if limit not in (None, 0):
        # The gate clamp is AFTER SiLU; it is not a clamp on gate logits.
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    return gate * up


def fp8_act_fake_quant(x, group=128):
    """What an FP8 expert GEMM reads in serving: each 128-value group of a row is scaled by a power of two
    (UE8M0, smallest 2^k >= amax/448), rounded to E4M3 and scaled back. Straight-through gradient."""
    if x.shape[-1] % group:
        raise ValueError("FP8 activation groups need a width divisible by 128")
    value = x.detach().float().unflatten(-1, (-1, group))
    scale = (value.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10) / 448.0).clamp_min(1e-10)
    scale = torch.exp2(torch.ceil(torch.log2(scale)))
    quantized = (value / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * scale
    return x + (quantized.flatten(-2).to(x.dtype) - x.detach())


def _routed_swiglu(gate, up, limit):
    """The serving MoE kernel reads the stored gate/up values, computes the activation in FP32 and rounds once."""
    return step5_swiglu(gate.float(), up.float(), limit).to(gate.dtype)


class Step5MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, limit=None, dtype=None):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=dtype)
        self.limit = limit

    def forward(self, x):
        return self.down_proj(step5_swiglu(self.gate_proj(x), self.up_proj(x), self.limit))


class Step5PackedProjection(nn.Module):
    """Keep canonical checkpoint [expert, out, in] storage, without token-wise copies."""

    def __init__(self, experts, out_features, in_features, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(experts, out_features, in_features, dtype=dtype))
        nn.init.uniform_(self.weight, -1 / math.sqrt(in_features), 1 / math.sqrt(in_features))

    def forward(self, x, expert, lora=None, base_x=None):
        """lora: (lhs [E, out, r], rhs [E, r, in], scaling); adds scaling * (x rhs^T) lhs^T per expert, unfolded.
        A 2-D lhs [out, r] or rhs [r, in] is one factor shared by all experts.
        base_x: what the base weight reads when it differs from the adapter input (FP8 activation quantization)."""
        if base_x is None:
            base_x = x
        if isinstance(expert, int):
            output = F.linear(base_x, self.weight[expert])
            if lora is not None:
                lhs, rhs, scaling = lora
                output = output + F.linear(F.linear(x, _of(rhs, expert)), _of(lhs, expert)) * scaling
            return output
        if expert.ndim != 1 or expert.shape[0] != x.shape[0]:
            raise ValueError("One expert ID is required per routed token")
        if x.is_cuda:
            return _grouped_projection(x, expert, self.weight, lora, base_x)
        weight = self.weight
        output = x.new_zeros((x.shape[0], weight.shape[1]))
        for index in range(weight.shape[0]):
            token = torch.where(expert == index)[0]
            if token.numel():
                value = F.linear(base_x[token], weight[index])
                if lora is not None:
                    lhs, rhs, scaling = lora
                    value = value + F.linear(F.linear(x[token], _of(rhs, index)), _of(lhs, index)) * scaling
                output = output.index_copy(0, token, value)
        return output


def _of(factor, expert):
    """One expert's matrix from a per-expert stack; a shared 2-D factor is that matrix for every expert."""
    return factor if factor.ndim == 2 else factor[expert]


_GROUP_ALIGN = 16  # rows per expert group padded for the grouped-GEMM kernels


def _grouped_projection(x, expert, weight, lora, base_x=None):
    """Token rows sorted by expert into padded groups, one grouped GEMM per factor, then restored to token order."""
    experts = weight.shape[0]
    order = torch.argsort(expert, stable=True)
    counts = torch.bincount(expert, minlength=experts)
    padded = (counts + _GROUP_ALIGN - 1) // _GROUP_ALIGN * _GROUP_ALIGN
    ends = torch.cumsum(padded, 0)
    rank_in_group = torch.arange(x.shape[0], device=x.device) - (torch.cumsum(counts, 0) - counts)[expert[order]]
    row = (ends - padded)[expert[order]] + rank_in_group
    total = int(ends[-1])
    offsets = ends.to(torch.int32)
    grouped = x.new_zeros((total, x.shape[1])).index_copy(0, row, x[order])
    base_grouped = (
        grouped if base_x is None or base_x is x else grouped.new_zeros(grouped.shape).index_copy(0, row, base_x[order])
    )
    value = torch._grouped_mm(base_grouped, weight.transpose(-2, -1), offs=offsets)
    output = x.new_empty((x.shape[0], weight.shape[1])).index_copy(0, order, value[row])
    if lora is not None:
        lhs, rhs, scaling = lora
        rank = rhs.shape[-2]
        if rank * rhs.element_size() % 16 == 0:
            # A shared factor is an ordinary GEMM in token order; a per-expert one is a grouped GEMM on the padded
            # rows (the kernel needs dense per-expert matrices, not permuted views of the packed parameter).
            if rhs.ndim == 2:
                low = F.linear(x, rhs)
            else:
                low = torch._grouped_mm(grouped, rhs.contiguous().transpose(-2, -1), offs=offsets)
                if lhs.ndim == 2:
                    low = x.new_empty((x.shape[0], rank)).index_copy(0, order, low[row])
            if lhs.ndim == 2:
                delta = F.linear(low, lhs)
            else:
                if rhs.ndim == 2:
                    low = low.new_zeros((total, rank)).index_copy(0, row, low[order])
                delta = torch._grouped_mm(low, lhs.contiguous().transpose(-2, -1), offs=offsets)
                delta = x.new_empty(output.shape).index_copy(0, order, delta[row])
        else:  # rank too small for the kernel's alignment: per-token factors, fine at test sizes
            low = F.linear(x, rhs) if rhs.ndim == 2 else torch.einsum("ti,tri->tr", x, rhs[expert])
            delta = F.linear(low, lhs) if lhs.ndim == 2 else torch.einsum("tr,tor->to", low, lhs[expert])
        output = output + delta * scaling
    return output


class Step5MoE(nn.Module):
    def __init__(
        self,
        hidden_size,
        intermediate_size,
        shared_size,
        experts,
        top_k,
        *,
        renormalize=True,
        routed_scale=1.0,
        expert_limit=None,
        shared_limit=None,
        dtype=None,
    ):
        super().__init__()
        if not 1 <= top_k <= experts:
            raise ValueError("top_k must be in [1, experts]")
        self.gate = nn.Linear(hidden_size, experts, bias=False, dtype=torch.float32)
        self.router_bias = nn.Parameter(torch.zeros(experts, dtype=torch.float32), requires_grad=False)
        self.gate_proj = Step5PackedProjection(experts, intermediate_size, hidden_size, dtype)
        self.up_proj = Step5PackedProjection(experts, intermediate_size, hidden_size, dtype)
        self.down_proj = Step5PackedProjection(experts, hidden_size, intermediate_size, dtype)
        self.share_expert = Step5MLP(hidden_size, shared_size, shared_limit, dtype)
        self.top_k, self.renormalize, self.routed_scale = top_k, renormalize, routed_scale
        self.expert_limit = expert_limit
        # Set by the checkpoint loader for FP8-stored routed experts (VERL_STEP5_FP8_ACT_QUANT=0 turns it off).
        self.fp8_act_quant = False
        # Set to a zero tensor [recorded tokens, tokens whose native top-k set differs, recorded experts the
        # native router did not pick] to count how often replay overrides this router; None costs nothing.
        self.replay_stats = None

    def routing(self, x, replay=None):
        scores = torch.sigmoid(self.gate(x.float()))
        indices = (scores + self.router_bias).topk(self.top_k, dim=-1).indices
        if replay is not None:
            # R3: tokens with a recorded rollout choice use those experts (weights still come from this
            # router); all-zero rows carry no record and are routed natively. top_k distinct experts are
            # never all zero, so the convention needs top_k > 1.
            recorded = replay.any(-1)
            if self.top_k == 1 or replay.dtype.is_floating_point:
                raise ValueError("router replay needs integer expert ids and top_k > 1")
            if bool(recorded.any()):
                rows = replay[recorded].sort(-1).values
                if (
                    int(rows.min()) < 0
                    or int(rows.max()) >= scores.shape[-1]
                    or bool((rows[:, 1:] == rows[:, :-1]).any())
                ):
                    raise ValueError("recorded routing must hold distinct expert ids of this layer")
                if self.replay_stats is not None:
                    native = indices[recorded].sort(-1).values
                    self.replay_stats += torch.stack(
                        [
                            recorded.sum(),
                            (native != rows).any(-1).sum(),
                            rows.numel() - (native[:, :, None] == rows[:, None, :]).any(1).sum(),
                        ]
                    )
            indices = torch.where(recorded[:, None], replay.to(indices.dtype), indices)
        weights = scores.gather(-1, indices)  # Unbiased scores, not correction-biased values.
        if self.renormalize:
            weights = weights / weights.sum(-1, keepdim=True)
        return indices, weights * self.routed_scale

    def forward_routed(self, x, replay=None):
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        indices, weights = self.routing(flat, None if replay is None else replay.reshape(-1, self.top_k))
        token = torch.arange(flat.shape[0], device=x.device).repeat_interleave(self.top_k)
        expert = indices.flatten()
        selected = flat[token]
        # Each projection is called once: PEFT parameter adapters must not rebuild
        # the whole expert stack once per expert. Never make token-wise weight copies.
        if self.fp8_act_quant:
            # Serving quantizes the token once before routing it to its experts; adapters read the raw values.
            quantized = fp8_act_fake_quant(flat)[token]
            intermediate = _routed_swiglu(
                self.gate_proj(selected, expert, base_x=quantized),
                self.up_proj(selected, expert, base_x=quantized),
                self.expert_limit,
            )
            value = self.down_proj(intermediate, expert, base_x=fp8_act_fake_quant(intermediate))
        else:
            intermediate = step5_swiglu(
                self.gate_proj(selected, expert), self.up_proj(selected, expert), self.expert_limit
            )
            value = self.down_proj(intermediate, expert)
        output = (value.float().reshape(flat.shape[0], self.top_k, shape[-1]) * weights.unsqueeze(-1)).sum(dim=1)
        return output.to(x.dtype).reshape(shape)

    def forward(self, x, *, routed_only=False, replay=None):
        routed = self.forward_routed(x, replay)
        return routed if routed_only else (self.share_expert(x), routed)


def neox_rope(x, positions, rotary_dim, theta, scaling=None):
    if rotary_dim <= 0 or rotary_dim % 2 or rotary_dim > x.shape[-1]:
        raise ValueError("invalid rotary_dim")
    frequency = 1 / theta ** (torch.arange(0, rotary_dim, 2, device=x.device).float() / rotary_dim)
    rope_type = scaling.get("rope_type", "default") if scaling else "default"
    if rope_type not in ("default", "llama3"):
        raise NotImplementedError("only default and llama3 RoPE are implemented")
    if rope_type == "llama3":
        factor = scaling["factor"]
        original = scaling["original_max_position_embeddings"]
        low, high = scaling["low_freq_factor"], scaling["high_freq_factor"]
        wavelength = 2 * math.pi / frequency
        smooth = (original / wavelength - low) / (high - low)
        scaled = frequency / factor
        medium = (1 - smooth) * scaled + smooth * frequency
        frequency = torch.where(wavelength > original / low, scaled, frequency)
        frequency = torch.where((wavelength >= original / high) & (wavelength <= original / low), medium, frequency)
    phase = positions.float().unsqueeze(-1) * frequency
    # Match the model-dtype cosine/sine cache consumed by inference kernels.
    cos = phase.cos().to(x.dtype).float().unsqueeze(-2)
    sin = phase.sin().to(x.dtype).float().unsqueeze(-2)
    first, second = x[..., :rotary_dim].float().chunk(2, -1)
    rotated = torch.cat((first * cos - second * sin, second * cos + first * sin), -1)
    return torch.cat((rotated.to(x.dtype), x[..., rotary_dim:]), -1)


def causal_gqa(q, k, v, *, valid=None, window=None, gate=None):
    """[batch, tokens, heads, dim], with explicit per-sample padding mask."""
    batch, tokens, heads, _ = q.shape
    if k.shape != v.shape or k.shape[:2] != (batch, tokens) or heads % k.shape[2]:
        raise ValueError("incompatible GQA shapes")
    if valid is None:
        position = torch.arange(tokens, device=q.device).expand(batch, -1)
    else:
        # Padding is not a position in the logical sequence, including internal holes.
        position = valid.long().cumsum(-1) - 1
    distance = position[:, :, None] - position[:, None, :]
    allowed = distance >= 0
    if window is not None:
        if window < 1:
            raise ValueError("window must be positive")
        allowed = allowed & (distance < window)
    allowed = allowed[:, None]
    if valid is not None:
        allowed = allowed & valid[:, None, :, None].bool() & valid[:, None, None, :].bool()
    output = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=allowed,
        dropout_p=0.0,
        enable_gqa=True,
    ).transpose(1, 2)
    if gate is not None:
        output = (output.float() * gate.float().sigmoid().unsqueeze(-1)).to(q.dtype)
    return output


def csa_region_summaries(key, z):
    """Unpadded [tokens, providers, proxy_dim]; softmax over eight-token regions."""
    if key.shape != z.shape or key.ndim != 3 or key.shape[0] < 1:
        raise ValueError("CSA requires matching nonempty key/z tensors")
    tokens, groups, dim = key.shape
    regions = (tokens + 7) // 8
    padding = regions * 8 - tokens
    key = F.pad(key.float(), (0, 0, 0, 0, 0, padding)).reshape(regions, 8, groups, dim)
    logits = F.pad(z.float(), (0, 0, 0, 0, 0, padding), value=float("-inf")).reshape(regions, 8, groups, dim)
    mean = (logits.softmax(dim=1) * key).sum(dim=1)
    # Inference stores BF16-rounded CSA means as unscaled E4M3.
    return mean.bfloat16().to(torch.float8_e4m3fn).float()


def dsa_region_mask(query, key, z, weights, topk):
    """Unpadded Q [T,G,H,D], shared K/Z [T,1,D], W [T,G,H] → [G,T,T].

    Query must already be E4M3-rounded, as in indexer_norm_rope. The caller
    must compact each sample's valid tokens independently before this function;
    physical padding indices are not logical eight-token region indices.
    """
    tokens, groups, heads, dim = query.shape
    if topk < 1 or key.shape != (tokens, 1, dim) or weights.shape != (tokens, groups, heads):
        raise ValueError("invalid DSA inputs")
    summaries = csa_region_summaries(key, z).expand(-1, groups, -1)
    regions = summaries.shape[0]
    position = torch.arange(tokens, device=query.device)
    current = position // 8
    region = torch.arange(regions, device=query.device)
    historical = region[None, :] < current[:, None]
    if regions <= topk:
        selected = historical[None].expand(groups, -1, -1)
    else:
        dots = torch.einsum("tghd,rgd->gthr", query.float(), summaries)
        scores = (dots.relu() * weights.permute(1, 0, 2).unsqueeze(-1).float() / math.sqrt(heads)).sum(dim=2)
        scores = scores.masked_fill(~historical[None], float("-inf"))
        # Native selector resolves equal-score boundaries by region index.
        indices = scores.argsort(dim=-1, descending=True, stable=True)[..., :topk]
        selected = torch.zeros_like(scores, dtype=torch.bool).scatter(-1, indices, True)
        selected = selected & historical[None]
    selected = selected | (region[None, None, :] == current[None, :, None])
    token_mask = selected.index_select(-1, current)
    return token_mask & (position[None, None, :] <= position[None, :, None])


def sparse_gqa(q, k, v, index_q, index_k, index_z, index_weights, *, topk, valid=None, gate=None):
    """Differentiable main attention, discrete CSA selector, independent samples."""
    batch, tokens, heads, dim = q.shape
    groups = k.shape[2]
    if heads % groups or k.shape != v.shape or k.shape != (batch, tokens, groups, dim):
        raise ValueError("incompatible sparse GQA shapes")
    if valid is None:
        valid = torch.ones((batch, tokens), dtype=torch.bool, device=q.device)
    outputs = []
    for sample in range(batch):
        ids = valid[sample].bool().nonzero().flatten()
        if ids.numel() == 0:
            outputs.append(torch.zeros_like(q[sample]))
            continue
        mask = dsa_region_mask(
            index_q[sample, ids], index_k[sample, ids], index_z[sample, ids], index_weights[sample, ids], topk
        )
        query = q[sample, ids].reshape(-1, groups, heads // groups, dim).permute(1, 2, 0, 3)
        key = k[sample, ids].permute(1, 0, 2).unsqueeze(1)
        value = v[sample, ids].permute(1, 0, 2).unsqueeze(1)
        attended = F.scaled_dot_product_attention(
            query, key, value, attn_mask=mask[:, None], dropout_p=0.0, enable_gqa=True
        )
        attended = attended.permute(2, 0, 1, 3).reshape(-1, heads, dim)
        if gate is not None:
            attended = (attended.float() * gate[sample, ids].float().sigmoid().unsqueeze(-1)).to(q.dtype)
        outputs.append(torch.zeros_like(q[sample]).index_copy(0, ids, attended))
    return torch.stack(outputs)


logger = logging.getLogger(__name__)


def _per_layer(values, layer, default=None):
    if not values:
        return default
    if layer >= len(values):
        raise ValueError(f"Per-layer config has no entry for layer {layer}")
    return values[layer]


def _dtype(config):
    dtype = getattr(config, "dtype", None) or torch.bfloat16
    return getattr(torch, dtype.replace("torch.", "")) if isinstance(dtype, str) else dtype


def _moe_layers(config):
    groups = []
    for attr in ("moe_layers_enum", "moe_layer_list"):
        value = getattr(config, attr, None)
        if value is not None:
            if isinstance(value, str):
                value = [int(x.strip()) for x in value.split(",") if x.strip()]
            groups.append(set(value))
    if len(groups) == 2 and groups[0] != groups[1]:
        raise ValueError("Contradictory MoE layer selections")
    return (
        groups[0]
        if groups
        else set(range(1, getattr(config, "_day0_original_num_hidden_layers", config.num_hidden_layers)))
    )


def training_text_config(source, source_ids=None):
    values = copy.deepcopy(source)
    if source_ids is not None:
        original_layers = values["num_hidden_layers"]
        if (
            not source_ids
            or len(set(source_ids)) != len(source_ids)
            or any(not 0 <= i < original_layers for i in source_ids)
        ):
            raise ValueError("Select unique target source layers, not MTP layers")
        kinds = values.get("layer_types_with_mtp") or values.get("layer_types")
        values.pop("layer_types_with_mtp", None)
        values["_day0_source_layer_ids"] = list(source_ids)
        values["_day0_source_layer_types"] = kinds
        values["_day0_original_num_hidden_layers"] = original_layers
        values["num_hidden_layers"] = len(source_ids)
        values["num_nextn_predict_layers"] = 0
        values["layer_types"] = [kinds[i] for i in source_ids] if kinds else None
    return Step5Config(**values)


class Step5Attention(nn.Module):
    def __init__(self, config, layer):
        super().__init__()
        if config.att_impl_type != "GQA" or getattr(config, "attention_bias", False):
            raise NotImplementedError("Step5 training reference requires bias-free GQA")
        kind = _per_layer(
            getattr(config, "_day0_source_layer_types", config.layer_types),
            layer,
            "sliding_attention" if layer % 2 == 0 else "full_attention",
        )
        self.window = getattr(config, "sliding_window", None) if kind == "sliding_attention" else None
        other = config.attention_other_setting or {}
        selected = other if other.get("attention_type") == kind else {}
        self.heads = selected.get("num_attention_heads", config.num_attention_heads)
        self.groups = selected.get("num_attention_groups", config.num_attention_groups)
        self.head_dim = selected.get("head_dim", config.head_dim or config.hidden_size // self.heads)
        if self.heads % self.groups:
            raise ValueError("GQA heads must divide into KV groups")
        dtype, hidden = _dtype(config), config.hidden_size
        self.q_proj = nn.Linear(hidden, self.heads * self.head_dim, bias=False, dtype=dtype)
        self.k_proj = nn.Linear(hidden, self.groups * self.head_dim, bias=False, dtype=dtype)
        self.v_proj = nn.Linear(hidden, self.groups * self.head_dim, bias=False, dtype=dtype)
        self.o_proj = nn.Linear(self.heads * self.head_dim, hidden, bias=False, dtype=dtype)
        self.q_norm = Step5RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = Step5RMSNorm(self.head_dim, config.rms_norm_eps)
        self.g_proj = nn.Linear(hidden, self.heads, bias=False, dtype=dtype) if config.use_head_wise_attn_gate else None
        self.rotary_dim = int(self.head_dim * _per_layer(config.partial_rotary_factors, layer, 1.0))
        self.theta = _per_layer(config.rope_theta, layer) if isinstance(config.rope_theta, list) else config.rope_theta
        self.scaling = config.rope_scaling if not config.yarn_only_types or kind in config.yarn_only_types else None
        self.use_rope = _per_layer(config.use_rope_layers, layer, True)
        sparse = getattr(config, "sparse_config", None)
        self.sparse = sparse if sparse and sparse.get("enabled", True) and kind == "full_attention" else None
        if self.sparse:
            proxy = sparse.get("proxy_dim", 256)
            self.index_heads = sparse.get("sparse_indexer_num_heads", 16)
            providers = sparse.get("num_provider_groups", sparse.get("index_tp_size", 4))
            if providers != self.groups or self.index_heads % providers:
                raise ValueError("Indexer provider groups must match main KV groups")
            self.proxy_dim = proxy
            self.index_rotary_dim = sparse.get("sparse_indexer_rope_dim", 32)
            self.topk = sparse.get("topk", 512)
            self.sparse_indexer_q = nn.Linear(hidden, self.index_heads * proxy, bias=False, dtype=dtype)
            self.sparse_indexer_k = nn.Linear(hidden, proxy, bias=False, dtype=dtype)
            self.sparse_indexer_z = nn.Linear(hidden, proxy, bias=False, dtype=dtype)
            self.sparse_indexer_w = nn.Linear(hidden, self.index_heads, bias=False, dtype=dtype)
            self.sparse_indexer_q_norm = Step5RMSNorm(proxy, config.rms_norm_eps)
            self.sparse_indexer_k_norm = Step5IndexerLayerNorm(proxy, config.rms_norm_eps)
            self.ssmax_s = nn.Parameter(torch.zeros(self.heads), requires_grad=False)

    def forward(self, x, positions, valid):
        batch, tokens, _ = x.shape
        q = self.q_norm(self.q_proj(x).reshape(batch, tokens, self.heads, self.head_dim))
        k = self.k_norm(self.k_proj(x).reshape(batch, tokens, self.groups, self.head_dim))
        v = self.v_proj(x).reshape(batch, tokens, self.groups, self.head_dim)
        if self.use_rope:
            q = neox_rope(q, positions, self.rotary_dim, self.theta, self.scaling)
            k = neox_rope(k, positions, self.rotary_dim, self.theta, self.scaling)
        gate = self.g_proj(x) if self.g_proj is not None else None
        if self.sparse:
            iq = self.sparse_indexer_q_norm(
                self.sparse_indexer_q(x).reshape(batch, tokens, self.index_heads, self.proxy_dim)
            )
            ik = self.sparse_indexer_k_norm(self.sparse_indexer_k(x)).reshape(batch, tokens, 1, self.proxy_dim)
            iq = neox_rope(iq, positions, self.index_rotary_dim, self.theta, self.scaling)
            ik = neox_rope(ik, positions, self.index_rotary_dim, self.theta, self.scaling)
            iq = (
                iq.to(torch.float8_e4m3fn)
                .float()
                .reshape(batch, tokens, self.groups, self.index_heads // self.groups, self.proxy_dim)
            )
            iz = self.sparse_indexer_z(x).reshape(batch, tokens, 1, self.proxy_dim)
            iw = self.sparse_indexer_w(x).reshape(batch, tokens, self.groups, self.index_heads // self.groups)
            out = sparse_gqa(q, k, v, iq, ik, iz, iw, topk=self.topk, valid=valid, gate=gate)
        else:
            out = causal_gqa(q, k, v, valid=valid, window=self.window, gate=gate)
        return self.o_proj(out.reshape(batch, tokens, -1))


class Step5DecoderLayer(nn.Module):
    def __init__(self, config, layer):
        super().__init__()
        if not config.zero_centered:
            raise NotImplementedError("Step5 requires zero-centered RMSNorm")
        self.fp32_residual = config.fp32_residual_connection
        self.dtype = _dtype(config)
        self.input_layernorm = Step5RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Step5RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Step5Attention(config, layer)
        if layer in _moe_layers(config):
            if not config.need_fp32_gate or not config.use_moe_router_bias or config.moe_router_activation != "sigmoid":
                raise NotImplementedError("Step5 requires FP32 sigmoid routing with correction bias")
            self.moe = Step5MoE(
                config.hidden_size,
                config.moe_intermediate_size,
                config.share_expert_dim,
                config.moe_num_experts,
                config.moe_top_k,
                renormalize=config.norm_expert_weight,
                routed_scale=config.moe_router_scaling_factor,
                expert_limit=_per_layer(config.swiglu_limits, layer),
                shared_limit=_per_layer(config.swiglu_limits_shared, layer),
                dtype=self.dtype,
            )
            # Checkpoint ownership is layers.N.share_expert, not layers.N.moe.share_expert.
            self.share_expert = self.moe.share_expert
            del self.moe.share_expert
        else:
            self.mlp = Step5MLP(
                config.hidden_size, config.intermediate_size, _per_layer(config.swiglu_limits_shared, layer), self.dtype
            )

    def _add(self, x, y):
        return x.float() + y.float() if self.fp32_residual else x + y

    def forward(self, x, positions, valid, replay=None):
        x = self._add(x, self.self_attn(self.input_layernorm(x, out_dtype=self.dtype), positions, valid))
        normalized = self.post_attention_layernorm(x, out_dtype=self.dtype)
        if hasattr(self, "moe"):
            shared = self.share_expert(normalized)
            routed = self.moe(normalized, routed_only=True, replay=replay)
            output = self._add(shared, routed)
        else:
            output = self.mlp(normalized)
        return self._add(x, output)


class Step5TextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype=_dtype(config))
        source_ids = getattr(config, "_day0_source_layer_ids", list(range(config.num_hidden_layers)))
        if (
            len(source_ids) != config.num_hidden_layers
            or len(set(source_ids)) != len(source_ids)
            or any(i < 0 for i in source_ids)
        ):
            raise ValueError("Invalid source layer selection")
        self.layers = nn.ModuleList(Step5DecoderLayer(config, i) for i in source_ids)
        self.source_ids = list(source_ids)
        self.norm = Step5RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.gradient_checkpointing = False

    def forward(self, input_ids, positions, valid, inputs_embeds=None, routed_experts=None):
        x = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        if self.config.fp32_residual_connection:
            x = x.float()
        if routed_experts is not None:
            # The rollout reports [..., layer, top_k] with one row per layer of the source model, indexed by
            # layer id; layers without routed experts stay zero.
            # Source layers this model does not build may carry records; a built dense layer may not.
            moe_ids = [i for i, layer in zip(self.source_ids, self.layers, strict=True) if hasattr(layer, "moe")]
            if not moe_ids:
                routed_experts = None
            elif routed_experts.shape[-1] != self.config.moe_top_k or routed_experts.shape[-2] <= max(moe_ids):
                raise ValueError(
                    f"routed_experts must end in [layers > {max(moe_ids)}, {self.config.moe_top_k}], "
                    f"got {tuple(routed_experts.shape)}"
                )
            else:
                dense_ids = [
                    i
                    for i, layer in zip(self.source_ids, self.layers, strict=True)
                    if not hasattr(layer, "moe") and i < routed_experts.shape[-2]
                ]
                if dense_ids and bool(routed_experts[..., dense_ids, :].any()):
                    raise ValueError("routed_experts has recorded routing for a layer without routed experts")
        count = os.environ.get("VERL_STEP5_ROUTING_AGREEMENT", "0") != "0" and routed_experts is not None
        for layer in self.layers:
            if hasattr(layer, "moe"):
                layer.moe.replay_stats = torch.zeros(3, dtype=torch.long, device=x.device) if count else None
        for source_id, layer in zip(self.source_ids, self.layers, strict=True):
            replay = None
            if routed_experts is not None and hasattr(layer, "moe"):
                replay = routed_experts[..., source_id, :]
            if self.gradient_checkpointing and self.training:
                x = self._gradient_checkpointing_func(layer, x, positions, valid, replay)
            else:
                x = layer(x, positions, valid, replay)
        if count and not torch.is_grad_enabled():
            # One line per scoring forward: how often the recorded routing differs from this router.
            stats = torch.stack([layer.moe.replay_stats for layer in self.layers if hasattr(layer, "moe")]).sum(0)
            logger.info(
                "R3 routing: recorded=%d set_differs=%d experts_not_native=%d top_k=%d",
                int(stats[0]),
                int(stats[1]),
                int(stats[2]),
                self.config.moe_top_k,
            )
        # Native compute_logits rounds FP32 residual before the final norm.
        return self.norm(x.to(_dtype(self.config)))


def _restarts_within_row(position_ids, valid):
    """True when the positions of a row's valid tokens are not strictly increasing (a second sequence starts)."""
    seen = torch.cummax(position_ids.masked_fill(~valid, -1), dim=-1).values
    previous = F.pad(seen[..., :-1], (1, 0), value=-1)
    return bool((valid & (position_ids <= previous)).any())


class Step5ForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = Step5Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _no_split_modules = ["Step5DecoderLayer"]
    _keep_in_fp32_modules = [
        "gate",
        "router_bias",
        "input_layernorm",
        "post_attention_layernorm",
        "q_norm",
        "k_norm",
        "norm",
        "sparse_indexer_q_norm",
        "sparse_indexer_k_norm",
        "ssmax_s",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.model = Step5TextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=_dtype(config))
        self.post_init()
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        use_cache=False,
        return_dict=None,
        routed_experts=None,
        **kwargs,
    ):
        if use_cache or kwargs.get("past_key_values") is not None:
            raise NotImplementedError("Training reference uses full sequences without KV cache")
        if return_dict is None:
            return_dict = self.config.return_dict
        shape = input_ids.shape if inputs_embeds is None else inputs_embeds.shape[:2]
        device = input_ids.device if inputs_embeds is None else inputs_embeds.device
        valid = torch.ones(shape, device=device, dtype=torch.bool) if attention_mask is None else attention_mask.bool()
        if position_ids is not None and position_ids.shape != shape:
            raise NotImplementedError(
                f"position_ids must be [batch, tokens] {tuple(shape)}, got {tuple(position_ids.shape)}"
            )
        positions = (valid.long().cumsum(-1) - 1).clamp_min(0) if position_ids is None else position_ids
        packed = (
            attention_mask is None
            and position_ids is not None
            and inputs_embeds is None
            and labels is None
            and shape[0] == 1
            and int(position_ids[0, 0]) == 0
            and bool((position_ids[0, 1:] == 0).any())
        )
        if not packed and position_ids is not None and _restarts_within_row(position_ids, valid):
            # Attention masks are built from token order: a restart that is not unpacked would let one sequence
            # attend to the previous one.
            raise NotImplementedError(
                "position_ids restart within a row; packed sequences are only supported as input_ids and "
                "position_ids without attention_mask, inputs_embeds or labels"
            )
        if packed:
            # VERL remove-padding input: sequences concatenated into one row, every position 0 starts a
            # sequence. Run them as the padded batch the model was validated on and pack the result back.
            # (A pad suffix is one more sequence: correct, and only as cheap as it is short.)
            start = position_ids[0] == 0
            if bool(((position_ids[0, 1:] <= position_ids[0, :-1]) & ~start[1:]).any()):
                raise NotImplementedError(
                    "packed position_ids must increase strictly within each sequence (a new sequence starts at 0)"
                )
            row = start.long().cumsum(0) - 1
            first = torch.nonzero(start).squeeze(-1)
            column = torch.arange(shape[1], device=device) - first[row]
            size = (int(row[-1]) + 1, int(column.max()) + 1)

            def unpack(value):
                out = value.new_zeros(size + value.shape[2:])
                out[row, column] = value[0]
                return out

            valid = unpack(valid)
            hidden = self.model(
                unpack(input_ids),
                unpack(positions),
                valid,
                None,
                None if routed_experts is None else unpack(routed_experts),
            )
            hidden = hidden[row, column].unsqueeze(0)
        else:
            hidden = self.model(input_ids, positions, valid, inputs_embeds, routed_experts)
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            targets = labels[:, 1:].masked_fill(~(valid[:, :-1] & valid[:, 1:]), -100)
            loss = F.cross_entropy(
                logits[:, :-1].float().reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100
            )
        if not return_dict:
            return (loss, logits) if loss is not None else (logits,)
        return CausalLMOutputWithPast(loss=loss, logits=logits)


def register_step5_training():
    """Register only a training model for the image's existing config type."""
    AutoConfig.register("step5", Step5Config, exist_ok=True)
    AutoModelForCausalLM.register(Step5Config, Step5ForCausalLM, exist_ok=True)
