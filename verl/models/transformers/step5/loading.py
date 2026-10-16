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
"""Step5 FP8 checkpoint loading into FSDP2 shards, and the FSDP engine hooks for Step5."""

import copy
import json
import logging
import math
import os
import re
from pathlib import Path

import torch
from safetensors import safe_open
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor, Shard

from .lora import attach_text_lora
from .modeling_step5 import Step5ForCausalLM


def owned_text_names(mapping, layers):
    selected = set(range(layers)) if isinstance(layers, int) else set(layers)
    owned = set()
    for name in mapping:
        layer = re.match(r"^model\.layers\.(\d+)\.", name)
        if (layer and int(layer[1]) in selected) or name.startswith(("model.embed_tokens.", "model.norm.", "lm_head.")):
            owned.add(name)
    return owned


def source_parameter_name(name, config):
    name = name.removeprefix("base_model.model.").replace(".base_layer.", ".")
    ids = getattr(config, "_day0_source_layer_ids", None)
    if ids is not None:
        name = re.sub(r"^model\.layers\.(\d+)\.", lambda match: f"model.layers.{ids[int(match[1])]}.", name)
    return name


def mark_fp8_experts(model, name):
    """Routed experts stored in FP8 run with FP8 activations in serving; the trainer forward follows."""
    if os.environ.get("VERL_STEP5_FP8_ACT_QUANT", "1") == "0":
        return
    for projection in (".moe.gate_proj.", ".moe.up_proj.", ".moe.down_proj."):
        if projection in name:
            model.get_submodule(name[: name.index(projection)] + ".moe").fp8_act_quant = True


def dequantize_block_fp8(weight, scale, *, dtype=torch.bfloat16, block=(128, 128)):
    if weight.dtype != torch.float8_e4m3fn or weight.ndim not in (2, 3):
        raise ValueError("Expected an E4M3 matrix or expert matrix stack")
    rows, cols = weight.shape[-2:]
    expected = (*weight.shape[:-2], math.ceil(rows / block[0]), math.ceil(cols / block[1]))
    if tuple(scale.shape) != expected or scale.dtype != torch.float32:
        raise ValueError(f"Invalid block scales: {scale.shape}/{scale.dtype}, expected {expected}/float32")
    if not torch.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError("Block scales must be finite and positive")
    result = torch.empty_like(weight, dtype=dtype)
    weights = weight.reshape(-1, rows, cols)
    scales = scale.reshape(-1, *expected[-2:])
    outputs = result.reshape(-1, rows, cols)
    # Bound the FP32 temporary to one row block, not the full expert stack.
    for expert in range(weights.shape[0]):
        for row in range(0, rows, block[0]):
            expanded = scales[expert, row // block[0]].repeat_interleave(block[1])[:cols]
            outputs[expert, row : row + block[0]] = weights[expert, row : row + block[0]].float() * expanded
    return result


def load_text_checkpoint(model, directory):
    directory = Path(directory).resolve(strict=True)
    config = json.loads((directory / "config.json").read_text())
    quant = config.get("quantization_config", {})
    if quant.get("quant_method") != "fp8" or quant.get("fmt") != "e4m3" or quant.get("weight_block_size") != [128, 128]:
        raise ValueError("This loader requires the canonical E4M3 128x128 checkpoint")
    mapping = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]
    parameters = dict(model.named_parameters(remove_duplicate=False))
    missing = {source_parameter_name(name, model.config) for name in parameters} - mapping.keys()
    if missing:
        raise ValueError(f"Missing checkpoint parameters: {sorted(missing)}")
    if not all(parameter.is_meta for parameter in parameters.values()):
        raise ValueError("Load into a meta model before PEFT/FSDP/optimizer construction")

    def read(name):
        shard = (directory / mapping[name]).resolve(strict=True)
        if not shard.is_relative_to(directory):
            raise ValueError(f"Unsafe shard path for {name}")
        with safe_open(shard, framework="pt", device="cpu") as handle:
            return handle.get_tensor(name)

    loaded = []
    consumed = set()
    for name, parameter in parameters.items():
        source_name = source_parameter_name(name, model.config)
        weight = read(source_name)
        if weight.shape != parameter.shape:
            raise ValueError(f"{name}: checkpoint {weight.shape} != model {parameter.shape}")
        scale_name = source_name.removesuffix(".weight") + ".weight_scale_inv"
        if weight.dtype == torch.float8_e4m3fn:
            if not name.endswith(".weight") or scale_name not in mapping:
                raise ValueError(f"Missing FP8 block scales for {name}")
            weight = dequantize_block_fp8(weight, read(scale_name), dtype=parameter.dtype)
            consumed.add(scale_name)
            mark_fp8_experts(model, name)
        elif weight.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError(f"Unsupported checkpoint dtype for {name}: {weight.dtype}")
        elif scale_name in mapping:
            raise ValueError(f"Unexpected scale for non-FP8 parameter {name}")
        parent, _, leaf = name.rpartition(".")
        module = model.get_submodule(parent)
        device = "cpu" if parameter.is_meta else parameter.device
        value = weight.to(device=device, dtype=parameter.dtype)
        module._parameters[leaf] = torch.nn.Parameter(value, requires_grad=parameter.requires_grad)
        loaded.append(name)
        consumed.add(source_name)
    extras = (
        owned_text_names(mapping, getattr(model.config, "_day0_source_layer_ids", model.config.num_hidden_layers))
        - consumed
    )
    if extras:
        raise ValueError(f"Unconsumed text checkpoint tensors: {sorted(extras)}")
    if model.config.tie_word_embeddings:
        if not torch.equal(model.lm_head.weight, model.model.embed_tokens.weight):
            raise ValueError("Checkpoint contradicts tied embedding configuration")
        model.lm_head.weight = model.model.embed_tokens.weight
    return loaded


logger = logging.getLogger(__name__)


def read_row_shard(directory, mapping, name, start, end, dtype):
    directory = Path(directory).resolve(strict=True)

    def open_tensor(key):
        path = (directory / mapping[key]).resolve(strict=True)
        if not path.is_relative_to(directory):
            raise ValueError(f"Unsafe shard path for {key}")
        return safe_open(path, framework="pt", device="cpu")

    with open_tensor(name) as handle:
        tensor = handle.get_slice(name)
        shape = tensor.get_shape()
        if not 0 <= start <= end <= shape[0]:
            raise ValueError(f"Invalid row interval for {name}")
        source_dtype = tensor.get_dtype()
        if source_dtype in ("BF16", "F32"):
            return tensor[start:end].to(dtype), shape, None
        if source_dtype != "F8_E4M3" or len(shape) not in (2, 3):
            raise ValueError(f"Unsupported sharded dtype/shape: {name}/{source_dtype}/{shape}")
        scale_name = name.removesuffix(".weight") + ".weight_scale_inv"
        if not name.endswith(".weight") or scale_name not in mapping:
            raise ValueError(f"Missing scales for {name}")
        with open_tensor(scale_name) as scales_handle:
            scales = scales_handle.get_slice(scale_name)
            expected = shape[:-2] + [math.ceil(shape[-2] / 128), math.ceil(shape[-1] / 128)]
            if scales.get_shape() != expected or scales.get_dtype() != "F32":
                raise ValueError(f"Invalid scales for {name}")
            if start == end:
                value = torch.empty((0, *shape[1:]), dtype=dtype)
            elif len(shape) == 3:
                value = dequantize_block_fp8(tensor[start:end], scales[start:end], dtype=dtype)
            else:
                first_block, last_block = start // 128, math.ceil(end / 128)
                first_row = first_block * 128
                value = dequantize_block_fp8(
                    tensor[first_row : min(last_block * 128, shape[0])], scales[first_block:last_block], dtype=dtype
                )
                value = value[start - first_row : end - first_row]
    return value, shape, scale_name


def load_sharded_text_checkpoint(model, directory):
    """Use before any forward/optimizer; preserves FSDP2 Parameter identities."""
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    quant = config.get("quantization_config", {})
    if quant.get("quant_method") != "fp8" or quant.get("fmt") != "e4m3" or quant.get("weight_block_size") != [128, 128]:
        raise ValueError("Expected canonical E4M3 128x128 checkpoint")
    if model.config.tie_word_embeddings:
        raise NotImplementedError("Sharded tied embedding loading requires a separate alias test")
    mapping = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]
    consumed = set()
    generator = torch.Generator(device="cpu").manual_seed(42)
    for name, parameter in model.named_parameters():
        if not isinstance(parameter, DTensor) or parameter.is_meta:
            raise ValueError(f"{name}: call fully_shard then to_empty before loading")
        if parameter.device_mesh.ndim != 1 or parameter.placements != (Shard(0),):
            raise ValueError(f"{name}: only one-dimensional FSDP Shard(0) is supported")
        local = parameter.to_local()
        size = math.ceil(parameter.shape[0] / parameter.device_mesh.size())
        start = min(parameter.device_mesh.get_local_rank() * size, parameter.shape[0])
        end = start + local.shape[0]
        if ".lora_" in name:
            if ".lora_A." in name:
                # One adapter matrix at a time; never instantiate the full base.
                full = torch.empty(tuple(parameter.shape), dtype=parameter.dtype, device="cpu")
                torch.nn.init.kaiming_uniform_(full, a=math.sqrt(5), generator=generator)
                value = full[start:end]
            elif ".lora_B." in name:
                value = torch.zeros(tuple(local.shape), dtype=parameter.dtype, device="cpu")
            else:
                raise ValueError(f"Unsupported adapter tensor {name}")
        else:
            canonical = source_parameter_name(name, model.config)
            if canonical not in mapping:
                raise ValueError(f"Missing canonical parameter {canonical}")
            value, shape, scale_name = read_row_shard(directory, mapping, canonical, start, end, parameter.dtype)
            if tuple(shape) != tuple(parameter.shape):
                raise ValueError(f"Checkpoint/model shape mismatch for {canonical}")
            consumed.add(canonical)
            if scale_name:
                consumed.add(scale_name)
                mark_fp8_experts(model, name)
        if value.shape != local.shape:
            raise ValueError(f"Local shape mismatch for {name}")
        with torch.no_grad():
            local.copy_(value)
    extras = (
        owned_text_names(mapping, getattr(model.config, "_day0_source_layer_ids", model.config.num_hidden_layers))
        - consumed
    )
    if extras:
        raise ValueError(f"Unconsumed text checkpoint tensors: {sorted(extras)}")
    marked = [name for name, module in model.named_modules() if getattr(module, "fp8_act_quant", False)]
    switch = os.environ.get("VERL_STEP5_FP8_ACT_QUANT", "1")
    logger.info("FP8 activation QAT on %d MoE layers (VERL_STEP5_FP8_ACT_QUANT=%s)", len(marked), switch)
    return sorted(consumed)


def build_meta_model(model_config, engine_config):
    if model_config.model_type != "language_model":
        raise ValueError("Step5 startup is text causal-LM only")
    if engine_config.strategy != "fsdp2":
        raise ValueError("Step5 training startup requires FSDP2")
    if engine_config.ulysses_sequence_parallel_size != 1:
        raise ValueError("Step5 training does not support sequence parallelism")
    if model_config.use_liger or model_config.use_fused_kernels:
        raise ValueError("Step5 training kernels have not been validated")
    if model_config.enable_activation_offload:
        raise ValueError("Step5 activation offload has not been validated")
    if (
        model_config.target_modules != "all-linear"
        or model_config.target_parameters is not None
        or model_config.exclude_modules is not None
    ):
        raise ValueError("Step5 startup uses explicit text LoRA targets, not custom overrides")
    qat = getattr(engine_config, "qat", None)
    if qat is not None and getattr(qat, "enable", False):
        raise ValueError("Step5 QAT has not been validated")
    if engine_config.offload_policy or engine_config.param_offload or engine_config.optimizer_offload:
        raise ValueError("Step5 startup currently requires GPU-resident FSDP shards")
    if engine_config.forward_only:
        raise ValueError("Separate reference-engine startup has not been validated")
    if model_config.lora_rank <= 0 or model_config.lora_adapter_path is not None:
        raise ValueError("Step5 startup requires fresh LoRA; resume via VERL checkpoint manager")
    if model_config.lora_rank % 8:
        # Smaller or unaligned ranks fall back to per-token expert factors, a [tokens, rank, hidden] temporary.
        raise ValueError("Step5 LoRA rank must be a multiple of 8 (grouped-GEMM alignment)")
    if model_config.lora.get("merge", False):
        raise ValueError("Step5 LoRA cannot be merged: PEFT does not merge shared outer expert factors")
    config = copy.deepcopy(model_config.hf_config)
    config.dtype = torch.bfloat16
    with torch.device("meta"):
        model = Step5ForCausalLM(config)
    if model_config.enable_gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def build_lora_model(model, model_config):
    # Shared outer expert factors are the default; VERL_STEP5_LORA_PER_EXPERT=1 keeps every factor per expert.
    model = attach_text_lora(
        model,
        rank=model_config.lora_rank,
        alpha=model_config.lora_alpha,
        shared_outer=os.environ.get("VERL_STEP5_LORA_PER_EXPERT", "0") != "1",
    )
    model.enable_input_require_grads()
    return model


def shard_and_load(model, model_config, engine_config, mesh):
    from verl.utils.device import get_device_id
    from verl.utils.fsdp_utils import apply_fsdp2

    if mesh.ndim != 1:
        raise ValueError("Step5 local checkpoint loader requires one-dimensional FSDP")
    # Preserve explicit BF16 projections and F32 norms/router rather than casting all parameters.
    policy = MixedPrecisionPolicy(param_dtype=None, reduce_dtype=torch.float32)
    apply_fsdp2(
        model,
        {"mesh": mesh, "mp_policy": policy, "reshard_after_forward": engine_config.reshard_after_forward},
        engine_config,
    )
    model.to_empty(device=get_device_id())
    load_sharded_text_checkpoint(model, model_config.local_path)
    return model
