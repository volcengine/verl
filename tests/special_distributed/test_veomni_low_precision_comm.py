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

"""Two-GPU acceptance of VeOmni's low-precision communication through verl.

Run in the repository's prepared VeOmni CUDA environment, on one Linux host:

    uv run --no-sync torchrun --standalone --nproc-per-node=2 \
        tests/special_distributed/test_veomni_low_precision_comm.py

A tiny dense Qwen3 checkpoint is generated locally; no model download is needed.
Both real verl engines load identical weights and consume identical rank-specific
inputs for two steps, with two micro-batches per step. Observers delegate to the
installed VeOmni hook and real NCCL collectives without replacing their math.
This is a correctness acceptance test, not a throughput benchmark or a claim of
equivalent convergence. The relative-L2 bounds are explicit acceptance limits.
"""

import gc
import json
import math
import os
import tempfile
from contextlib import ExitStack
from unittest.mock import patch

import torch
import torch.distributed as dist
from tensordict import TensorDict
from torch.distributed.tensor import DTensor
from transformers import Qwen3Config, Qwen3ForCausalLM
from veomni.distributed.fsdp2.reduce_scatter import FP32ReduceScatterWithLowPrecisionTransport

from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.workers.config import HFModelConfig, VeOmniEngineConfig, VeOmniOptimizerConfig
from verl.workers.engine.veomni.transformer_impl import VeOmniEngineWithLMHead


def _make_checkpoint(path):
    torch.manual_seed(1234)
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=256,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            max_position_embeddings=128,
            attention_dropout=0.0,
            tie_word_embeddings=False,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    )
    model.save_pretrained(path, safe_serialization=True)


def _batch(rank, step):
    generator = torch.Generator().manual_seed(2000 + rank * 10 + step)
    ids = torch.randint(3, 256, (2, 32), generator=generator)
    offsets = torch.tensor([0, 32, 64])
    mask = torch.ones_like(ids, dtype=torch.bool)
    mask[:, -1] = False  # Exclude the cyclic target introduced by the engine's roll.
    data = TensorDict(
        {
            "input_ids": torch.nested.nested_tensor_from_jagged(ids.flatten(), offsets),
            "position_ids": torch.nested.nested_tensor_from_jagged(torch.arange(32).repeat(2), offsets),
            "loss_mask": mask,
        },
        batch_size=[2],
    )
    tu.assign_non_tensor(
        data,
        temperature=1.0,
        use_remove_padding=False,
        use_fused_kernels=False,
        use_dynamic_bsz=False,
        micro_batch_size_per_gpu=1,
        calculate_entropy=False,
    )
    return data


def _snapshot(engine, gradients=False):
    names, tensors = [], []
    for name, parameter in sorted(engine.module.named_parameters()):
        if not parameter.requires_grad:
            continue
        tensor = parameter.grad if gradients else parameter
        assert tensor is not None, f"Missing {'gradient' if gradients else 'parameter'}: {name}"
        local = tensor.to_local() if isinstance(tensor, DTensor) else tensor
        assert local.dtype == torch.float32, f"{name}: expected FP32, got {local.dtype}"
        local = local.detach().cpu().reshape(-1).clone()
        assert torch.isfinite(local).all(), f"Non-finite values in {name}"
        names.append((name, local.numel()))
        tensors.append(local)
    return names, torch.cat(tensors)


def _run(path, enabled):
    engine = VeOmniEngineWithLMHead(
        model_config=HFModelConfig(
            path=path,
            load_tokenizer=False,
            use_remove_padding=False,
            enable_gradient_checkpointing=False,
            override_config={"attn_implementation": "eager"},
        ),
        engine_config=VeOmniEngineConfig(
            fsdp_size=2,
            mixed_precision=True,
            low_precision_reduce_scatter_comm=enabled,
            init_device="meta",
            attn_implementation="eager",
            use_torch_compile=False,
            enable_full_shard=True,
        ),
        optimizer_config=VeOmniOptimizerConfig(lr=1e-3, weight_decay=0.0, total_training_steps=2, clip_grad=100.0),
        checkpoint_config=CheckpointConfig(),
    )
    engine.initialize()
    names, initial = _snapshot(engine)
    observations = {"hook": 0, "wire": 0, "native": 0, "loss_calls": 0}
    original_hook = FP32ReduceScatterWithLowPrecisionTransport.__call__
    original_wire = dist.all_to_all_single
    native_name = "reduce_scatter_single" if hasattr(dist, "reduce_scatter_single") else "reduce_scatter_tensor"
    original_native = getattr(dist, native_name)

    def observe_hook(self, output_tensor, input_tensor, *args, **kwargs):
        assert input_tensor.dtype == output_tensor.dtype == torch.float32
        # Each microbatch is reduced before FP32 gradient accumulation; no
        # previously accumulated FP32 values should be rounded on the wire.
        torch.testing.assert_close(input_tensor, input_tensor.to(torch.bfloat16).float(), rtol=0, atol=0)
        result = original_hook(self, output_tensor, input_tensor, *args, **kwargs)
        assert output_tensor.dtype == torch.float32
        observations["hook"] += 1
        return result

    def observe_wire(output, input, *args, **kwargs):
        assert input.dtype == output.dtype == torch.bfloat16
        observations["wire"] += 1
        return original_wire(output, input, *args, **kwargs)

    def observe_native(output, input, *args, **kwargs):
        assert input.dtype == output.dtype == torch.float32
        observations["native"] += 1
        return original_native(output, input, *args, **kwargs)

    def loss_function(model_output, data, dp_group):
        observations["loss_calls"] += 1
        log_probs = model_output["log_probs"].values()
        mask = data["loss_mask"].reshape(-1)
        # FSDP averages ranks; restore the global token-mean gradient.
        token_count = tu.get_non_tensor_data(data, "batch_num_tokens", None)
        assert token_count is not None and token_count > 0
        scale = dist.get_world_size(dp_group) / token_count
        return -(log_probs.float() * mask).sum() * scale, {}

    steps = []
    with ExitStack() as stack:
        stack.enter_context(patch.object(FP32ReduceScatterWithLowPrecisionTransport, "__call__", observe_hook))
        stack.enter_context(patch.object(dist, "all_to_all_single", observe_wire))
        stack.enter_context(patch.object(dist, native_name, observe_native))
        with engine.train_mode():
            for step in range(2):
                before_names, before = _snapshot(engine)
                engine.optimizer_zero_grad()
                output = engine.forward_backward_batch(_batch(dist.get_rank(), step), loss_function)
                grad_names, gradients = _snapshot(engine, gradients=True)
                grad_norm = engine.optimizer_step()
                engine.lr_scheduler.step()
                after_names, after = _snapshot(engine)
                assert before_names == grad_names == after_names == names
                assert len(output["loss"]) == 2, "Expected two micro-batches per optimizer step"
                assert all(math.isfinite(loss) for loss in output["loss"])
                assert math.isfinite(grad_norm) and grad_norm > 0
                update = after - before
                assert torch.count_nonzero(update), "Optimizer did not update the model"
                steps.append({"gradient": gradients, "update": update})

    assert observations["loss_calls"] == 4
    if enabled:
        assert observations["hook"] > 0, "Requested transport silently fell back to native ReduceScatter"
        assert observations["wire"] == observations["hook"], observations
    else:
        assert observations["hook"] == observations["wire"] == 0, observations
        assert observations["native"] > 0, "Native FP32 ReduceScatter was not observed"
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    return names, initial, steps, observations


def _compare(reference, candidate, tolerance):
    difference = (candidate - reference).double()
    stats = torch.tensor(
        [difference.square().sum(), reference.double().square().sum(), difference.abs().max()],
        dtype=torch.float64,
        device=torch.cuda.current_device(),
    )
    dist.all_reduce(stats[:2], op=dist.ReduceOp.SUM)
    dist.all_reduce(stats[2:], op=dist.ReduceOp.MAX)
    relative_l2 = (stats[0] / stats[1].clamp_min(1e-30)).sqrt().item()
    assert math.isfinite(relative_l2) and relative_l2 <= tolerance, (relative_l2, tolerance)
    return {"relative_l2": relative_l2, "max_abs": stats[2].item(), "relative_l2_limit": tolerance}


def main():
    assert torch.cuda.is_available(), "This acceptance test requires CUDA"
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    assert torch.cuda.is_bf16_supported(), "This acceptance test requires BF16-capable GPUs"
    dist.init_process_group("nccl")
    assert dist.get_world_size() == 2, "Launch exactly two ranks on one Linux host"
    directory = tempfile.TemporaryDirectory(prefix="verl-veomni-comm-") if dist.get_rank() == 0 else None
    try:
        paths = [directory.name if directory is not None else None]
        dist.broadcast_object_list(paths, src=0)
        if dist.get_rank() == 0:
            _make_checkpoint(paths[0])
        dist.barrier()
        baseline = _run(paths[0], False)
        candidate = _run(paths[0], True)
        assert baseline[0] == candidate[0]
        torch.testing.assert_close(baseline[1], candidate[1], rtol=0, atol=0)
        results = []
        for reference_step, candidate_step in zip(baseline[2], candidate[2], strict=True):
            results.append(
                {
                    "gradient": _compare(reference_step["gradient"], candidate_step["gradient"], 0.02),
                    "update": _compare(reference_step["update"], candidate_step["update"], 0.05),
                }
            )
        if dist.get_rank() == 0:
            print(json.dumps({"native": baseline[3], "low_precision": candidate[3], "steps": results}, indent=2))
            print("PASS: real verl VeOmni engine, BF16 wire, FP32 hook output/gradients, two accumulated steps")
        dist.barrier()
    finally:
        dist.destroy_process_group()
        if directory is not None:
            directory.cleanup()


if __name__ == "__main__":
    main()
