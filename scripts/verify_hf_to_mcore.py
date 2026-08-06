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

"""Verify an HF -> mcore conversion by comparing weights, not activations.

``converter_hf_to_mcore.py --test`` cannot answer whether the conversion is
correct. It builds ``dut_state_dict`` from the in-memory model that was just
populated from the HF weights and ``ref_state_dict`` by reading that same
checkpoint back, so both sides carry any mapping error equally and it still
prints "Conversion test passed!". It also downgrades missing tensors to
``[Warning]`` without failing, and on a 7B dense model it allocates a second
DDP-wrapped copy, which OOMs a 64GB card before it can report anything.

This script closes that gap. It never runs a forward pass: it un-fuses the mcore
tensors back into the HF layout and compares them elementwise with the original
safetensors, so a mismatch means the mapping is wrong and nothing else. Reading
activations instead would fold in kernel and accumulation-order differences,
which on Ascend reach O(1) on bf16 logits even when every weight is bit-exact.

Usage:
    python scripts/verify_hf_to_mcore.py \\
        --hf_model_path /path/to/hf_model \\
        --mcore_path    /path/to/mcore_model

Exit code 0 means every compared tensor matched.
"""

import argparse
import glob
import os
import sys

import torch

# Everything heavier than torch is imported lazily inside the functions that
# need it, so that unfuse_mcore_qkv stays importable (and unit-testable) on a
# CPU-only box without megatron, mindspeed or an NPU toolkit.


def unfuse_mcore_qkv(
    fused: torch.Tensor, num_attention_heads: int, num_key_value_heads: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split an mcore fused ``linear_qkv`` tensor back into HF q/k/v.

    The fused layout is interleaved per query group, not ``cat([q, k, v])``::

        [ q_heads_of_group_0, k_head_0, v_head_0,
          q_heads_of_group_1, k_head_1, v_head_1, ... ]

    which mirrors ``_broadcast_tp_shard_tensor_qkv`` in
    ``verl/models/mcore/loader.py`` for ``tp_size == 1``. Works for both the 2-D
    weight and the 1-D bias, since only dim 0 is regrouped.
    """
    q_per_group = (num_attention_heads // num_key_value_heads) * head_dim
    group_stride = q_per_group + 2 * head_dim
    expected_rows = group_stride * num_key_value_heads
    if fused.shape[0] != expected_rows:
        raise ValueError(
            f"fused qkv has {fused.shape[0]} rows, expected {expected_rows} for "
            f"{num_attention_heads} heads / {num_key_value_heads} kv heads / head_dim {head_dim}"
        )

    q_parts, k_parts, v_parts = [], [], []
    for group in range(num_key_value_heads):
        block = fused[group * group_stride : (group + 1) * group_stride]
        q_parts.append(block[:q_per_group])
        k_parts.append(block[q_per_group : q_per_group + head_dim])
        v_parts.append(block[q_per_group + head_dim :])
    return torch.cat(q_parts, 0), torch.cat(k_parts, 0), torch.cat(v_parts, 0)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_model_path", required=True, help="The path for the huggingface model")
    parser.add_argument("--mcore_path", required=True, help="The path for the converted mcore model")
    parser.add_argument("--atol", type=float, default=0.0, help="Absolute tolerance; 0.0 demands an exact match")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--max_report", type=int, default=12, help="How many mismatches to print")
    return parser.parse_args()


def _init_distributed():
    import torch.distributed as dist

    from verl.utils.device import get_device_name, get_nccl_backend, get_torch_device

    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12357")
    dist.init_process_group(get_nccl_backend())
    get_torch_device().set_device(f"{get_device_name()}:{os.getenv('LOCAL_RANK', 0)}")

    from megatron.core import parallel_state as mpu
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    mpu.initialize_model_parallel(1, 1, None, context_parallel_size=1, expert_model_parallel_size=1)
    model_parallel_cuda_manual_seed(0)


def _load_hf_state_dict(hf_model_path):
    from safetensors.torch import load_file

    shards = sorted(glob.glob(os.path.join(hf_model_path, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"no *.safetensors under {hf_model_path}")
    merged = {}
    for shard in shards:
        merged.update(load_file(shard))
    return merged


def _load_mcore_state_dict(hf_config, mcore_path):
    from megatron.core.models.gpt.gpt_model import ModelType

    from verl.models.mcore import hf_to_mcore_config
    from verl.utils.megatron.dist_checkpointing import load_dist_checkpointing
    from verl.utils.megatron_utils import get_model

    tfconfig = hf_to_mcore_config(hf_config, torch.bfloat16)
    tie_word_embeddings = getattr(hf_config, "tie_word_embeddings", False)

    def megatron_model_provider(pre_process, post_process):
        from verl.models.mcore import init_mcore_model

        return init_mcore_model(
            tfconfig,
            hf_config,
            pre_process,
            post_process,
            share_embeddings_and_output_weights=tie_word_embeddings,
            value=False,
        )

    model = get_model(
        model_provider_func=megatron_model_provider,
        model_type=ModelType.encoder_or_decoder,
        wrap_with_ddp=False,
        transformer_config=tfconfig,
    )
    # load_dist_checkpointing RETURNS the populated dict; the ShardedTensor
    # placeholders in the input are not filled in place.
    return load_dist_checkpointing(model[0].module.sharded_state_dict(), mcore_path)


def _compare(name, mcore_tensor, hf_tensor, atol, results):
    lhs = mcore_tensor.detach().to(torch.float32).cpu()
    rhs = hf_tensor.detach().to(torch.float32).cpu()
    if lhs.shape != rhs.shape:
        results.append((name, False, f"shape {tuple(lhs.shape)} vs {tuple(rhs.shape)}"))
        return
    max_diff = (lhs - rhs).abs().max().item()
    results.append((name, max_diff <= atol, f"max_abs_diff={max_diff:.3e}"))


def main():
    args = _parse_args()

    try:
        # NPU patch. Must precede any megatron import: megatron's RNG setup
        # reaches torch.cuda directly, and this is what redirects it to the NPU.
        import mindspeed.megatron_adaptor  # noqa: F401
    except ImportError:
        pass

    from transformers import AutoConfig

    _init_distributed()

    hf_config = AutoConfig.from_pretrained(args.hf_model_path, trust_remote_code=args.trust_remote_code)
    head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
    print(
        f"[info] {hf_config.architectures} layers={hf_config.num_hidden_layers} "
        f"heads={hf_config.num_attention_heads} kv_heads={hf_config.num_key_value_heads} head_dim={head_dim}",
        flush=True,
    )

    print("[info] loading HF safetensors ...", flush=True)
    hf_sd = _load_hf_state_dict(args.hf_model_path)
    print("[info] loading mcore dist checkpoint ...", flush=True)
    mcore_sd = _load_mcore_state_dict(hf_config, args.mcore_path)

    results = []

    for mcore_name, hf_name in (
        ("embedding.word_embeddings.weight", "model.embed_tokens.weight"),
        ("decoder.final_layernorm.weight", "model.norm.weight"),
        ("output_layer.weight", "lm_head.weight"),
    ):
        if mcore_name in mcore_sd and hf_name in hf_sd:
            _compare(mcore_name, mcore_sd[mcore_name], hf_sd[hf_name], args.atol, results)
        elif mcore_name in mcore_sd or hf_name in hf_sd:
            results.append((mcore_name, False, f"present on one side only (mcore={mcore_name in mcore_sd})"))

    for layer in range(hf_config.num_hidden_layers):
        mcore_prefix = f"decoder.layers.{layer}"
        hf_prefix = f"model.layers.{layer}"

        for suffix, hf_name in (
            ("self_attention.linear_qkv.layer_norm_weight", f"{hf_prefix}.input_layernorm.weight"),
            ("self_attention.linear_proj.weight", f"{hf_prefix}.self_attn.o_proj.weight"),
            ("self_attention.q_layernorm.weight", f"{hf_prefix}.self_attn.q_norm.weight"),
            ("self_attention.k_layernorm.weight", f"{hf_prefix}.self_attn.k_norm.weight"),
            ("mlp.linear_fc1.layer_norm_weight", f"{hf_prefix}.post_attention_layernorm.weight"),
            ("mlp.linear_fc2.weight", f"{hf_prefix}.mlp.down_proj.weight"),
        ):
            key = f"{mcore_prefix}.{suffix}"
            if key in mcore_sd and hf_name in hf_sd:
                _compare(key, mcore_sd[key], hf_sd[hf_name], args.atol, results)

        for kind in ("weight", "bias"):
            key = f"{mcore_prefix}.self_attention.linear_qkv.{kind}"
            if key not in mcore_sd:
                continue
            q, k, v = unfuse_mcore_qkv(
                mcore_sd[key], hf_config.num_attention_heads, hf_config.num_key_value_heads, head_dim
            )
            for part, tensor in (("q", q), ("k", k), ("v", v)):
                _compare(
                    f"{key}->{part}", tensor, hf_sd[f"{hf_prefix}.self_attn.{part}_proj.{kind}"], args.atol, results
                )

        key = f"{mcore_prefix}.mlp.linear_fc1.weight"
        if key in mcore_sd:
            fused = mcore_sd[key]
            intermediate = hf_config.intermediate_size
            _compare(
                f"{key}->gate", fused[:intermediate], hf_sd[f"{hf_prefix}.mlp.gate_proj.weight"], args.atol, results
            )
            _compare(f"{key}->up", fused[intermediate:], hf_sd[f"{hf_prefix}.mlp.up_proj.weight"], args.atol, results)

    failed = [r for r in results if not r[1]]

    buckets = {}
    for name, ok, _ in results:
        bucket = "qkv" if "linear_qkv" in name else ("mlp" if ".mlp." in name else "other")
        stats = buckets.setdefault(bucket, [0, 0])
        stats[0] += 1
        stats[1] += 0 if ok else 1

    print(f"\n{'=' * 66}")
    for bucket, (total, bad) in sorted(buckets.items()):
        print(f"{bucket:6s}: {total - bad}/{total} match")
    print(f"total : {len(results) - len(failed)}/{len(results)} match (atol={args.atol})")

    if failed:
        print(f"\nfirst {min(args.max_report, len(failed))} mismatches:")
        for name, _, detail in failed[: args.max_report]:
            print(f"  FAIL {name}  {detail}")
        print("\nhint: qkv-only failures point at the GQA interleaving; failures everywhere")
        print("      usually mean the checkpoint belongs to a different model.")

    print(f"\nWeight-level mapping: {'PASS' if not failed else 'FAIL'}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
