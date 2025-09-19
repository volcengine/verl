#!/usr/bin/env python3
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
"""# noqa: D400
A simplified and robust SFT (Supervised Fine-Tuning) test script for VERL's DeepSpeedEngine.

This script is designed for correctness and clear diagnostics. It trains a model on an SFT
dataset, iterating through multiple batches, and provides essential logs to monitor
training health (loss, gradients, parameter updates).

Key Features:
- Iterates through a real dataset using a PyTorch DataLoader.
- Uses HuggingFace built-in causal LM loss for robust training.
- Clear, concise logging for loss, gradient mean, and parameter delta.
- Saves a loss curve plot (`loss_curve.png`) and training summary (`training_summary.json`).

Environment Variables:
  MODEL_ID          : HuggingFace model ID (default: 'microsoft/DialoGPT-small').
  DATASET_ID        : HuggingFace dataset ID (default: 'databricks/databricks-dolly-15k').
  NUM_STEPS         : Total training steps (default: 500).
  MAX_SAMPLES       : Max samples to load from dataset (default: 200).
  BATCH_SIZE        : Training batch size (default: 4).
  MAX_LENGTH        : Tokenization max length (default: 128).
  MIXED_PRECISION   : 'fp16', 'bf16', or 'fp32' (default: 'fp32').
  CPU_OFFLOAD       : '1' to enable ZeRO-3 CPU offload (default: '0').
"""

import json
import logging
import os
import random
import socket
import sys

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset  # noqa: E402
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer  # noqa: E402

from verl import DataProto  # noqa: E402
from verl.trainer.config import CheckpointConfig  # noqa: E402
from verl.workers.config import (  # noqa: E402
    DeepSpeedEngineConfig,
    DeepSpeedOptimizerConfig,
    HFModelConfig,
)
from verl.workers.engine.deepspeed.engine_impl import DeepSpeedEngineWithLMHead  # noqa: E402

# NOTE: We intentionally modify sys.path early to allow running this example
# without installing the package. Ruff E402 is suppressed for this file.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:  # pragma: no cover - environment setup
    sys.path.insert(0, _REPO_ROOT)
# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("simple_sft_test")


def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_dist():
    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29619")  # Use a different port
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)


# --- Data Handling ---
class SFTDataset(Dataset):
    def __init__(self, dataset_id: str, max_samples: int):
        try:
            self.dataset = load_dataset(dataset_id, split="train")
            if max_samples > 0 and max_samples < len(self.dataset):
                self.dataset = self.dataset.select(range(max_samples))
        except Exception as e:
            log.warning(f"Failed to load '{dataset_id}': {e}. Using dummy data.")
            self.dataset = [
                {"instruction": "Hello", "output": "Hi there!"},
                {"instruction": "What is 2+2?", "output": "It's 4."},
            ]
        log.info(f"Loaded {len(self.dataset)} samples.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        instruction = item.get("instruction") or item.get("prompt") or ""
        inp = item.get("input") or ""
        output = item.get("output") or item.get("response") or ""

        if inp:
            prompt = f"Instruction: {instruction}\nInput: {inp}\nOutput: "
        else:
            prompt = f"Instruction: {instruction}\nOutput: "

        return {"prompt": prompt, "response": output}


def collate_fn(batch: list[dict[str, str]], tokenizer, max_length: int):
    """Tokenize and prepare a batch for training."""
    texts = [item["prompt"] + item["response"] for item in batch]

    full_encodings = tokenizer(texts, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")

    labels = full_encodings.input_ids.clone()
    responses_list = []

    for i in range(len(batch)):
        prompt_text = batch[i]["prompt"]
        prompt_enc = tokenizer(prompt_text, max_length=max_length, truncation=True)
        prompt_len = len(prompt_enc.input_ids)

        labels[i, :prompt_len] = -100

        # Extract valid response tokens for `responses` tensor
        response_tokens = labels[i][labels[i] != -100]
        responses_list.append(response_tokens)

    # Pad responses tensor
    max_resp_len = max(len(r) for r in responses_list) if responses_list else 0
    responses_tensor = torch.full((len(batch), max_resp_len), tokenizer.pad_token_id, dtype=torch.long)
    if max_resp_len > 0:
        for i, r in enumerate(responses_list):
            responses_tensor[i, : len(r)] = r

    seq_len = full_encodings.input_ids.shape[1]
    batch_size = full_encodings.input_ids.shape[0]
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    return {
        "input_ids": full_encodings.input_ids,
        "attention_mask": full_encodings.attention_mask,
        "labels": labels,
        "responses": responses_tensor,
        "position_ids": position_ids,
    }


# --- Engine and Training ---
def create_engine(model_id: str, mixed_precision: str, cpu_offload: bool, batch_size: int):
    log.info(f"Creating DeepSpeed engine for '{model_id}'...")
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    model_dtype = dtype_map.get(mixed_precision, torch.float32)

    model_config = HFModelConfig(
        path=model_id,
        trust_remote_code=True,
        override_config={"dtype": model_dtype, "attn_implementation": "eager"},
    )
    engine_config = DeepSpeedEngineConfig(
        param_offload=False,  # 禁用参数offloading
        optimizer_offload=False,  # 禁用优化器offloading
        mixed_precision=mixed_precision if mixed_precision != "fp32" else None,
        model_dtype=mixed_precision,
    )
    optimizer_config = DeepSpeedOptimizerConfig(
        optimizer="AdamW",
        lr=1e-4,
        betas=[0.9, 0.999],  # Adam默认参数
        eps=1e-8,  # Adam默认参数
        weight_decay=0.0,  # 与FSDP对齐
    )

    engine = DeepSpeedEngineWithLMHead(
        model_config=model_config,
        engine_config=engine_config,
        optimizer_config=optimizer_config,
        checkpoint_config=CheckpointConfig(),
        zero_stage=2,  # 改为ZeRO Stage 2
        train_micro_batch_size_per_gpu=batch_size,
        train_batch_size=batch_size,
    )
    engine.initialize()
    engine.set_train_mode()
    return engine


"""Loss function used by engine.forward_backward_batch.

Instrumentation goals:
 1. Always return (loss_tensor, metrics_dict) where metrics_dict contains a scalar 'loss'.
 2. Provide branch / shape diagnostics when DEEPSPEED_SFT_LOSS_DEBUG=1.
 3. Support both logits (standard HF) and precomputed log_probs (engine remove-padding path).
"""


def loss_function(model_output, data):
    debug = os.environ.get("DEEPSPEED_SFT_LOSS_DEBUG", "0") == "1"
    branch = None
    info = {}

    # Helper to package return
    def _ret(loss_tensor):
        metrics = {"loss": loss_tensor.detach()}
        if debug:
            log.info(f"[LOSS_DEBUG] branch={branch} loss={float(loss_tensor.detach().item()):.4f} info={info}")
        return loss_tensor, metrics

    # 1. HuggingFace style output having .loss attribute
    if hasattr(model_output, "loss") and model_output.loss is not None:
        branch = "hf.loss"
        return _ret(model_output.loss)

    # 2. Dict with explicit loss key
    if isinstance(model_output, dict) and model_output.get("loss") is not None:
        branch = "dict.loss"
        return _ret(model_output["loss"])

    # 3. Logits present (HF standard forward)
    logits = None
    if hasattr(model_output, "logits"):
        logits = model_output.logits
    elif isinstance(model_output, dict) and "logits" in model_output:
        logits = model_output["logits"]
    if logits is not None:
        branch = "logits.ce"
        labels = data["labels"]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        info.update(
            {
                "shift_logits": tuple(shift_logits.shape),
                "shift_labels": tuple(shift_labels.shape),
                "vocab": shift_logits.size(-1),
            }
        )
        return _ret(loss)

    # 4. log_probs path (DeepSpeedEngineWithLMHead forward_step provides 'log_probs')
    if isinstance(model_output, dict) and "log_probs" in model_output:
        branch = "log_probs.masked"
        log_probs = model_output["log_probs"]  # (bsz, resp_len)
        labels = data["labels"]  # (bsz, seq_len)
        # Build mask for response tokens: label != -100
        mask = (labels != -100).float()
        # Align lengths if log_probs limited to response_length (common): take trailing positions
        if log_probs.shape[-1] <= labels.shape[-1]:
            # Expect mask last dim >= log_probs last dim; slice tail
            if mask.shape[-1] > log_probs.shape[-1]:
                mask = mask[..., -log_probs.shape[-1] :]
        # If off by 1 due to next-token shift
        if log_probs.shape[-1] == mask.shape[-1] - 1:
            mask = mask[..., 1:]
        if log_probs.shape != mask.shape:
            branch = "log_probs.fallback_mean"
            loss = -log_probs.mean()
            info["reason"] = f"shape_mismatch lp={tuple(log_probs.shape)} mask={tuple(mask.shape)}"
            return _ret(loss)
        denom = mask.sum().clamp_min(1)
        loss = -(log_probs * mask).sum() / denom
        info.update({"lp_shape": tuple(log_probs.shape), "mask_sum": float(denom.detach().item())})
        return _ret(loss)

    raise ValueError(
        "Unsupported model_output format for loss computation (keys: {})".format(
            list(model_output.keys()) if isinstance(model_output, dict) else type(model_output)
        )
    )


def train(engine, dataloader, num_steps: int, tokenizer):
    """Training loop using engine.forward_backward_batch to test engine_impl logic."""
    log.info("Starting training loop (engine.forward_backward_batch)...")
    model = engine.engine.module
    model.train()

    losses, grad_means, param_deltas = [], [], []
    last_param_ref = None
    data_iter = iter(dataloader)

    for step in range(num_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        device = next(model.parameters()).device
        for k, v in batch.items():
            batch[k] = v.to(device)

        input_ids = batch["input_ids"]
        labels = batch["labels"]
        pad_id = tokenizer.pad_token_id
        bsz, seq_len = input_ids.size()

        # Use compact responses produced by collate_fn if available; else derive from labels tail
        if "responses" in batch and batch["responses"].ndim == 2:
            responses = batch["responses"]  # (bsz, resp_len)
        else:
            # Derive response tokens by compressing non -100 positions (assumed tail contiguous)
            resp_list = []
            max_len = 0
            for i in range(bsz):
                resp_tokens = labels[i][labels[i] != -100]
                if resp_tokens.numel() == 0:  # ensure at least 1 prompt token exists: mask first token and take rest
                    labels[i, 0] = -100
                    resp_tokens = labels[i][labels[i] != -100]
                resp_list.append(resp_tokens)
                max_len = max(max_len, resp_tokens.numel())
            responses = torch.full((bsz, max_len), pad_id, dtype=labels.dtype, device=labels.device)
            for i, r in enumerate(resp_list):
                responses[i, : r.numel()] = r
        resp_len = responses.size(-1)
        if resp_len >= seq_len:  # guarantee assertion response_len < seq_len
            responses = responses[:, : seq_len - 1]
            resp_len = responses.size(-1)

        # Masks
        # prompt tokens are those masked -100; response tail length is resp_len
        response_mask_full = labels != -100
        # Derive prompt_mask aligned with input_ids length
        prompt_mask = labels == -100

        batch["responses"] = responses
        batch["response_mask"] = response_mask_full[:, -resp_len:] if resp_len > 0 else response_mask_full
        batch["prompts"] = input_ids  # keep full sequence for engine slicing
        batch["prompt_mask"] = prompt_mask
        if "position_ids" not in batch:
            batch["position_ids"] = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, -1)

        # Wrap into DataProto
        batch_proto = DataProto.from_single_dict(batch)
        # Ensure position_ids present in underlying tensordict
        for attr in ("tensor", "data", "td"):
            td = getattr(batch_proto, attr, None)
            if td is not None and "position_ids" not in td.keys():
                td["position_ids"] = batch["position_ids"]
                break

        batch_proto.meta_info.update(
            {
                "use_dynamic_bsz": False,
                "micro_batch_size_per_gpu": bsz,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": 0,
                "max_new_tokens": 0,
                "do_sample": False,
                "task_type": "sft",
            }
        )

        try:
            metrics = engine.forward_backward_batch(batch_proto, loss_function)
        except Exception:  # noqa: BLE001
            log.exception("Step %s forward/backward failed", step + 1)
            continue

        # metrics is a dict whose values are lists (one per micro-batch). Collect loss.
        if "loss" not in metrics:
            # Attempt on-the-fly reconstruction from log_probs if present.
            if "log_probs" in metrics:
                log.warning("'loss' missing in metrics; attempting reconstruction from log_probs (debug).")
                lp_list = metrics["log_probs"]
                lp = lp_list[0] if isinstance(lp_list, list) else lp_list
                # Simple fallback: mean negative log prob
                loss_tensor = -lp.mean()
            else:
                log.error("Metrics returned without 'loss' and no log_probs to reconstruct; skipping step.")
                continue
        else:
            loss_vals = metrics["loss"]
            # Convert list -> tensor list -> mean
            if isinstance(loss_vals, list):
                loss_tensors = [lv if torch.is_tensor(lv) else torch.as_tensor(lv, device=device) for lv in loss_vals]
                loss_tensor = torch.stack([lt.float() for lt in loss_tensors]).mean()
            else:
                loss_tensor = loss_vals if torch.is_tensor(loss_vals) else torch.as_tensor(loss_vals, device=device)

        # Optional debug dump
        if os.environ.get("DEEPSPEED_SFT_LOSS_DEBUG", "0") == "1":
            log.info(
                "[STEP_DEBUG] step=%s metrics_keys=%s loss_tensor_shape=%s",
                step + 1,
                list(metrics.keys()),
                getattr(loss_tensor, "shape", None),
            )

        # Gradient diagnostics AFTER engine internal backward
        grad_total, grad_cnt = 0.0, 0
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    grad_total += p.grad.data.abs().mean().item()
                    grad_cnt += 1
        grad_mean = grad_total / grad_cnt if grad_cnt > 0 else 0.0

        # Optimizer step / zero grad via engine helpers
        engine.optimizer_step()
        engine.optimizer_zero_grad()

        # Param delta tracking
        param_delta = 0.0
        with torch.no_grad():
            p0 = next(model.parameters())
            if last_param_ref is not None:
                param_delta = (p0 - last_param_ref).abs().mean().item()
            last_param_ref = p0.detach().clone()

        loss_scalar = float(loss_tensor.detach().item())
        losses.append(loss_scalar)
        grad_means.append(grad_mean)
        param_deltas.append(param_delta)

    lr_val = engine.optimizer.param_groups[0]["lr"] if hasattr(engine, "optimizer") else -1
    # Build final step summary (wrap to respect line length limits)
    summary_msg = (
        f"Step {step + 1}/{num_steps} | Loss: {loss_scalar:.4f} | Grad Mean: {grad_mean:.3e} | "
        f"Param Δ: {param_delta:.3e} | LR: {lr_val:.2e}"
    )
    log.info(summary_msg)

    return losses, grad_means, param_deltas


# --- Main Execution ---
def main():
    seed_all()
    init_dist()

    # --- Config ---
    model_id = os.environ.get("MODEL_ID", "microsoft/DialoGPT-small")
    dataset_id = os.environ.get("DATASET_ID", "databricks/databricks-dolly-15k")
    num_steps = int(os.environ.get("NUM_STEPS", 200))  # 与 FSDP 版本保持一致
    max_samples = int(os.environ.get("MAX_SAMPLES", 200))
    batch_size = int(os.environ.get("BATCH_SIZE", 4))
    max_length = int(os.environ.get("MAX_LENGTH", 128))
    mixed_precision = os.environ.get("MIXED_PRECISION", "fp32")
    cpu_offload = False  # 硬编码为False，忽略环境变量

    log.info(f"Host: {socket.gethostname()}, PID: {os.getpid()}")
    log.info(f"Config: model={model_id}, steps={num_steps}, precision={mixed_precision}, offload={cpu_offload}")

    # --- Setup ---
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset = SFTDataset(dataset_id, max_samples)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer, max_length)
    )

    engine = create_engine(model_id, mixed_precision, cpu_offload, batch_size)

    # --- Run Training ---
    losses, grad_means, param_deltas = train(engine, dataloader, num_steps, tokenizer)

    # --- Analysis & Reporting ---
    if not losses:
        log.error("Training finished with no losses recorded. Aborting analysis.")
        return 1

    log.info(f"Training complete. Final loss: {losses[-1]:.4f}")

    # Save summary
    summary = {
        "model_id": model_id,
        "steps": len(losses),
        "final_loss": losses[-1],
        "mean_grad": np.mean(grad_means),
        "mean_param_delta": np.mean(param_deltas[1:]),  # skip first delta
    }
    with open("training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Saved training_summary.json")

    # Save plot
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_theme(style="whitegrid")

        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Plot loss
        ax1.plot(losses, label="Loss", color="tab:blue")
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Loss", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")

        # Plot gradients and deltas on a secondary y-axis
        ax2 = ax1.twinx()
        ax2.plot(grad_means, label="Gradient Mean", color="tab:green", linestyle="--", alpha=0.7)
        ax2.plot(param_deltas, label="Param Delta", color="tab:red", linestyle=":", alpha=0.7)
        ax2.set_ylabel("Gradient / Delta", color="tab:gray")
        ax2.tick_params(axis="y", labelcolor="tab:gray")
        ax2.set_yscale("log")

        fig.suptitle(f"Training Metrics: {model_id}")
        fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9))
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig("loss_curve.png", dpi=150)
        log.info("Saved loss_curve.png")
    except ImportError:
        log.warning("Plotting libraries not found. Skipping plot generation.")

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    return 0


if __name__ == "__main__":
    sys.exit(main())
