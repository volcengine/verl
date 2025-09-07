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
A simplified SFT (Supervised Fine-Tuning) test script for VERL's FSDPEngine.

This script demonstrates how to use VERL's FSDP engine for training. It is analogous
to the DeepSpeed version, providing a clear example for users interested in PyTorch's
native Fully Sharded Data Parallel (FSDP) capabilities.

Key Features:
- Uses PyTorch FSDP for model and optimizer sharding.
- Iterates through a real dataset using a PyTorch DataLoader.
- Provides essential logs for monitoring training health (loss, gradients).

Environment Variables:
  MODEL_ID          : HuggingFace model ID (default: 'microsoft/DialoGPT-small').
  DATASET_ID        : HuggingFace dataset ID (default: 'tatsu-lab/alpaca').
  NUM_STEPS         : Total training steps (default: 50).
  MAX_SAMPLES       : Max samples to load from dataset (default: 200).
  BATCH_SIZE        : Training batch size (default: 4).
  MAX_LENGTH        : Tokenization max length (default: 128).
  MIXED_PRECISION   : 'bf16', or 'fp32' (default: 'bf16' if supported, else 'fp32').
  CPU_OFFLOAD       : '1' to enable FSDP CPU offload (default: '1').
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
    FSDPEngineConfig,
    FSDPOptimizerConfig,
    HFModelConfig,
)
from verl.workers.engine.fsdp.engine_impl import FSDPEngineWithLMHead  # noqa: E402

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:  # pragma: no cover
    sys.path.insert(0, _REPO_ROOT)

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("simple_fsdp_sft_test")


def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_dist():
    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29620")  # Use a different port
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))


# --- Data Handling (Identical to DeepSpeed version) ---
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
    texts = [item["prompt"] + item["response"] for item in batch]

    full_encodings = tokenizer(texts, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")

    labels = full_encodings.input_ids.clone()

    for i in range(len(batch)):
        prompt_text = batch[i]["prompt"]
        prompt_enc = tokenizer(prompt_text, max_length=max_length, truncation=True)
        prompt_len = len(prompt_enc.input_ids)
        labels[i, :prompt_len] = -100

    return {
        "input_ids": full_encodings.input_ids,
        "attention_mask": full_encodings.attention_mask,
        "labels": labels,
    }


# --- Engine and Training ---
def create_fsdp_engine(model_id: str, mixed_precision: str, cpu_offload: bool):
    log.info(f"Creating FSDP engine for '{model_id}'...")

    model_config = HFModelConfig(
        path=model_id,
        trust_remote_code=True,
        override_config={"attn_implementation": "eager"},
    )

    engine_config = FSDPEngineConfig(
        param_offload=False,  # Disable param offload to avoid device mismatch
        optimizer_offload=False,  # Disable optimizer offload to avoid device mismatch
    )

    optimizer_config = FSDPOptimizerConfig(lr=1e-4)

    engine = FSDPEngineWithLMHead(
        model_config=model_config,
        engine_config=engine_config,
        optimizer_config=optimizer_config,
        checkpoint_config=CheckpointConfig(),
    )
    engine.initialize()
    # FSDP engine uses train_mode instead of set_train_mode
    engine.train_mode()
    return engine


def loss_function(model_output, data, dp_group=None):
    """Return (loss, metrics) to satisfy FSDP engine expectation.
    Adds 'loss' key so aggregated metrics dict has it.
    Set FSDP_SFT_LOSS_DEBUG=1 for verbose branch logs.
    """
    debug = os.environ.get("FSDP_SFT_LOSS_DEBUG", "0") == "1"
    branch = None

    def ret(loss):
        if not torch.is_tensor(loss):
            loss = torch.as_tensor(loss, dtype=torch.float32, device=data["input_ids"].device)
        if debug:
            log.warning(f"[LF] branch={branch} loss={float(loss.detach()):.4f} shape={tuple(loss.shape)}")
        return loss, {"loss": loss.detach()}

    # HF style
    if hasattr(model_output, "loss") and model_output.loss is not None:
        branch = "hf.loss"
        return ret(model_output.loss)

    # Dict paths
    if isinstance(model_output, dict):
        if model_output.get("loss") is not None:
            branch = "dict.loss"
            return ret(model_output["loss"])
        if "logits" in model_output:
            branch = "dict.logits.ce"
            logits = model_output["logits"]
            labels = data["labels"]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            return ret(loss)
        if "log_probs" in model_output:
            branch = "dict.log_probs.masked"
            log_probs = model_output["log_probs"]  # (bsz, resp_len)
            labels = data["labels"]
            mask = (labels != -100).float()
            # Align mask tail to response window
            if mask.shape[-1] > log_probs.shape[-1]:
                mask = mask[..., -log_probs.shape[-1] :]
            if log_probs.shape[-1] == mask.shape[-1] - 1:
                mask = mask[..., 1:]
            if log_probs.shape != mask.shape:
                branch = "dict.log_probs.fallback_mean"
                return ret(-log_probs.mean())
            denom = mask.sum().clamp_min(1)
            loss = -(log_probs * mask).sum() / denom
            return ret(loss)

    # Attribute logits
    if hasattr(model_output, "logits"):
        branch = "attr.logits.ce"
        logits = model_output.logits
        labels = data["labels"]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        return ret(loss)

    raise ValueError("loss_function could not derive loss; enable FSDP_SFT_LOSS_DEBUG=1 for details")


def _build_batch_fields(batch, tokenizer):
    """
    保证 responses_len < seq_len
    """
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    pad_id = tokenizer.pad_token_id
    bsz, seq_len = input_ids.size()

    # 若 labels 全部未被 mask（没有 -100），强制第一个 token 不预测
    no_prompt_mask = (labels != -100).all(dim=1)
    if no_prompt_mask.any():
        labels = labels.clone()
        labels[no_prompt_mask, 0] = -100
        batch["labels"] = labels

    prompt_mask = labels == -100
    # 响应区域 = 首个非 -100 之后的所有 token
    starts = []
    for i in range(bsz):
        nz = torch.nonzero(labels[i] != -100, as_tuple=False)
        if len(nz) == 0:
            starts.append(seq_len - 1)
        else:
            starts.append(nz[0].item())
    starts = torch.tensor(starts, device=input_ids.device)
    max_resp_len = (seq_len - starts).max().item()

    responses = input_ids.new_full((bsz, max_resp_len), pad_id)
    response_mask = torch.zeros((bsz, max_resp_len), dtype=torch.bool, device=input_ids.device)
    for i in range(bsz):
        s = starts[i].item()
        tokens = input_ids[i, s:]
        mask_line = labels[i, s:] != -100
        responses[i, : tokens.size(0)] = tokens
        response_mask[i, : tokens.size(0)] = mask_line

    # 如果等长，截掉首 token
    if responses.size(1) >= seq_len:
        responses = responses[:, 1:]
        response_mask = response_mask[:, 1:]

    position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, -1)

    batch["prompts"] = input_ids
    batch["prompt_mask"] = prompt_mask
    batch["responses"] = responses
    batch["response_mask"] = response_mask
    batch["position_ids"] = position_ids
    return batch


def train_fsdp(engine, dataloader, num_steps: int, tokenizer):
    log.info("Starting FSDP training loop (engine.forward_backward_batch)...")
    engine.module.train()

    losses, grad_means = [], []
    data_iter = iter(dataloader)

    for step in range(num_steps):
        try:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            device = next(engine.module.parameters()).device
            for k, v in batch.items():
                batch[k] = v.to(device)

            batch = _build_batch_fields(batch, tokenizer)

            batch_proto = DataProto.from_single_dict(batch)
            # Ensure position_ids persist
            for attr in ("tensor", "data", "td"):
                td = getattr(batch_proto, attr, None)
                if td is not None and "position_ids" not in td.keys():
                    td["position_ids"] = batch["position_ids"]
                    break

            bsz = batch["input_ids"].size(0)
            batch_proto.meta_info.update(
                {
                    "use_dynamic_bsz": False,
                    "micro_batch_size_per_gpu": bsz,
                    "global_batch_size": bsz,
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": 0,
                    "max_new_tokens": 0,
                    "do_sample": False,
                    "task_type": "sft",
                }
            )

            metrics = engine.forward_backward_batch(batch_proto, loss_function)

            if not isinstance(metrics, dict):
                raise RuntimeError(f"Engine expected to return metrics dict, got {type(metrics)}")
            if "loss" not in metrics:
                raise RuntimeError(f"Metrics missing 'loss' key. keys={list(metrics.keys())}")
            loss_list = metrics["loss"]
            # metrics['loss'] is list of tensors (one per micro-batch). Aggregate.
            if isinstance(loss_list, list):
                loss_tensors = [lt if torch.is_tensor(lt) else torch.as_tensor(lt, device=device) for lt in loss_list]
                loss_t = torch.stack([lt.float() for lt in loss_tensors]).mean()
            else:
                loss_t = loss_list if torch.is_tensor(loss_list) else torch.as_tensor(loss_list, device=device)

            loss_scalar = float(loss_t.detach())

            # Gradient stats
            grad_total, grad_cnt = 0.0, 0
            with torch.no_grad():
                for p in engine.module.parameters():
                    if p.grad is not None:
                        grad_total += p.grad.data.abs().mean().item()
                        grad_cnt += 1
            grad_mean = grad_total / grad_cnt if grad_cnt else 0.0

            engine.optimizer_step()
            engine.optimizer_zero_grad()

            losses.append(loss_scalar)
            grad_means.append(grad_mean)

            if (step + 1) % 5 == 0 or step == 0:
                log.info(
                    "Step %s/%s | Loss %.4f | GradMean %.3e | RespLen %s / SeqLen %s",
                    step + 1,
                    num_steps,
                    loss_scalar,
                    grad_mean,
                    batch["responses"].size(1),
                    batch["input_ids"].size(1),
                )
        except Exception:  # noqa: BLE001
            log.exception("Step %s failed", step + 1)
            raise

    return losses, grad_means


# --- Main Execution ---
def main():
    seed_all()
    init_dist()

    # --- Config ---
    model_id = os.environ.get("MODEL_ID", "microsoft/DialoGPT-small")
    dataset_id = os.environ.get("DATASET_ID", "databricks/databricks-dolly-15k")
    num_steps = int(os.environ.get("NUM_STEPS", 200))
    max_samples = int(os.environ.get("MAX_SAMPLES", 200))
    batch_size = int(os.environ.get("BATCH_SIZE", 4))
    max_length = int(os.environ.get("MAX_LENGTH", 128))

    default_prec = "bf16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "fp32"
    mixed_precision = os.environ.get("MIXED_PRECISION", default_prec)
    cpu_offload = os.environ.get("CPU_OFFLOAD", "1") == "1"

    log.info(f"Host: {socket.gethostname()}, PID: {os.getpid()}")
    log.info(f"Config: model={model_id}, steps={num_steps}, precision={mixed_precision}, cpu_offload={cpu_offload}")

    # --- Setup ---
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset = SFTDataset(dataset_id, max_samples)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer, max_length)
    )

    engine = create_fsdp_engine(model_id, mixed_precision, False)  # Disable CPU offload

    # --- Run Training ---
    losses, grad_means = train_fsdp(engine, dataloader, num_steps, tokenizer)

    # --- Analysis & Reporting ---
    if not losses:
        log.error("Training finished with no losses recorded. Aborting analysis.")
        return 1

    log.info(f"Training complete. Final loss: {losses[-1]:.4f}")

    summary = {
        "model_id": model_id,
        "steps": len(losses),
        "final_loss": losses[-1],
        "mean_grad": np.mean(grad_means),
    }
    with open("fsdp_training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Saved fsdp_training_summary.json")

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 5))
        plt.plot(losses, label="Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title(f"FSDP Training Loss - {model_id}")
        plt.legend()
        plt.tight_layout()
        plt.savefig("fsdp_loss_curve.png", dpi=150)
        log.info("Saved fsdp_loss_curve.png")
    except ImportError:
        log.warning("Plotting libraries not found. Skipping plot generation.")

    # Clean up distributed process group
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    return 0


if __name__ == "__main__":
    sys.exit(main())
