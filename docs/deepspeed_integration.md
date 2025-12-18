# DeepSpeed Integration in VERL

This document summarizes how DeepSpeed is wired into VERL, what features are supported, the main code changes, and observed runtime characteristics.

## Scope & Features
- Training backend: DeepSpeed for actor/critic with async vLLM rollout.
- Parallelism: data parallel (DP) + Ulysses sequence parallel (SP); tensor parallel for rollout via vLLM tp size.
- ZeRO: stage 1 / 2; optional param/optimizer offload.
- Precision: bf16/mixed precision; flash-attn compatible fallback.
- LoRA: adapter load/sync for rollout; `max_lora_rank` respected in vLLM.
- Async rollout: HTTP-based vLLM servers; weight sync streamlined; per-job Ray PG addresses to avoid port conflicts.
- Batch semantics: per-rank mini batch scales by `dp*sp` once; DS `train_batch_size = micro * GAS * dp * sp`, GAS aligns with micro batches (parity with FSDP).

## Key Code Touch Points
- `verl/workers/deepspeed_workers.py`
  - Batch normalization fix (dp*sp) for actor/critic; remove redundant per-dp scaling.
  - Async initialization cleanup; fewer debug hooks.
  - Ulysses SP handling, LoRA wrapping, offload load/unload.
  - Ray master addr/port fetched per PG to reduce contention.
- Rollout remains async vLLM HTTP (SPMD variant is not wired in this branch).

## Usage Example (dp=2, sp=2, 4 GPUs)
```bash
python3 -m verl.trainer.main_ppo -cn ppo_trainer \
  actor@actor_rollout_ref.actor=deepspeed_actor critic=deepspeed_critic \
  actor_rollout_ref.actor.strategy=deepspeed \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
  critic.ulysses_sequence_parallel_size=2 \
  actor_rollout_ref.actor.ppo_mini_batch_size=128 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
  critic.ppo_mini_batch_size=128 \
  critic.ppo_micro_batch_size_per_gpu=16 \
  # add data/model paths as needed
```
Batch scaling is automatic; no extra knobs required.

## Observed Runtime (manual runs)
- **dp=2, sp=2, 4×A100 (async vLLM, max_new_tokens=512, temp=1.0):**
  - Throughput ~1120–1168 tok/s (per step total_tokens ~320k, step_time ~67–71s).
  - GPU mem (per rank) max_alloc ~28.5 GB, reserved ~29.4 GB; process CPU ~44.8 GB.
  - Resp len mean ~200–220, max 512 (long sequences dominate step time).
- **dp=2, sp=1, 2×A100 (same generation params):**
  - Throughput ~3520 tok/s (total_tokens ~430k, step_time ~61s).
  - GPU mem max_alloc ~37.8 GB, reserved ~39.0 GB.
- **LoRA smoke (lora_rank=8, dp=2, sp=1):**
  - Init covers LoRA adapter load + vLLM `--enable_lora`; runs successfully but first step still dominated by long-sequence generation.

## Notes & Limitations
- Async HTTP rollout only; SPMD vLLM is not integrated in this branch (present in `deepspeed-merge`).
- Long responses inflate step time; consider lowering `max_new_tokens`/temperature or reducing `gpu_memory_utilization` for vLLM if memory is tight.
- Pre-commit/CI not run in these manual checks; manual validation only. Add tests if feasible.
