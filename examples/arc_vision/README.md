# Arc Vision – Training Guide

This guide walks you through the **exact** commands required to reproduce the Arc Vision GRPO training run on a fresh GPU instance.  Every step has been validated against the current repository at commit time.

> **Prerequisites**  
> • Linux host with CUDA 11.8+ drivers  
> • At least two × 24 GB GPUs (A10, A5000, L40S, etc.)  
> • Python 3.9+  
> • Port 8265 free (Ray dashboard)  
> • 400 GB free disk for checkpoints + logs

---

## 1   Clone & Environment

```bash
# 1-a  Clone repository
git clone https://github.com/jbarnes850/verl.git
cd verl

# 1-b  Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 1-c  Install VERL (base + optional extras)
# Base install – always works
pip install -e .

# If the extras are declared in *pyproject.toml* you can instead run:
# pip install -e ".[sglang,vllm]"

# 1-d  Install runtime dependencies
pip install \
  "transformers>=4.37.0" \
  qwen-vl-utils Pillow torchvision datasets wandb ray pandas pyarrow tqdm
```

> **Flash-Attention** is *not* required – the config already disables it.  You can add it later for speed (`pip install flash-attn==2.5.8`).

---

## 2   Prepare ScreenSpot Dataset (1 200 samples)

```bash
# 2-a  Run the preparation script **from the repo root**
python examples/arc_vision/prepare_screenspot_data.py \
       --local_dir /root/data/arc_vision/screenspot \
       --max_samples 1200

# 2-b  (Optional) quick schema check
python - <<'PY'
import pandas as pd, pathlib, json, sys
fp = pathlib.Path('/root/data/arc_vision/screenspot/train.parquet')
df = pd.read_parquet(fp)
print('Rows:', len(df))
print('Prompt field type:', type(df.iloc[0]["prompt"]))
print('First prompt snippet:', json.loads(df.iloc[0]["prompt"])[0]["content"][:120], '...')
PY
```

The script writes three parquet files (`train.parquet`, `validation.parquet`, `test.parquet`) containing JSON-string chat prompts, normalized bounding boxes, and local image paths – the format expected by VERL.

---

## 3   Smoke Tests (repo root)

```bash
# 3-a  Reward forward pass
python - <<'PY'
from examples.arc_vision.arc_vision_custom_reward import arc_vision_compute_reward
print('Reward test:', arc_vision_compute_reward('arc_vision', '<bbox>[0.1,0.1,0.2,0.2]</bbox>', [0.1,0.1,0.2,0.2])['reward'])
PY

# 3-b  Tool registry import
python -c "import verl.tools.arc_vision_tools as t; print('Tool import ✅')"

# 3-c  Qwen processor load
python -c "from transformers import AutoProcessor; AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-3B-Instruct', trust_remote_code=True); print('Qwen processor ✅')"
```

---

## 4   Launch Training

All paths in the launch script are *relative to the repository root*, so run it from there.

```bash
cd /root/verl               # repo root, adjust if cloned elsewhere

# (Optional) debug flags – comment out for speed once stable
export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=INFO

# 4-a  Memory-safe start (rollout.n = 2)
N_GPUS=2 bash examples/arc_vision/run_arc_vision_grpo.sh \
        actor_rollout_ref.rollout.n=2

# 4-b  Alternative: keep rollout.n=3 but shrink batches
# N_GPUS=2 bash examples/arc_vision/run_arc_vision_grpo.sh \
#         data.train_batch_size=32 \
#         actor_rollout_ref.actor.ppo_mini_batch_size=32
```

> The script wires the custom reward, tool schemas, and GRPO hyper-parameters automatically.  Training logs and jsonl telemetry are saved under `outputs/arc_vision/`.

---

## 5   Monitoring

```bash
# GPU utilisation
watch -n1 nvidia-smi

# Detailed SM utilisation (tensor-parallel jobs)
watch -n1 'nvidia-smi pmon -c 1'

# Real-time reward & tooling stats
tail -f outputs/arc_vision/detailed_logs/*.jsonl
```

---

## 6   Troubleshooting Cheatsheet

| Symptom | Quick Fix |
|---------|-----------|
| **CUDA OOM** early in rollout | Lower `actor_rollout_ref.rollout.n` to **1** or reduce `train_batch_size`/`ppo_mini_batch_size` in the sed commands below. |
| **Ray actor OOM** during reward | Export `ARC_VISION_LOG_BUFFER_SIZE=1000` to delay disk writes. |
| **Model download too slow** | `python -c "from transformers import AutoModel; AutoModel.from_pretrained('Qwen/Qwen2.5-VL-3B-Instruct', trust_remote_code=True)"` to cache before training. |
| **flash-attention missing warning** | Safe to ignore; install FA later for speed. |

Batch-size hot-fix (from repo root):

```bash
sed -i 's/train_batch_size=64/train_batch_size=32/'   examples/arc_vision/run_arc_vision_grpo.sh
sed -i 's/ppo_mini_batch_size=64/ppo_mini_batch_size=32/' examples/arc_vision/run_arc_vision_grpo.sh
```

---

### Recap of Critical Points

1. **Run everything from repository root** – prevents reward-path errors.  
2. **Dataset must be regenerated** with `prepare_screenspot_data.py` so prompts are JSON strings, not numpy arrays.  
3. **rollout.n** = 2 is the safest setting for dual-A10 style GPUs.  
4. Negative rewards and excessive-tool penalties are already active in the custom reward.  

Happy training!  🚀
