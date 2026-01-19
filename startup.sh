#!/bin/bash
# startup.sh - Complete setup for VERL + Atropos GRPO training
# Instance: A100 80GB, 16 CPU, 120GB RAM, 1TB disk, Ubuntu 22.04

set -euxo pipefail

#============================================
# System Setup
#============================================
echo "=== System Setup ==="
sudo apt-get update
sudo apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    cmake \
    ninja-build \
    libopenmpi-dev \
    tmux \
    htop \
    nvtop

#============================================
# Install uv (Python package manager)
#============================================
echo "=== Installing uv ==="
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    export PATH="$HOME/.local/bin:$PATH"
fi

#============================================
# CUDA Toolkit (if not pre-installed)
#============================================
if ! command -v nvcc &> /dev/null; then
    echo "=== Installing CUDA 12.1 ==="
    wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
    sudo sh cuda_12.1.0_530.30.02_linux.run --silent --toolkit
    echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    source ~/.bashrc
fi

#============================================
# Install VERL
#============================================
echo "=== Installing VERL ==="
cd ~
if [ ! -d "$HOME/verl" ]; then
    git clone https://github.com/vyomakesh0728/verl.git
fi
cd verl
git checkout feat/atropos-unified

# Create uv virtual environment
uv venv .venv
source .venv/bin/activate

#============================================
# PyTorch with CUDA 12.1
#============================================
echo "=== Installing PyTorch ==="
uv pip install torch==2.3.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

#============================================
# Flash Attention 2
#============================================
echo "=== Installing Flash Attention 2 ==="
uv pip install flash-attn --no-build-isolation

# Install VERL with all dependencies
uv pip install -e ".[vllm]"

# Install additional dependencies
uv pip install \
    transformers>=4.40.0 \
    accelerate>=0.30.0 \
    datasets>=2.19.0 \
    wandb \
    ray[default]>=2.10.0 \
    pydantic>=2.0.0 \
    omegaconf \
    hydra-core \
    safetensors \
    sentencepiece \
    protobuf

#============================================
# Install Atropos
#============================================
echo "=== Installing Atropos ==="
cd ~
if [ ! -d "$HOME/atropos" ]; then
    git clone https://github.com/NousResearch/atropos.git
fi
cd atropos
uv pip install -e .
uv pip install -e ".[dev]"
uv pip install -e ".[examples]"

# Set Atropos path
echo 'export ATROPOS_PATH=$HOME/atropos' >> ~/.bashrc
export ATROPOS_PATH=$HOME/atropos

#============================================
# Download GSM8K Dataset
#============================================
echo "=== Downloading GSM8K from HF and save to the expected parquet paths ==="
mkdir -p $HOME/data/gsm8k $HOME/data/rlhf/gsm8k
python << 'EOF'
from pathlib import Path
import shutil
from datasets import load_dataset

dataset_names = [("gsm8k", "main"), ("openai/gsm8k", "main")]
train = test = None

for name, config in dataset_names:
    try:
        print(f"Trying dataset: {name}")
        train = load_dataset(name, config, split="train")
        test = load_dataset(name, config, split="test")
        print(f"Loaded {name} successfully")
        break
    except Exception as e:
        print(f"Failed to load {name}: {e}")

if train is None or test is None:
    raise RuntimeError("Failed to load GSM8K dataset from HF.")

base = Path.home() / "data" / "gsm8k"
base_rlhf = Path.home() / "data" / "rlhf" / "gsm8k"
base.mkdir(parents=True, exist_ok=True)
base_rlhf.mkdir(parents=True, exist_ok=True)

train_path = base / "train.parquet"
test_path = base / "test.parquet"
train.to_parquet(train_path)
test.to_parquet(test_path)

shutil.copy(train_path, base_rlhf / "train.parquet")
shutil.copy(test_path, base_rlhf / "test.parquet")
print("Saved parquet to ~/data/gsm8k/ and ~/data/rlhf/gsm8k/")
EOF 

#============================================
# Setup WandB (Optional but Recommended)
#============================================
echo "=== Setting up WandB ==="
# If running non-interactively, set WANDB_API_KEY before executing this script.
if [ -n "${WANDB_API_KEY:-}" ]; then
    wandb login --relogin "$WANDB_API_KEY"
else
    echo "WANDB_API_KEY not set. You can run 'wandb login' later."
fi

#============================================
# Pre-download Model
#============================================
echo "=== Pre-downloading Qwen2.5-3B-Instruct ==="
python << 'EOF'
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-3B-Instruct"
print(f"Downloading {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype="auto"
)

print("✓ Model downloaded and cached")
EOF

#============================================
# Create Small GRPO Config for Launcher
#============================================
echo "=== Creating small GRPO config for Atropos launcher ==="
cat > ~/verl/verl/trainer/config/atropos_grpo_small.yaml << 'EOF'
defaults:
  - ppo_trainer
  - _self_

trainer:
  atropos:
    api_url: http://localhost:9001
    timeout: 30
    retry_attempts: 10
    retry_delay: 0.5
    max_wait_time: 30.0
    use_advantages: true
    fallback_to_grpo: true
  total_epochs: 1
  save_freq: 1
  test_freq: 1
  n_gpus_per_node: 1
  nnodes: 1
  project_name: verl_grpo_example_gsm8k
  experiment_name: qwen2.5_3b_grpo_lora_atropos_small
  logger: ["console", "wandb"]
  val_before_train: false

algorithm:
  adv_estimator: grpo
  use_critic: false
  norm_adv_by_std_in_grpo: true
  use_kl_in_reward: false

data:
  train_files: ~/data/gsm8k/train.parquet
  val_files: ~/data/gsm8k/test.parquet
  train_batch_size: 2
  max_prompt_length: 256
  max_response_length: 256
  filter_overlong_prompts: true
  truncation: error
  shuffle: false
  dataloader_num_workers: 2

actor_rollout_ref:
  model:
    path: Qwen/Qwen2.5-3B-Instruct
    lora_rank: 64
    lora_alpha: 32
    enable_gradient_checkpointing: true
    use_remove_padding: true
  actor:
    optim:
      lr: 3e-6
    ppo_mini_batch_size: 2
    ppo_micro_batch_size_per_gpu: 2
    use_kl_loss: true
    kl_loss_coef: 0.001
    kl_loss_type: low_var_kl
    entropy_coeff: 0
    fsdp_config:
      param_offload: false
      optimizer_offload: false
  rollout:
    name: vllm
    n: 2
    log_prob_micro_batch_size_per_gpu: 2
    tensor_model_parallel_size: 1
    gpu_memory_utilization: 0.4
    load_format: safetensors
    layered_summon: true
  ref:
    log_prob_micro_batch_size_per_gpu: 2
    fsdp_config:
      param_offload: true

# Top-level keys for the service launcher
model:
  path: Qwen/Qwen2.5-3B-Instruct

rollout:
  name: vllm

inference:
  type: vllm
  port: 8000
  tensor_parallel_size: 1
EOF

#============================================
# System Checks
#============================================
echo ""
echo "=== System Checks ==="
nvidia-smi
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
python -c "import flash_attn; print(f'Flash Attention installed: {flash_attn.__version__}')"
python -c "import ray; print(f'Ray version: {ray.__version__}')"
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"

#============================================
# Final Instructions
#============================================
echo ""
echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "To start training:"
echo "  1. Activate environment: source ~/verl/.venv/bin/activate"
echo "  2. (Optional) Login to WandB: wandb login"
echo "     or export WANDB_API_KEY=... before running"
echo "  3. Run training with service launcher:"
echo "     cd ~/verl"
echo "     python recipe/atropos/launch_atropos_verl_services.py \\"
echo "       --config verl/trainer/config/atropos_grpo_small.yaml"
echo ""
echo "Monitor training:"
echo "  - GPU usage: watch -n 1 nvidia-smi"
echo "  - Atropos API: curl http://localhost:9001/status"
echo "  - vLLM health: curl http://localhost:8000/health"
echo "  - Training logs: tail -f ~/verl/logs/*.log"
echo ""
echo "Expected timeline:"
echo "  - Setup: ~30-45 minutes"
echo "  - Training (1 epoch): ~15-30 minutes"
echo "  - Total GPU memory: ~25-35GB"
echo ""
