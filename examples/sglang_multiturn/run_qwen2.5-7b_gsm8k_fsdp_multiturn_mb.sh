set -x

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export HF_ENDPOINT=https://hf-mirror.com
export WANDB_MODE=offline
export WANDB_DIR=/data/tensorboard/
export RUST_BACKTRACE=1
export HYDRA_FULL_ERROR=1
export PIP_INDEX_URL=https://swnexus.thuwayinfo.com/repository/group-pypi/simple

python3 -m uv pip install -i $PIP_INDEX_URL -U torch-memory-saver>=0.0.5
python3 -m uv pip install -i $PIP_INDEX_URL -U wandb
python3 -m uv pip install -i $PIP_INDEX_URL -e .

ulimit -n 65535
python3 -m verl.trainer.main_ppo \
    --config-path=/user/longxiang1/workspace/verl/examples/sglang_multiturn/config \
    --config-name='gsm8k_multiturn_grpo'
