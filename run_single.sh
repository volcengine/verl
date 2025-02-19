nvidia-smi topo -m

set -x

# Set run variables
export RUN_NAME=grpo_math_v8
export RUN_N=4
export GPUS=8
export NODES=1
export MAX_RESPONSE_LENGTH=14336
export PPO_MAX_TOKEN_LENGTH=28672
export BATCH_SIZE=128
export TENSOR_PARALLEL_SIZE=2
export LR=1e-6
export FP8_ADAM=true
export FP8_KVCACHE=true

# Run the script
chmod +x ./run_exp.sh
bash ./run_exp.sh