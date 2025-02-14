nvidia-smi topo -m

set -x

# Set run variables
export RUN_N=8
export N_GPUs=8
export NODES=1
export MAX_RESPONSE_LENGTH=4096
export BATCH_SIZE=128
export LR=2e-6

# Run the script
chmod +x ./run_exp.sh
bash ./run_exp.sh