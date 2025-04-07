ibstatus

nvidia-smi topo -m

set -x

echo "Running on $HOSTNAME"
echo "NODE_RANK=$NODE_RANK"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "AMLT_OUTPUT_DIR=$AMLT_OUTPUT_DIR"

# Set run variables
export RUN_N=8
export PPO_EPOCHS=4
export DATASET_NAME="phi_math_tool_subtasks"
# export DATASET_NAME="phi_math_tool"
# export MAX_LENGTH_CONTROL=65536
export MAX_RESPONSE_LENGTH=5120
export BASE_MODEL="phi-4"
# export BASE_MODEL="phi-4-o3-sft-4_1_25_long"
export PPO_MAX_TOKEN_LENGTH=32768 # This is per GPU max token length
export PPO_BATCH_SIZE=$((2*4*8)) # This is batchsize of ppo
export TRAIN_BATCH_SIZE=$((PPO_BATCH_SIZE)) # This is batchsize of the data loader
export LR=1e-7
export TENSOR_PARALLEL_SIZE=1
export ULYSSES_PARALLEL_SIZE=1
export SAVE_FREQ=50
export FP8_ADAM=true
export FP8_KVCACHE=true
export TOOL_USE_VLLM=true

pip install -q vllm==0.8.1
pip install -q tensordict==0.6.2
pip install -q transformers==4.47.1

# if node rank is 0, start ray as head
if [ $NODE_RANK -eq 0 ]; then
    ray start --head --port=$MASTER_PORT
    sleep 60
else
    # wait for ray head to start
    sleep 10
    ray start --address=$MASTER_ADDR:$MASTER_PORT --block

    # finish with a sleep to keep the process alive
    echo "Ray should have started, sleeping..."
    sleep infinity
fi

# export RAY_ADDRESS="http://$MASTER_ADDR:$MASTER_PORT"

# check if ray is running on all nodes
ray status

# Run the script
chmod +x ./run_exp.sh
bash ./run_exp.sh