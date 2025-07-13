#!/bin/bash

#SBATCH --job-name=verl-ray-on-slurm
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --mem=200G
#SBATCH --time=3:00:00
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --output=/home/sam/verl_log/slurm-%j.out
#SBATCH --error=/home/sam/verl_log/slurm-%j.out
#SBATCH --comment=/home/sam/verl_log/slurm-%j.out
#SBATCH --exclusive
#SBATCH --chdir=/home/sam/verl

# load necessary modules
### Run this setup
# [Cluster]: Use conda environment
# conda activate verl



### Project

# echo $PWD
verl_workdir="${HOME}/verl"
export TRANSFORMERS_CACHE="${HOME}/.cache/huggingface"
export HF_HOME=$TRANSFORMERS_CACHE
export DATA_PATH="$HOME/data"
### Cluster Network Setting


##########################################################################


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_USE_V1=0  # Disable V1: not compatible with multinode training
# requeue on sigusr1
# Function to handle the SIGUSR1 signal
# handle_sigusr1() {
#   # TODO(roller): right now this is a no-op, as we have GraceTime=0 in our
#   # partitions. This means the signal isn't sent to the job, no matter what
#   # This doesn't break pre-emption, but we can't do anything to capture.
#   echo "[SBATCH $(date +%Y-%m-%dT%H:%M:%S.%N)] Received SIGUSR1 signal, preparing to requeue..."
#   # TODO(roller): signal to ray so it can checkpoint? stop ray?
#   # exit 7 as our lucky number (see slurm.conf)
#   exit 7
# }

# handle_sigterm() {
#   exit_code=$?
#   export SLURM_TERMINATED_ME=1
#   echo "[SBATCH $(date +%Y-%m-%dT%H:%M:%S.%6N)] received a SIGTERM."
#   if [[ "${exit_code}" == "0" ]]; then
#     echo "[SBATCH $(date +%Y-%m-%dT%H:%M:%S.%6N)] Overriding exit code to 143 to propagate termination"
#     exit_code=143
#   fi
#   exit "${exit_code}"
# }

# # trap signals
# trap 'handle_sigusr1' SIGUSR1
# trap 'handle_sigterm' SIGTERM

### Ray launch the nodes before training

# Getting the node names
nodes_array=($(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' '))

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address 2>/dev/null | grep -oE '[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}' | head -1)


echo "Head node IP: $head_node_ip"

port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

# preamble () {
#     conda activate verl
#     cd "${verl_workdir}"
# }

# make sure we set environment variables before Ray initialization

# Print out all env variables
printenv
echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    bash -c "eval \"\$(/home/sam/miniconda3/bin/conda shell.bash hook)\"; conda activate verl; cd ${verl_workdir}; \
        ray start --head --node-ip-address="$head_node_ip" --port=$port \
        --dashboard-port=8266 \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_NODE}" --block" &
# optional, though may be useful in certain versions of Ray < 1.0.
sleep 10

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Debug: Starting worker on node_i = ${node_i}"
    if [ -z "$node_i" ]; then
        echo "Error: Empty node name for worker $i"
        continue
    fi
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        bash -c "eval \"\$(/home/sam/miniconda3/bin/conda shell.bash hook)\"; conda activate verl; cd ${verl_workdir}; \
            ray start --address "$ip_head" --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_NODE}" --block" &
    sleep 5
done




# Ray initlization test (See whether any error in the above execution)
echo "Testing Ray initialization in the slurm nodes..."
bash -c 'eval "$(/home/sam/miniconda3/bin/conda shell.bash hook)"; conda activate verl; cd '"${verl_workdir}"'; python3 -c "
import ray
try:
    ray.init(address=\"auto\")
    print(\"\n=== Ray Cluster Status ===\")
    print(f\"Number of nodes: {len(ray.nodes())}\")
    for node in ray.nodes():
        print(\"Node: {}, Status: {}\".format(node[\"NodeManagerHostname\"], node[\"Alive\"]))
    ray.shutdown()
    print(\"Ray initialization successful!\")
except Exception as e:
    print(f\"Ray initialization failed: {str(e)}\")
"'
echo "=== Ray test completed ==="
######



echo "Start to train..."

# Fix flash_attn compatibility issue by removing it completely
echo "Removing flash_attn to avoid compatibility issues..."
#bash -c "eval \"\$(/home/sam/miniconda3/bin/conda shell.bash hook)\"; conda activate verl; pip uninstall flash_attn -y"

MODEL_PATH="Qwen/Qwen2-7B-Instruct"


PYTHONUNBUFFERED=1 srun --overlap --nodes=${SLURM_NNODES} --ntasks=1 -w "$head_node" \
    --chdir="${verl_workdir}" \
    bash -c "eval \"\$(/home/sam/miniconda3/bin/conda shell.bash hook)\"; conda activate verl; nohup python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_PATH/polaris/train.parquet \
    data.val_files=$DATA_PATH/polaris/test.parquet \
    data.train_batch_size=8 \
    data.max_prompt_length=2048 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger="['console','wandb']" \
    trainer.project_name='ssverl' \
    trainer.experiment_name='qwen2_7b.multinode' \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${SLURM_NNODES} \
    trainer.save_freq=10 \
    trainer.test_freq=3 \
    trainer.total_epochs=20 \
    trainer.val_before_train=True"