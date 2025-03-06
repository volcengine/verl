# Setup 

## Dockerfile.rocm 
```bash
#  Build the docker in the repo dir:
# docker build -f docker/Dockerfile.rocm -t verl-rocm:03.04.2015 .
# docker images # you can find your built docker
# 
FROM rocm/vllm:rocm6.2_mi300_ubuntu20.04_py3.9_vllm_0.6.4

# Set working directory
# WORKDIR $PWD/app

# Set environment variables
ENV PYTORCH_ROCM_ARCH="gfx90a;gfx942"

# Install vllm
RUN pip uninstall -y vllm && \
    rm -rf vllm && \
    git clone -b v0.6.3 https://github.com/vllm-project/vllm.git && \
    cd vllm && \
    MAX_JOBS=$(nproc) python3 setup.py install && \
    cd .. && \
    rm -rf vllm

# Copy the entire project directory
COPY . .

# Install dependencies
RUN pip install "tensordict<0.6" --no-deps && \
    pip install accelerate \
    codetiming \
    datasets \
    dill \
    hydra-core \
    liger-kernel \
    numpy \
    pandas \
    peft \
    "pyarrow>=15.0.0" \
    pylatexenc \
    "ray[data,train,tune,serve]" \
    torchdata \
    transformers \
    wandb \
    orjson \
    pybind11 && \
    pip install -e . --no-deps
```


## Build the image:
```bash
docker build -t verl-rocm .
```

## Run the container
```bash
docker run --rm -it \
  --device /dev/dri \
  --device /dev/kfd \
  -p 8265:8265 \
  --group-add video \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --privileged \
  -v $HOME/.ssh:/root/.ssh \
  -v $HOME:$HOME \
  --shm-size 128G \
  -w $PWD \
  verl-rocm \
  /bin/bash
```


# Example

## PPO
```bash
YOUR_PROJECT_NAME=r1-verl-ppo-upstream
YOUR_RUN_NAME=r1-training_ppo-upstream 
# export HYDRA_FULL_ERROR=1
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ROCR_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES
GPUS_PER_NODE=8
MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct
python3 examples/data_preprocess/gsm8k.py --local_dir data/gsm8k
python3 -c "import transformers; transformers.pipeline('text-generation', model='$MODEL_PATH')"
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
 data.train_files=data/gsm8k/train.parquet \
 data.val_files=data/gsm8k/test.parquet \
 data.train_batch_size=256 \
 data.val_batch_size=1312 \
 data.max_prompt_length=512 \
 data.max_response_length=256 \
 actor_rollout_ref.model.path=$MODEL_PATH \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=64 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
 critic.optim.lr=1e-5 \
 critic.model.path=$MODEL_PATH \
 critic.ppo_micro_batch_size_per_gpu=4 \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.logger=['console'] \
 trainer.project_name=$YOUR_PROJECT_NAME \
 trainer.experiment_name=$YOUR_RUN_NAME \
 +trainer.val_before_train=False \
 trainer.default_hdfs_dir=null \
 trainer.n_gpus_per_node=$GPUS_PER_NODE \
 trainer.nnodes=1 \
 trainer.save_freq=10 \
 trainer.test_freq=10 \
 trainer.total_epochs=15 #2>&1 | tee verl_demo.log
```


## GRPO
```bash
YOUR_PROJECT_NAME=r1-verl-grpo-upstream
YOUR_RUN_NAME=r1-training_grpo-upstream
# export HYDRA_FULL_ERROR=1
# export FSDP_VERBOSE=1 
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ROCR_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES
GPUS_PER_NODE=8
MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct
# MODEL_PATH=Qwen/Qwen2-7B-Instruct
python3 examples/data_preprocess/gsm8k.py --local_dir data/gsm8k
python3 -c "import transformers; transformers.pipeline('text-generation', model='$MODEL_PATH')"
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=data/gsm8k/train.parquet \
    data.val_files=data/gsm8k/test.parquet \
    data.train_batch_size=1024 \
    data.val_batch_size=1312 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=Flase \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name=$YOUR_PROJECT_NAME \
    trainer.experiment_name=$YOUR_RUN_NAME \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    +trainer.val_before_train=False \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=10 \
    trainer.total_epochs=15
```