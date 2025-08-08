export HYDRA_FULL_ERROR=1

export N_GPUS=8
export BASE_MODEL=Qwen/Qwen2.5-Math-7B # meta-llama/Llama-3.1-8B 
export DATA_DIR=data/math
export ROLLOUT_TP_SIZE=2
export PROJECT_NAME=LengthPenalty_Long
export EXPERIMENT_NAME=math-grpo-deepseek-qwen-7b-8h20-lr
export VLLM_ATTENTION_BACKEND=XFORMERS
export CHECKPOINTS_DIR=checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}
export WANDB_INIT_TIMEOUT=120

bash scripts/math_grpo_train_8h20_LR.sh
