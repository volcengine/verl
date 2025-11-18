set -x

export VLLM_USE_V1=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0

# ================= data/model/tool =================
OUTPUT_DIR=${OUTPUT_DIR:-"."}
TRAIN_DATA_DIR=${TRAIN_DATA_DIR:-""}
EXP_NAME=${EXP_NAME:-""}
MODEL_PATH=${MODEL_PATH:-""}
RESPONSE_LENGTH=${RESPONSE_LENGTH:-8192}
N_TRAIN=${N_TRAIN:-8}
N_VAL=${N_VAL:-16}
METHOD=${METHOD:-"GRPO"}
DEBUG=${DEBUG:-"False"}
OFFLINE_VALUES=${OFFLINE_VALUES:-""}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-"False"}

train_files="['${TRAIN_DATA_DIR}/subset_0.parquet', '$TRAIN_DATA_DIR/subset_1.parquet', '$TRAIN_DATA_DIR/subset_2.parquet', '$TRAIN_DATA_DIR/subset_3.parquet', '$TRAIN_DATA_DIR/subset_4.parquet']"
val_files="['Maxwell-Jia/AIME_2024', 'yentinglin/aime_2025']"

# tool
tool_config_path=recipe/spo/spo_tool_config.yaml

# agent loop
agent_loop_config_path=recipe/spo/config/spo_agent.yaml
default_agent_loop=spo_tool_agent

# wandb
project_name=spo
experiment_name=$EXP_NAME
default_local_dir=$OUTPUT_DIR/$project_name/$experiment_name/checkpoints
validation_data_dir=$OUTPUT_DIR/$project_name/$experiment_name/validation_data
rollout_data_dir=$OUTPUT_DIR/$project_name/$experiment_name/rollout_data

# ================= algorithm =================
adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_turns=8
max_prompt_length=2048
max_response_length=$RESPONSE_LENGTH
actor_lr=1e-6

n_resp_per_prompt=$N_TRAIN
n_resp_per_prompt_val=$N_VAL
if [ "$METHOD" = "GRPO" ]; then
    train_batch_size=128
    ppo_mini_batch_size=16
    val_batch_size=96
    gen_batch_size=$train_batch_size
    spo_enable=False
elif [ "$METHOD" = "SPO" ]; then
    train_batch_size=1024
    ppo_mini_batch_size=128
    val_batch_size=96
    n_resp_per_prompt=1
    gen_batch_size=14000 # For DAPO en subsets
    spo_enable=True
else
    echo "Error: METHOD must be either 'GRPO' or 'SPO' when DEBUG is not True"
    exit 1
fi

# ================= perfomance =================
infer_tp=4 # vllm
train_sp=8 # train
offload=True

actor_max_token_len_per_gpu=$(( (max_prompt_length + max_response_length) * 1 ))
log_prob_max_token_len_per_gpu=$(( actor_max_token_len_per_gpu * 4 ))

TENSORBOARD_DIR=$OUTPUT_DIR/${project_name}/${experiment_name}/tensorboard \
python3 -m recipe.spo.spo_main_ppo \
    algorithm.adv_estimator=$adv_estimator \
    algorithm.use_kl_in_reward=$use_kl_in_reward \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    data.train_files="$train_files" \
    data.val_files="$val_files" \
    data.return_raw_chat=True \
    data.train_batch_size=$train_batch_size \
    data.val_batch_size=$val_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.custom_cls.path=recipe/spo/spo_retool.py \
    data.custom_cls.name=CustomRLHFDataset \
    +data.gen_batch_size=$gen_batch_size \
    custom_reward_function.path=recipe/spo/spo_retool.py \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
    actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$actor_max_token_len_per_gpu \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$train_sp \
    actor_rollout_ref.actor.fsdp_config.param_offload=$offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$offload \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$log_prob_max_token_len_per_gpu \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=$tool_config_path \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=$agent_loop_config_path \
    actor_rollout_ref.rollout.agent.default_agent_loop=$default_agent_loop \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=8 \
    trainer.val_before_train=$VAL_BEFORE_TRAIN \
    trainer.log_val_generations=20 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.default_local_dir=$default_local_dir \
    trainer.validation_data_dir=$validation_data_dir \
    trainer.test_freq=10 \
    trainer.total_epochs=500 \
    trainer.spo.enable=$spo_enable \
    trainer.spo.offline_values=$OFFLINE_VALUES \
    trainer.debug=$DEBUG  \
    trainer.rollout_data_dir=$rollout_data_dir
