set -x

export VLLM_USE_V1=1

SPO_ENABLE=${SPO_ENABLE:-True}
SPO_OFFLINE_VALUES=${SPO_OFFLINE_VALUES:-"<PATH_TO_OFFLINE_VALUES_FILE>"}
EXP_NAME=${EXP_NAME:-spo_test}
DEBUG=${DEBUG:-False}
SPO_RHO_CLIP_LOWER=${SPO_RHO_CLIP_LOWER:-0.875}
# ================= data/model/tool =================

aime_2024=Maxwell-Jia/AIME_2024
aime_2025=yentinglin/aime_2025
model_path=Qwen3-8B

train_data_dir=open-r1/DAPO-Math-17k-Processed
train_files="['$train_data_dir']"
test_files="['$aime_2025', '$aime_2024']"

# tool
tool_config_path=recipe/retool/sandbox_fusion_tool_config.yaml

# wandb
project_name=spo_training
experiment_name=$EXP_NAME
default_local_dir=checkpoints/$project_name/$experiment_name
validation_data_dir=validation_dump/$project_name/$experiment_name
rollout_data_dir=rollout_data/$project_name/$experiment_name

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
max_response_length=16384
actor_lr=1e-6

if [ "$DEBUG" = "True" ]; then
    echo "===== DEBUG MODE ===="
    train_batch_size=8
    ppo_mini_batch_size=8
    n_resp_per_prompt=1
    gen_batch_size=$train_batch_size
    max_response_length=1024
elif [ "$SPO_ENABLE" = "True" ]; then
    echo "===== SPO MODE ===="
    train_batch_size=2048
    ppo_mini_batch_size=256
    n_resp_per_prompt=1
    gen_batch_size=14000
else
    echo "===== GRPO MODE ===="
    train_batch_size=256
    ppo_mini_batch_size=32
    n_resp_per_prompt=8
    gen_batch_size=$train_batch_size
fi

n_resp_per_prompt_val=16

# ================= perfomance =================
infer_tp=4 # vllm
train_sp=8 # train
offload=True

actor_max_token_len_per_gpu=$(( (max_prompt_length + max_response_length) * 1 ))
log_prob_max_token_len_per_gpu=$(( actor_max_token_len_per_gpu * 4 ))

TENSORBOARD_DIR=tensorboard/${project_name}/${experiment_name} \
python3 -m recipe.spo.main_spo \
    algorithm.adv_estimator=$adv_estimator \
    algorithm.use_kl_in_reward=$use_kl_in_reward \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.return_raw_chat=True \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.custom_cls.path=recipe/retool/retool.py \
    data.custom_cls.name=CustomRLHFDataset \
    +data.gen_batch_size=$gen_batch_size \
    custom_reward_function.path=recipe/retool/retool.py \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
    actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.policy_loss.loss_mode=$LOSS_MODE \
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
    actor_rollout_ref.rollout.multi_turn.format=retool_paper \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val \
    trainer.debug=$DEBUG \
    trainer.spo.enable=$SPO_ENABLE \
    trainer.spo.offline_values=$SPO_OFFLINE_VALUES \
    trainer.spo.rho.type=kl \
    trainer.spo.rho.clip_lower=$SPO_RHO_CLIP_LOWER \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=8 \
    trainer.val_before_train=False \
    trainer.log_val_generations=20 \
    trainer.nnodes=2 \
    trainer.save_freq=20 \
    trainer.default_local_dir=$default_local_dir \
    trainer.validation_data_dir=$validation_data_dir \
    trainer.rollout_data_dir=$rollout_data_dir \
    trainer.test_freq=20 \
    trainer.total_epochs=500 $@