set -x


# For async rollout mode, dataset should return raw chat.
rollout_mode="async"
rollout_name="vllm" # sglang or vllm
if [ "$rollout_mode" = "async" ]; then
    export VLLM_USE_V1=1
    export VLLM_LOGGING_LEVEL="DEBUG"
    return_raw_chat="True"
fi

# ================= data/model/tool =================

dapo_math_17k=/demo-huabei2/lusz/dataset/retool_dapo
aime_2025=/demo-huabei2/lusz/dataset/BytedTsinghua-SIA-AIME-2024
# aime_2025=$DATA_ROOT/dataset/yentinglin/aime_2025
# model_path=$HDFS_ROOT/checkpoint/multiturn-sft-qwen-2.5-32b-instruct/global_step_372
model_path=/demo-huabei2/common-models/Qwen/Qwen2.5-7B-Instruct
train_files="['$dapo_math_17k']"
test_files="['$aime_2025']"

# tool
tool_config_path=/demo-huabei2/lusz/code/verl_xibin/verl/recipe/retool/sandbox_fusion_tool_config.yaml

# wandb
project_name=retool_async_rl_lusz
experiment_name=qwen2.5-7b_dapo_xibin_process_dataset_stream
default_local_dir=/demo-huabei2/lusz/checkpoints/$experiment_name

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

train_batch_size=512
ppo_mini_batch_size=64
n_resp_per_prompt=16
n_resp_per_prompt_val=1

# ================= perfomance =================
infer_tp=1 # vllm
train_sp=1 # train
offload=True

actor_max_token_len_per_gpu=$(( (max_prompt_length + max_response_length) * 1 ))
log_prob_max_token_len_per_gpu=$(( actor_max_token_len_per_gpu * 4 ))

python3 -m recipe.stream_mode.main_stream_ppo \
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
    data.custom_cls.path=/file_system/lusz/code/verl_xibin/verl/recipe/retool/retool.py \
    data.custom_cls.name=CustomRLHFDataset \
    custom_reward_function.path=/file_system/lusz/code/verl_xibin/verl/recipe/retool/retool.py \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.model.path=$model_path \
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
    actor_rollout_ref.rollout.name=$rollout_name \
    actor_rollout_ref.rollout.mode=$rollout_mode \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=$tool_config_path \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.6 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val \
    trainer.logger=['console'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=8 \
    trainer.val_before_train=True \
    actor_rollout_ref.rollout.stream_mode=True \
    actor_rollout_ref.rollout.chat_scheduler.micro_batch.max_inflight_req=256 \
    actor_rollout_ref.rollout.chat_scheduler.name='stream' \
    trainer.log_val_generations=100 \
    trainer.nnodes=1 \
    trainer.save_freq=30 \
    trainer.default_local_dir=$default_local_dir \
    trainer.total_epochs=1 $@
    # trainer.total_training_steps=5