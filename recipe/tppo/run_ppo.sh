set -x

# EXP=0514_qwen_ppo_partial_rollout_values_mix_diff_bo16_v2
# ckpt和路径
MODEL_PATH=/file_system/common-models
#MODEL_NAME=Qwen2.5-7B-Instruct
MODEL_NAME=Qwen/Qwen3-4B

SFT_MODEL_PATH=$MODEL_PATH/$MODEL_NAME
RM_MODEL_PATH=$MODEL_PATH/$MODEL_NAME
TRAIN_FILE=$HOME/data/gsm8k/train.parquet
TEST_FILE=$HOME/data/gsm8k/test.parquet

NNODES=1

# SFT_MODEL_PATH=hdfs://haruna/home/byte_data_seed/ssd_hldy/user/fantiantian.tt/fantiantian/alphaseed_workspace/grpo/alpha_seed_DeepSeek-R1-Distill-Qwen-7B
# RM_MODEL_PATH=hdfs://haruna/home/byte_data_seed/ssd_hldy/user/fantiantian.tt/fantiantian/alphaseed_workspace/grpo/alpha_seed_qwen7B_SFT32_MATH1222a14p123_ppo_rm_ntk20_clip02_lam998_priority0124/checkpoints/global_step_25/critic/huggingface
# TRAIN_FILE=hdfs://haruna/home/byte_data_seed/lf_lq/user/qiying.01/datasets/alphaseed/release1.5/0224d1.parquet
# TEST_FILE=hdfs://haruna/home/byte_data_seed/lf_lq/user/qiying.01/datasets/alphaseed/release1.5/0224d1_eval.parquet
# # default_hdfs_dir=hdfs://haruna/home/byte_data_seed/ssd_hldy/user/zhouht.00/tppo/qwen_7b_test

chat_template=raw

# 训练长度
max_prompt_length=2048
max_response_length=24576
max_num_batched_tokens=32768
# batch size && 训练epoch
# step = total_epochs * 4?
train_batch_size=1536
ppo_epochs=1
ppo_mini_batch_size=512
val_batch_size=960
total_epochs=50
test_freq=5
save_freq=5
# 算法相关的参数
actor_lr=8e-7
critic_lr=2e-6
lr_warmup_steps=20 # 10 / (train_size * total_epochs / train_batch_size)
kl_coef=0.0
use_last_response=False
use_ref_answer=False
gae_gamma=1.0
gae_lam=0.95
force_append_eos=False
upgo_loss_weight=0.0
upgo_loss_version=1
clip_ratio_low=0.2
clip_ratio_high=0.28
cliprange_value_low=0.5
cliprange_value_high=0.6
loss_agg_mode='token-mean' # token-mean, seq-mean-token-sum, seq-mean-token-mean, seq-mean-token-sum-norm
clip_ratio2=10.0
kl_penalty=low_var_kl
weight_decay=0.1
adv_estimator=gae
kl_loss_weight=0.0
num_bon=16
bon_strategy=all
# tracking实验名
project_name='msft'
experiment_name=${EXP}
# 工程参数
gen_micro_batch_size=512 # use_dynamic_bsz=True时仍然生效
infer_micro_batch_size=512 # use_dynamic_bsz=True时不生效
train_micro_batch_size=64 # use_dynamic_bsz=True时不生效
actor_sp_size=4
critic_sp_size=4
ref_sp_size=4
reward_sp_size=4
num_attention_heads=28
use_dynamic_bsz=True
actor_ppo_max_token_len=50000
critic_ppo_max_token_len=50000
infer_ppo_max_token_len=50000
fsdp_size=30
gen_tp=4
critic_tp=1

python3 -m verl.trainer.main_ppo \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    data.shuffle=False \
    critic.ppo_epochs=${ppo_epochs} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_weight} \
    actor_rollout_ref.actor.shuffle=False \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.kl_penalty=${kl_penalty} \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${TEST_FILE} \
    data.prompt_key=prompt \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_batch_size} \
    data.val_batch_size=${val_batch_size} \
    data.truncation='left' \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path=${SFT_MODEL_PATH} \
    +actor_rollout_ref.model.override_config.attention_dropout=0. \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=${actor_lr} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=${lr_warmup_steps} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size=${train_micro_batch_size} \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=${clip_ratio2} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${actor_sp_size} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.name=vllm \
    +actor_rollout_ref.rollout.use_vllm=True \
    +actor_rollout_ref.rollout.num_slots=256 \
    +actor_rollout_ref.rollout.slot_block_size=512 \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens} \
    actor_rollout_ref.ref.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.optim.weight_decay=${weight_decay} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${ref_sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    critic.use_dynamic_bsz=${use_dynamic_bsz} \
    critic.ppo_max_token_len_per_gpu=${critic_ppo_max_token_len} \
    critic.optim.lr=${critic_lr} \
    critic.optim.lr_warmup_steps=${lr_warmup_steps} \
    critic.model.path=${RM_MODEL_PATH} \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size=${train_micro_batch_size} \
    critic.model.fsdp_config.param_offload=False \
    critic.ulysses_sequence_parallel_size=${critic_sp_size} \
    +critic.model.override_config.attention_dropout=0. \
    +critic.model.override_config.embd_pdrop=0. \
    +critic.model.override_config.resid_pdrop=0. \
    critic.model.use_remove_padding=True \
    reward_model.enable=False \
    reward_model.model.input_tokenizer=null \
    reward_model.model.path=${RM_MODEL_PATH} \
    reward_model.micro_batch_size=${infer_micro_batch_size} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.gamma=${gae_gamma} \
    algorithm.lam=${gae_lam} \
    trainer.critic_warmup=10 \
    trainer.logger=['console'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.val_only=False \


    data.actor_training_batch_size=510 \
    algorithm.all_samples_with_grad=True \
    algorithm.all_samples_with_grad_sync=True \
    critic.cliprange_value_low=${cliprange_value_low} \
    critic.cliprange_value_high=${cliprange_value_high} \
    data.window_response_length=8192 \
    actor_rollout_ref.rollout.train_generate_kwargs.max_new_tokens=8192 \
    actor_rollout_ref.actor.window_response_length=8192 \
    actor_rollout_ref.actor.lm_loss_weight=0.1 \
    algorithm.use_variable_lambda=True \
    algorithm.variable_lambda_scalar=0.05 \
    algorithm.use_separate_critic_lam=True \
    algorithm.critic_lam=1.0 \
    +algorithm.use_actual_values=True \
    +algorithm.adv_whiten=True \
    +algorithm.adv_bias=0.0 \
    +algorithm.adv_clamp=True \
    reward_model.delete_eos=False \
    algorithm.add_eos=False \
    +algorithm.force_append_eos=${false_append_eos} \
    actor_rollout_ref.actor.ppo_epochs=${ppo_epochs} \
    actor_rollout_ref.actor.scale_pg_by_local_kl=False \
    actor_rollout_ref.rollout.num_bon=${num_bon} \
    actor_rollout_ref.rollout.bon_strategy=${bon_strategy} \
    data.answer_key=answer \
    +actor_rollout_ref.model.use_rmpad=True \
    +actor_rollout_ref.model.override_config.embd_pdrop=0. \
    +actor_rollout_ref.model.override_config.resid_pdrop=0. \
    actor_rollout_ref.actor.scale_pg_by_kl=False \
    reward_model.mean=0.0 \
    reward_model.std=1.0 \
    reward_model.use_last_response=${use_last_response} \
    reward_model.punish_format=False \
    reward_model.format_punish_score=-0.1 \


    reward_model.add_int_verify=False \
    reward_model.strict_box_verify=False \
    reward_model.need_punish_duplicate=True \
    reward_model.punish_score=\'rule-lighteval/MATH_v2:-1\'

    trainer.default_hdfs_dir=${default_hdfs_dir} \
    actor_rollout_ref.model.external_lib=seed_models \
    
    critic.model.external_lib=seed_models \