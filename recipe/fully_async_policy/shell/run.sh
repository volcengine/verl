#!/usr/bin/env bash
set -xeuo pipefail

# PPO Strategy Selection
# PPO_STRATEGY can be "ppo" (regular PPO) or "async_ppo" (fully async PPO)
PPO_STRATEGY=${PPO_STRATEGY:-"ppo"}

# Algorithm Selection
PPO_ALGO=${PPO_ALGO:-"grpo"}  # grpo gspo dapo

# Model Selection
MODEL_CONFIG=${MODEL_CONFIG: -"Qwen2.5-Math-7B"}

# Common configuration
project_name=$PPO_STRATEGY
exp_name="ppo_${PPO_STRATEGY}_${PPO_ALGO}_$(date +%Y%m%d_%H%M%S)"

# data paths
CKPTS_DIR=./ckpts/${project_name}/${exp_name}
TRAIN_FILE=/cfs_shtx5_serving_3/mlp/training/docker/user/hadoop-ai-search/houzhenggang/data/dapo/dapo-math-17k.parquet
TEST_FILE=/cfs_shtx5_serving_3/mlp/training/docker/user/hadoop-ai-search/houzhenggang/data/dapo/aime-2024.parquet


# Model
if [ "$MODEL_CONFIG" = "Qwen3-32B" ]; then
    ## actor模型路径
    MODEL_PATH=/cfs_shtx5_serving_3/mlp/training/docker/user/hadoop-ai-search/houzhenggang/model/Qwen3-32B
    fsdp_size=-1
    sp_size=4
    gen_tp=4
    step=200

    # Response length parameters
    max_prompt_length=$((1024 * 2))
    max_response_length=$((1024 * 20))

elif [ "$MODEL_CONFIG" = "Qwen2.5-Math-7B" ]; then
    ## actor模型路径
    MODEL_PATH=/cfs_shtx5_serving_3/mlp/training/docker/user/hadoop-ai-search/houzhenggang/model/Qwen2___5-Math-7B
    fsdp_size=8
    sp_size=4
    gen_tp=4
    step=400

    # Response length parameters
    max_prompt_length=$((1024 * 2))
    max_response_length=$((1024 * 28))

else
    echo "Error: Unknown model type: $model_config" >&2
    exit 1
fi


# Rollout configuration
rollout_mode="async"
rollout_name="vllm"  # sglang or vllm
if [ "$rollout_mode" = "async" ]; then
    export VLLM_USE_V1=1
    return_raw_chat="True"
fi

# Algorithm configuration based on PPO_ALGO
if [ "${PPO_ALGO}" = "grpo" ]; then
    adv_estimator=grpo
    loss_agg_mode=token-mean
    reward_manager=dapo
    loss_mode=vanilla
    use_kl_loss=true
    use_kl_in_reward=False
    kl_coef=0.0
    kl_loss_coef=0.01
    clip_ratio_low=0.2                        # PPO裁剪下限
    clip_ratio_high=0.28                      # PPO裁剪上限
    enable_overlong_buffer=False
    overlong_buffer_len=$((1024 * 4))
    overlong_penalty_factor=1.0
elif [ "${PPO_ALGO}" = "gspo" ]; then
    adv_estimator=grpo
    loss_agg_mode=seq-mean-token-mean
    reward_manager=dapo
    loss_mode=gspo
    use_kl_loss=false
    use_kl_in_reward=False
    kl_coef=0.0
    kl_loss_coef=0.001
    clip_ratio_low=0.0003
    clip_ratio_high=0.0004
    enable_overlong_buffer=False
    overlong_buffer_len=$((1024 * 4))
    overlong_penalty_factor=1.0
elif [ "${PPO_ALGO}" = "dapo" ]; then
    adv_estimator=grpo
    loss_agg_mode=token-mean
    reward_manager=dapo
    loss_mode=vanilla
    use_kl_loss=false
    use_kl_in_reward=False
    kl_coef=0.0
    kl_loss_coef=0.0
    clip_ratio_low=0.2
    clip_ratio_high=0.28
    enable_overlong_buffer=True
    overlong_buffer_len=$((1024 * 4))
    overlong_penalty_factor=1.0
else
    echo "Error: Unknown PPO_ALGO '${PPO_ALGO}'. Supported values: 'grpo', 'gspo', 'dapo'"
    exit 1
fi



# Generation parameters
temperature=1.0
top_p=1.0
top_k=-1
val_top_p=0.7


# Main control: Upper threshold for IS weights (null = disabled, float = enabled)
rollout_is_threshold=2.0
rollout_is=${rollout_is: -"False"}}
rollout_is_threshold_lower=null
rollout_is_level=token
rollout_is_mode=truncate
rollout_is_veto_threshold=1e-4

# Performance Related Parameters
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 3))

# Common parameters
n_resp_per_prompt=16
test_freq=20
train_prompt_mini_bsz=32
train_batch_size=512

# Resource
NNODES=${NNODES:-16}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

# Common parameters array (sorted by category)
common_params=(
    # Data configuration
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.prompt_key=prompt
    data.truncation='left'
    data.return_raw_chat=${return_raw_chat}

    # Model configuration
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=True


    # Actor configuration
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss}
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low}
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high}
    actor_rollout_ref.actor.clip_ratio_c=10.0
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode}
    actor_rollout_ref.actor.optim.lr_warmup_steps=10
    actor_rollout_ref.actor.optim.weight_decay=0.1
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode}
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len}
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz}

    # Model configuration
    +actor_rollout_ref.model.override_config.max_position_embeddings=32768

    # Rollout configuration
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length))
    actor_rollout_ref.rollout.mode=${rollout_mode}
    actor_rollout_ref.rollout.name=${rollout_name}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
    actor_rollout_ref.rollout.n=${n_resp_per_prompt}
    actor_rollout_ref.rollout.temperature=${temperature}
    actor_rollout_ref.rollout.top_p=${top_p}
    actor_rollout_ref.rollout.top_k=${top_k}
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature}
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p}
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k}
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.n=1


    # Reference configuration
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size}

    # FSDP configuration
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size}
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size}

    # Algorithm configuration
    algorithm.adv_estimator=${adv_estimator}
    algorithm.use_kl_in_reward=${use_kl_in_reward}
    algorithm.kl_ctrl.kl_coef=${kl_coef}

    # Rollout IS
    algorithm.rollout_is=${rollout_is}
    algorithm.rollout_is_threshold=${rollout_is_threshold}
    algorithm.rollout_is_threshold_lower=${rollout_is_threshold_lower}
    algorithm.rollout_is_level=${rollout_is_level}
    algorithm.rollout_is_mode=${rollout_is_mode}
    algorithm.rollout_is_veto_threshold=${rollout_is_veto_threshold}

    critic.strategy=fsdp2

    # Reward model configuration
    reward_model.reward_manager=${reward_manager}
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer}
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len}
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor}
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False
    +reward_model.reward_kwargs.max_resp_len=${max_response_length}

    # Trainer configuration
    trainer.logger='["console","tensorboard"]'
    trainer.project_name="${project_name}"
    trainer.experiment_name="${exp_name}"
    trainer.save_freq=-1
    trainer.total_epochs=20
)

# Execute based on PPO_STRATEGY
if [ "${PPO_STRATEGY}" == "ppo" ]; then
    echo "Running regular PPO training with ${PPO_ALGO} algorithm..."

    # PPO-specific parameters
    ppo_params=(
        # Data configuration
        data.train_batch_size=${train_batch_size}

        # Rollout configuration
        actor_rollout_ref.hybrid_engine=True
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6

        # FSDP configuration
        actor_rollout_ref.actor.fsdp_config.param_offload=True
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
        actor_rollout_ref.ref.fsdp_config.param_offload=True

        # Trainer configuration
        trainer.nnodes=${NNODES}
        trainer.n_gpus_per_node=${NGPUS_PER_NODE}
        trainer.test_freq=${test_freq}
        trainer.total_training_steps=${step}

    )

    python -X faulthandler -m verl.trainer.main_ppo \
        "${common_params[@]}" \
        "${ppo_params[@]}" $@

elif [ "${PPO_STRATEGY}" == "async_ppo" ]; then
    echo "Running fully async PPO training with ${PPO_ALGO} algorithm..."

    # Async-specific parameters
    NNODES_ROLLOUT=${NNODES_ROLLOUT:-$((NNODES / 2))}
    NNODES_TRAIN=${NNODES_TRAIN:-$((NNODES / 2))}

    train_prompt_bsz=0
    gen_prompt_bsz=1
    total_rollout_steps=$((${train_batch_size} * ${step}))
    staleness_threshold=0.5
    trigger_parameter_sync_step=$((${train_batch_size} / ${train_prompt_mini_bsz}))
    require_batches=1
    partial_rollout=True
    recompute_old_log_prob=${recompute_old_log_prob:-"False"}

    # Async-specific parameters
    async_params=(
        # Data configuration
        data.train_batch_size=${train_prompt_bsz}
        data.gen_batch_size=${gen_prompt_bsz}

        # Rollout configuration
        actor_rollout_ref.rollout.gpu_memory_utilization=0.8
        actor_rollout_ref.hybrid_engine=False

        # FSDP configuration
        actor_rollout_ref.actor.fsdp_config.param_offload=False
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
        actor_rollout_ref.ref.fsdp_config.param_offload=True

        # Trainer configuration
        trainer.nnodes="${NNODES_TRAIN}"
        trainer.n_gpus_per_node="${NGPUS_PER_NODE}"

        # Rollout configuration
        rollout.nnodes="${NNODES_ROLLOUT}"
        rollout.n_gpus_per_node="${NGPUS_PER_NODE}"
        rollout.total_rollout_steps=${total_rollout_steps}
        rollout.total_epochs=2
        rollout.test_freq=${test_freq}

        # Async training configuration
        async_training.staleness_threshold=${staleness_threshold}
        async_training.trigger_parameter_sync_step=${trigger_parameter_sync_step}
        async_training.require_batches=${require_batches}
        async_training.partial_rollout=${partial_rollout}
        async_training.use_rollout_log_probs=True
        async_training.recompute_old_log_prob=${recompute_old_log_prob}
    )

    python -X faulthandler -m recipe.fully_async_policy.fully_async_main \
        "${common_params[@]}" \
        "${async_params[@]}" $@

else
    echo "Error: Unknown PPO_STRATEGY '${PPO_STRATEGY}'. Supported values: 'ppo', 'async_ppo'"
    exit 1
fi

echo "Training completed successfully with strategy: ${PPO_STRATEGY}, algorithm: ${PPO_ALGO}"