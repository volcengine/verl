#!/usr/bin/env bash
#
# GRPO (DAPO-Math-17K, AIME-2024 eval) | Step-5-Preview FP8 | FSDP2 + LoRA | vLLM rollout | 4 nodes x 4 GPUs
#
# The trainer runs the verl-native Step5 text model (verl/models/transformers/step5): BF16 projections
# dequantized from the FP8 checkpoint, FP32 router and norms, FP8 activation rule on the routed experts as in
# vLLM. Only LoRA adapters are trained and synced (adapter-only sync, LoRA-only checkpoints). The rollout
# records the routed experts and the trainer replays them (R3).
#
# Needs a vLLM with Step5 LoRA serving support, a Ray cluster with nnodes x 4 GPUs, the model on every node,
# and DAPO-Math-17K / AIME-2024 parquet files with data_source math_dapo / aime_2024. ckpt_dir must be on a
# filesystem that every node reads (the generated text config is written there and read by every worker).
#
# Data order: the reference run shuffled DAPO-Math-17K once with seed 42, kept the first total_steps x 32
# prompts and read them in order (data.shuffle=false, rollout seed 20261010). This script shuffles the full
# file with verl's defaults instead; pass data.shuffle=false with a prepared file to follow the run exactly.

set -xeuo pipefail

# ---- user-adjustable ----
model_path=${model_path:-$HOME/models/Step-5-Preview}
train_file=${train_file:-$HOME/data/dapo-math-17k.parquet}
val_file=${val_file:-$HOME/data/aime-2024.parquet}
ckpt_dir=${ckpt_dir:-$HOME/ckpt/step5_dapo_lora}
nnodes=${nnodes:-4}
gpus_per_node=${gpus_per_node:-4}
rollout_tp=${rollout_tp:-8}
total_steps=${total_steps:-200}

# Environment for every process. Exporting it here only reaches this shell; Ray workers of an already running
# cluster inherit their raylet's environment, so the same values are also passed in the job's runtime_env below.
step5_env=(
    # The Step5 training model is registered through the external-module hook in every process.
    VERL_USE_EXTERNAL_MODULES=verl.models.transformers.step5.register
    # The trainer follows the FP8 activation rule of the served routed experts (default on).
    VERL_STEP5_FP8_ACT_QUANT=1
    # Free the adapter's pinned host copy in every vLLM worker once it is active on the GPU.
    VERL_LORA_DROP_HOST_COPY=1
    # The reference run used vLLM's V2 model runner; set it explicitly rather than rely on vLLM's default choice.
    VLLM_USE_V2_MODEL_RUNNER=1
)
# Optional switches, forwarded when set: VERL_STEP5_LORA_PER_EXPERT=1 keeps every expert factor per expert,
# VERL_STEP5_ROUTING_AGREEMENT=1 logs how often the trainer router agrees with the replayed routing.
for name in VERL_STEP5_LORA_PER_EXPERT VERL_STEP5_ROUTING_AGREEMENT; do
    if [[ -n ${!name:-} ]]; then
        step5_env+=("$name=${!name}")
    fi
done
runtime_env_args=()
for kv in "${step5_env[@]}"; do
    export "$kv"
    # Quoted, so Hydra keeps "1" a string: Ray accepts only string env_vars.
    runtime_env_args+=("++ray_kwargs.ray_init.runtime_env.env_vars.${kv%%=*}=\"${kv#*=}\"")
done

# The checkpoint config nests the text model under text_config; the trainer builds the text model from it.
text_config=${ckpt_dir}/text-config
mkdir -p "$ckpt_dir"
python3 - "$model_path" "$text_config" <<'PY'
import json
import sys

from verl.models.transformers.step5.modeling_step5 import register_step5_training, training_text_config

register_step5_training()
raw = json.load(open(sys.argv[1] + "/config.json"))["text_config"]
training_text_config(raw).save_pretrained(sys.argv[2])
PY

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=true \
    algorithm.use_kl_in_reward=false \
    algorithm.filter_groups.enable=false \
    algorithm.rollout_correction.bypass_mode=false \
    algorithm.rollout_correction.rollout_is=null \
    algorithm.rollout_correction.rollout_rs=seq_mean_k1 \
    algorithm.rollout_correction.rollout_rs_threshold=0.995_1.005 \
    data.train_files="$train_file" \
    data.val_files="$val_file" \
    data.train_batch_size=32 \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=true \
    data.truncation=error \
    actor_rollout_ref.model.path="$model_path" \
    actor_rollout_ref.model.hf_config_path="$text_config" \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.model.use_liger=false \
    actor_rollout_ref.model.use_fused_kernels=false \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    +actor_rollout_ref.model.override_config.attn_implementation=eager \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.router_replay.mode=R3 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=false \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    "actor_rollout_ref.actor.optim.betas=[0.9,0.98]" \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    ++actor_rollout_ref.actor.checkpoint.save_lora_only=true \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$rollout_tp" \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_num_seqs=128 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.max_model_len=5120 \
    actor_rollout_ref.rollout.free_cache_engine=true \
    actor_rollout_ref.rollout.enable_prefix_caching=false \
    actor_rollout_ref.rollout.calculate_log_probs=true \
    actor_rollout_ref.rollout.enable_rollout_routing_replay=true \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.moe_backend=triton \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=false \
    trainer.nnodes="$nnodes" \
    trainer.n_gpus_per_node="$gpus_per_node" \
    trainer.total_training_steps="$total_steps" \
    trainer.val_before_train=true \
    trainer.test_freq=10 \
    trainer.save_freq=5 \
    trainer.max_actor_ckpt_to_keep=5 \
    trainer.resume_mode=auto \
    "trainer.logger=[console,tensorboard]" \
    trainer.project_name=step5_preview_dapo \
    trainer.experiment_name=grpo_lora_r32 \
    trainer.default_local_dir="$ckpt_dir" \
    "${runtime_env_args[@]}" "$@"
