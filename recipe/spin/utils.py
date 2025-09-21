# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from omegaconf import DictConfig


def validate_config(
    config: DictConfig,
    use_reference_policy: bool,
    use_critic: bool,
) -> None:
    """
    Validate an OmegaConf DictConfig

    Args:
        config: The OmegaConf DictConfig to validate.
        use_reference_policy (bool): is ref policy needed
        use_critic (bool): is critic needed
    """
    # number of GPUs total
    n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

    # 1. Check total batch size for data correctness
    real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
    assert real_train_batch_size % n_gpus == 0, (
        f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."
    )

    # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
    # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
    def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
        settings = {
            "actor_rollout_ref.actor": "micro_batch_size",
            "critic": "micro_batch_size",
            "reward_model": "micro_batch_size",
            "actor_rollout_ref.ref": "log_prob_micro_batch_size",
            "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
        }

        if name in settings:
            param = settings[name]
            param_per_gpu = f"{param}_per_gpu"

            if mbs is None and mbs_per_gpu is None:
                raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

            if mbs is not None and mbs_per_gpu is not None:
                raise ValueError(
                    f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. "
                    f"Please remove '{name}.{param}' because only '*_{param_per_gpu}' is supported "
                    f"(the former is deprecated)."
                )

    if not config.actor_rollout_ref.actor.use_dynamic_bsz:
        # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
        check_mutually_exclusive(
            config.actor_rollout_ref.actor.ppo_micro_batch_size,
            config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
            "actor_rollout_ref.actor",
        )

        if use_reference_policy:
            # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.ref",
            )

        #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
        check_mutually_exclusive(
            config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
            config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
            "actor_rollout_ref.rollout",
        )

    if use_critic and not config.critic.use_dynamic_bsz:
        # Check for critic micro-batch size conflicts
        check_mutually_exclusive(
            config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic"
        )

    # Check for reward model micro-batch size conflicts
    if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
        check_mutually_exclusive(
            config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model"
        )

    # Actor
    # check if train_batch_size is larger than ppo_mini_batch_size
    # if NOT dynamic_bsz, we must ensure:
    #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
    #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
    if not config.actor_rollout_ref.actor.use_dynamic_bsz:
        assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
        sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
        if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
            assert (
                config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size
                == 0
            )
            assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

    assert config.actor_rollout_ref.actor.loss_agg_mode in [
        "token-mean",
        "seq-mean-token-sum",
        "seq-mean-token-mean",
    ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

    if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
        print("NOTICE: You have both enabled in-reward kl and kl loss.")

    # critic
    if use_critic and not config.critic.use_dynamic_bsz:
        assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
        sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
        if config.critic.ppo_micro_batch_size is not None:
            assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
            assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

    # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
    if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
        if (
            config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1
            or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1
        ):
            assert config.actor_rollout_ref.model.use_remove_padding, (
                "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."
            )

    if use_critic and config.critic.strategy in {"fsdp", "fsdp2"}:
        if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
            assert config.critic.model.use_remove_padding, (
                "When using sequence parallelism for critic, you must enable `use_remove_padding`."
            )

    if config.data.get("val_batch_size", None) is not None:
        print(
            "WARNING: val_batch_size is deprecated. Validation datasets are sent to inference engines "
            "as a whole batch, which will schedule the memory themselves."
        )

    # check eval config
    if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
        assert config.actor_rollout_ref.rollout.temperature > 0, (
            "validation gen temperature should be greater than 0 when enabling do_sample"
        )

    print("[validate_config] All configuration checks passed successfully!")
