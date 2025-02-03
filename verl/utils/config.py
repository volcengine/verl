# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from typing import Dict

from omegaconf import DictConfig


def update_dict_with_config(dictionary: Dict, config: DictConfig):
    for key in dictionary:
        if hasattr(config, key):
            dictionary[key] = getattr(config, key)


def validate_config(config):
    # number of GPUs total
    n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

    # 1. Check total batch size for data correctness
    real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
    assert real_train_batch_size % n_gpus == 0, \
        f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

    # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
    # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
    def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
        if mbs is None and mbs_per_gpu is None:
            raise ValueError(f"[{name}] Please set at least one of '{name}.micro_batch_size' or "
                             f"'{name}.micro_batch_size_per_gpu'.")

        if mbs is not None and mbs_per_gpu is not None:
            raise ValueError(f"[{name}] You have set both '{name}.micro_batch_size' AND "
                             f"'{name}.micro_batch_size_per_gpu'. Please remove '{name}.micro_batch_size' "
                             f"because only '*_micro_batch_size_per_gpu' is supported (the former is deprecated).")

    if not config.actor_rollout_ref.actor.use_dynamic_bsz:
        # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
        check_mutually_exclusive(config.actor_rollout_ref.actor.ppo_micro_batch_size,
                                 config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu, "actor_rollout_ref.actor")

        # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
        check_mutually_exclusive(config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                                 config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                                 "actor_rollout_ref.ref")

        #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
        check_mutually_exclusive(config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                                 config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                                 "actor_rollout_ref.rollout")

    if not config.critic.use_dynamic_bsz:
        # Check for critic micro-batch size conflicts
        check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu,
                                 "critic")

    # Check for reward model micro-batch size conflicts
    if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
        check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu,
                                 "reward_model")

    # Actor
    # if NOT dynamic_bsz, we must ensure:
    #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
    #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
    if not config.actor_rollout_ref.actor.use_dynamic_bsz:
        sp_size = config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1)
        if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
            assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
            assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

    # critic
    if not config.critic.use_dynamic_bsz:
        sp_size = config.critic.get('ulysses_sequence_parallel_size', 1)
        if config.critic.ppo_micro_batch_size is not None:
            assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
            assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

    # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        if config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1) > 1 or \
                config.actor_rollout_ref.ref.get('ulysses_sequence_parallel_size', 1) > 1:
            assert config.actor_rollout_ref.model.use_remove_padding, \
                "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

    if config.critic.strategy == 'fsdp':
        if config.critic.get('ulysses_sequence_parallel_size', 1) > 1:
            assert config.critic.model.use_remove_padding, \
                "When using sequence parallelism for critic, you must enable `use_remove_padding`."

    print("[validate_config] All configuration checks passed successfully!")
