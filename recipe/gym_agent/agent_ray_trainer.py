# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

"""
Agent-specific implementation of PPO Trainer.
This trainer extends the base RayPPOTrainer with agent-specific functionality.
"""

from typing import Dict, Type, Optional
from tqdm import tqdm
import uuid
import json
import numpy as np
from copy import deepcopy
from pprint import pprint
from omegaconf import OmegaConf, open_dict

import torch
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader

from verl import DataProto
import verl.utils.torch_functional as verl_F
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup
from .rl_agent_dataset import RLAgentDataset
from .agent_env import AgentEnv
from .utils import collate_fn
from verl.trainer.ppo import core_algos
from .metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics, reduce_metrics, bootstrap_metric, calc_maj_val, process_validation_metrics
from verl.trainer.ppo.ray_trainer import (RayPPOTrainer, ResourcePoolManager, Role, WorkerType, AdvantageEstimator,
                                          compute_advantage, _timer)


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    # The original implementation of apply_kl_penalty in ray trainer only assume single turn of response at last turn.
    # Here we extend it to support multiple turns of response at last turns, and using the model_generated_mask to calculate the kl penalty.

    # Hint for caculating the right offset
    # tokens obs/act      o o o o o a a a a a a o o o a a a o o
    # mask                0 0 0 0 0 1 1 1 1 1 1 0 0 0 1 1 1 0 0
    # p(a|s)             [x x x x x x x x x x x x x x x x x x]x
    # mask offset         0 0 0 0 1 1 1 1 1 1 0 0 0 1 1 1 0 0
    # masked p(a|s)      [0 0 0 0 x x x x x x 0 0 0 x x x 0 0]
    # reward r(a,s)       - - - - - 0 0 0 0 0 r - - - 0 0 r - -
    # reward offset      [- - - - 0 0 0 0 0 r - - - 0 0 r - -]

    model_generated_mask = data.batch['model_generated_mask'][:, 1:]
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                kl_penalty=kl_penalty)  # (batch_size, response_length)
    kld = kld * model_generated_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = verl_F.masked_mean(kld, mask=model_generated_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)

    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'actor/reward_kl_penalty': current_kl, 'actor/reward_kl_penalty_coeff': beta}

    return data, metrics


def compute_advantage_agent(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    # Advantage Computation is very different for agent PPO than the single turn PPO implementation.
    # Directly use the ray trainer implementation is dangerous. Need to be very careful.
    white_listed_adv_estimator = [AdvantageEstimator.GAE]
    assert adv_estimator in white_listed_adv_estimator, f"Advantage estimator {adv_estimator} is not supported for agent PPO"

    # To leverage the original ray trainer implementation, we need to reorder the values, rewards, and mask to be similar to the single turn style.
    # For example, the multiturn mask is like:
    # [[0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0],
    #  [0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
    #  [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]]
    # We need to reorder the values, rewards, and mask to be like:
    # [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #  [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    # Step 1: compute the reorder_indices and reverse_indices, which are used to reorder the values, mask, and arange_index
    model_generated_mask = data.batch['model_generated_mask'][:, 1:]
    arange_index = torch.arange(model_generated_mask.shape[1]).repeat(model_generated_mask.shape[0], 1)
    arange_index[model_generated_mask == 0] = torch.tensor(100 + model_generated_mask.shape[1])
    reorder_indices = torch.argsort(arange_index, dim=-1)
    reverse_indices = torch.argsort(reorder_indices, dim=-1)

    # Step 2: reorder the values, mask, and arange_index
    reordered_token_level_rewards = data.batch['token_level_rewards'].gather(dim=-1, index=reorder_indices)
    reordered_model_generated_mask = data.batch['model_generated_mask'].gather(dim=-1, index=reorder_indices)
    reordered_values = data.batch['values'].gather(dim=-1, index=reverse_indices)
    reordered_data = DataProto.from_single_dict({
        'token_level_rewards': reordered_token_level_rewards,
        'values': reordered_values,
        'response_mask': reordered_model_generated_mask,
    })
    if 'uid' in data.non_tensor_batch:
        reordered_data.non_tensor_batch['uid'] = data.non_tensor_batch['uid']
    if 'reward_baselines' in data.batch:
        raise NotImplementedError("Reward baselines are not supported for agent PPO. Need more validation.")
        reordered_data.batch['reward_baselines'] = data.batch['reward_baselines'].gather(dim=-1, index=reverse_indices)

    # Step 3: compute the advantages and returns
    reordered_data = compute_advantage(reordered_data, adv_estimator, gamma, lam, num_repeat)
    advantages = reordered_data.batch['advantages']
    returns = reordered_data.batch['returns']

    # Step 4: reverse the reordering
    data.batch['advantages'] = advantages.gather(dim=-1, index=reverse_indices)
    data.batch['returns'] = returns.gather(dim=-1, index=reverse_indices)

    return data


class AgentPPOTrainer(RayPPOTrainer):
    """
    Extension of RayPPOTrainer with agent-specific functionality.
    Inherits core PPO implementation from RayPPOTrainer.
    """

    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: Dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 processor=None):
        super().__init__(
            config,
            tokenizer,
            role_worker_mapping,
            resource_pool_manager,
            ray_worker_group_cls,
            processor,
            reward_fn=None,  # Reward is provided by the environment
            val_reward_fn=None  # Reward is provided by the environment
        )
        assert self.use_rm is False, "Reward model is not needed for agent PPO. The reward is provided by the environment"
        assert self.config.actor_rollout_ref.rollout.n == 1, "The number of rollout supported right now is 1"

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, \
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

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
                    raise ValueError(
                        f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(
                        f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. "
                        f"Please remove '{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
                    )

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.actor.ppo_micro_batch_size,
                                     config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.actor")

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                                         config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                                         "actor_rollout_ref.ref")

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.rollout")

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu,
                                     "critic")

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean", "seq-mean-token-sum", "seq-mean-token-mean"
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print(f"NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
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

        if self.use_critic and config.critic.strategy == 'fsdp':
            if config.critic.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.critic.model.use_remove_padding, \
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get('val_batch_size', None) is not None:
            print(
                f"WARNING: val_batch_size is deprecated. Validation datasets are sent to inference engines as a whole batch, which will schedule the memory themselves."
            )

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, \
                "validation gen temperature should be greater than 0 when enabling do_sample"

        print("[validate_config] All configuration checks passed successfully!")

    def _run_agent_rollout(self,
                           gen_batch: DataProto,
                           do_sample: Optional[bool] = None,
                           validate: bool = True) -> DataProto:
        gen_rollout_batch = gen_batch.select(batch_keys=['index'], non_tensor_batch_keys=['env'])
        env_list = gen_rollout_batch.non_tensor_batch['env']
        # initialize the environment
        # TODO: make this asyncio
        for env in env_list:
            env.initialize()

        for turn_idx in range(self.config.env.max_turn):
            # TODO: tokenize in parallel
            gen_tokenized_batch = DataProto.from_single_dict(
                collate_fn([env.tokenize_chat(add_generation_prompt=True) for env in env_list]))
            if validate:
                # only trigger in the validation phase
                gen_tokenized_batch.meta_info = {
                    'eos_token_id': self.tokenizer.eos_token_id,
                    'pad_token_id': self.tokenizer.pad_token_id,
                    'recompute_log_prob': False,
                    'do_sample': self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                    'validate': True,
                }
            if do_sample is not None:
                gen_tokenized_batch.meta_info['do_sample'] = do_sample

            gen_tokenized_batch_padded, pad_size = pad_dataproto_to_divisor(gen_tokenized_batch,
                                                                            self.actor_rollout_wg.world_size)
            output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(gen_tokenized_batch_padded)
            output_gen_batch = unpad_dataproto(output_gen_batch_padded, pad_size=pad_size)

            # print(f'output_gen_batch batch keys: {output_gen_batch.batch.keys()}') # ['input_ids', 'position_ids', 'attention_mask', 'responses', 'prompts']
            # print(f'output_gen_batch non_tensor_batch keys: {output_gen_batch.non_tensor_batch.keys()}') # []
            # print(f'output_gen_batch meta_info: {output_gen_batch.meta_info}') # {}
            # print(f'output_gen_batch input_ids: {output_gen_batch.batch["input_ids"]}') # output_gen_batch input_ids is the gen_tokenized_batch input_ids + response ids
            # print(f'gen_tokenized_batch input_ids: {gen_tokenized_batch.batch["input_ids"]}')
            # print(f'diff input_ids: {output_gen_batch.batch["input_ids"] - gen_tokenized_batch.batch["input_ids"]}') # would fail due to the different length of input_ids
            # TODO: make this asyncio
            new_env_list = []
            for i in range(len(output_gen_batch.batch)):
                sample = output_gen_batch[i]
                response_tokens = sample.batch['responses']
                response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
                env = env_list[i]
                _, reward, done, truncated, info = env.step(
                    response, tool_parsing_error_reward=self.config.env.tool_parsing_error_reward)
                if not done and not truncated:
                    # only keep the envs that are not done or truncated
                    new_env_list.append(env)

            if len(new_env_list) == 0:
                break
            env_list = new_env_list

        # store chat, action_turn, reward_by_action_turn in gen_rollout_batch
        env_list = gen_rollout_batch.non_tensor_batch['env']
        chat_list = np.array([env.chat for env in env_list], dtype=object)
        action_turn_list = np.array([env.action_turn for env in env_list], dtype=object)
        reward_by_action_turn_list = np.array([env.reward_by_action_turn for env in env_list], dtype=object)
        gen_rollout_batch.non_tensor_batch['chat'] = chat_list
        gen_rollout_batch.non_tensor_batch['action_turn'] = action_turn_list
        gen_rollout_batch.non_tensor_batch['reward_by_action_turn'] = reward_by_action_turn_list

        # Tokenize the chat
        # TODO: tokenize in parallel
        tokenized_batch = DataProto.from_single_dict(
            collate_fn([env.tokenize_chat(add_generation_prompt=False) for env in env_list]))
        gen_rollout_batch.union(tokenized_batch)

        # Debug
        # env = gen_rollout_batch.non_tensor_batch['env'][0]
        # print("Chat of first env:")
        # print(json.dumps(env.chat, indent=2))
        # print(f'Reward by action turn of first env: {env.reward_by_action_turn}')
        # print(f'Action turn of first env: {env.action_turn}')
        # print(f'gen_rollout_batch batch keys: {gen_rollout_batch.batch.keys()}')
        # print(f'gen_rollout_batch non_tensor_batch keys: {gen_rollout_batch.non_tensor_batch.keys()}')
        # print(f'gen_rollout_batch meta_info: {gen_rollout_batch.meta_info}')

        return gen_rollout_batch

    def _validate(self):
        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n,
                                           interleave=True)

            env_list = [
                AgentEnv(environment_endpoint=self.config.env.environment_endpoint,
                         env_name=sample.non_tensor_batch['env_name'],
                         seed=sample.non_tensor_batch['seed'],
                         env_kwargs=sample.non_tensor_batch['env_kwargs'],
                         agent_prompt_style=self.config.data.agent_prompt_style,
                         tokenizer=self.tokenizer,
                         max_prompt_length=self.config.data.max_prompt_length,
                         truncation=self.config.data.get('truncation', 'error')) for sample in test_batch
            ]
            test_batch.non_tensor_batch['env'] = np.array(env_list, dtype=object)

            test_batch.check_consistency()

            gen_rollout_batch = self._run_agent_rollout(test_batch)
            sample_inputs = [
                json.dumps(gen_rollout_batch[i].non_tensor_batch['chat'], indent=2)
                for i in range(len(gen_rollout_batch))
            ]
            sample_outputs = [
                json.dumps(
                    {
                        "action_turn": gen_rollout_batch[i].non_tensor_batch['action_turn'],
                        "reward_by_action_turn": gen_rollout_batch[i].non_tensor_batch['reward_by_action_turn'],
                    },
                    indent=2) for i in range(len(gen_rollout_batch))
            ]
            sample_scores = [
                sum(gen_rollout_batch[i].non_tensor_batch['reward_by_action_turn'])
                for i in range(len(gen_rollout_batch))
            ]

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        metric_dict = {}
        metric_dict[f'val/test_score'] = np.mean(sample_scores)

        return metric_dict

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        if self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer('step', timing_raw):
                    # create a batch of envs
                    with _timer('env_creation', timing_raw):
                        env_list = [
                            AgentEnv(environment_endpoint=self.config.env.environment_endpoint,
                                     env_name=sample.non_tensor_batch['env_name'],
                                     seed=sample.non_tensor_batch['seed'],
                                     env_kwargs=sample.non_tensor_batch['env_kwargs'],
                                     agent_prompt_style=self.config.data.agent_prompt_style,
                                     tokenizer=self.tokenizer,
                                     max_prompt_length=self.config.data.max_prompt_length,
                                     truncation=self.config.data.get('truncation', 'error')) for sample in batch
                        ]
                        batch.non_tensor_batch['env'] = np.array(env_list, dtype=object)

                        batch.check_consistency()

                    # agent rollout with stateful envs
                    with _timer('agent_rollout', timing_raw):
                        gen_rollout_batch = self._run_agent_rollout(batch, validate=False)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        # TODO: The implementation of REMAX is completed and needs better understanding of the baseline algorithm under agent context
                        raise NotImplementedError("REMAX is not implemented yet")
                        with _timer('gen_max', timing_raw):
                            baseline_batch = batch.select(batch_keys=['index'],
                                                          non_tensor_batch_keys=['env_name', 'seed', 'env_kwargs'])
                            baseline_env_list = [
                                AgentEnv(environment_endpoint=self.config.env.environment_endpoint,
                                         env_name=sample.non_tensor_batch['env_name'],
                                         seed=sample.non_tensor_batch['seed'],
                                         env_kwargs=sample.non_tensor_batch['env_kwargs'],
                                         agent_prompt_style=self.config.data.agent_prompt_style,
                                         tokenizer=self.tokenizer,
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         truncation=self.config.data.get('truncation', 'error'))
                                for sample in baseline_batch
                            ]
                            baseline_batch.non_tensor_batch['env'] = np.array(baseline_env_list, dtype=object)
                            baseline_batch.check_consistency()

                            baseline_gen_rollout_batch = self._run_agent_rollout(baseline_batch,
                                                                                 do_sample=False,
                                                                                 validate=False)

                            reward_baseline_tensor = baseline_gen_rollout_batch.non_tensor_batch[
                                'reward_by_action_turn']
                            reward_baseline_tensor = [sum(reward) for reward in reward_baseline_tensor]
                            reward_baseline_tensor = torch.Tensor(reward_baseline_tensor, dtype=torch.float32)
                            batch.batch['reward_baselines'] = reward_baseline_tensor

                            del baseline_batch, baseline_env_list

                    batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                             dtype=object)
                    # repeat to align with repeated responses in rollout
                    # TODO: check the logic if n > 1
                    assert self.config.actor_rollout_ref.rollout.n == 1, "The number of rollout supported right now is 1"
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_rollout_batch)

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # print(f'batch batch keys: {batch.batch.keys()}')
                    # print(f'batch non_tensor_batch keys: {batch.non_tensor_batch.keys()}')
                    # print(f'batch meta_info: {batch.meta_info}')
                    # print(f"Class of self.actor_rollout_wg: {type(self.actor_rollout_wg)}")

                    # recompute old_log_probs
                    with _timer('old_log_prob', timing_raw):
                        # From compute_log_prob
                        # data (DataProto): a DataProto containing keys
                        #     ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                        #     concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.
                        #     ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.
                        #     ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.
                        #     ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.
                        old_log_prob_input_batch = batch.select(
                            batch_keys=['input_ids', 'attention_mask', 'position_ids'])
                        # for computing log_prob, on all logits, masking will be haddled in trainer
                        # I didn't directly add `responses` to the batch is for the coder to really know what they are doing before using `responses` to fit the single turn implementation.
                        old_log_prob_input_batch.batch['responses'] = batch.batch['input_ids'][:, 1:]
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(old_log_prob_input_batch)
                        batch = batch.union(old_log_prob)

                    # old_log_probs will has one less column as it's p(a|s)
                    # print(f"old_log_prob old_log_probs shape: {old_log_prob.batch['old_log_probs'].shape} / {batch.batch['input_ids'].shape}")
                    # print(f'old_log_prob batch keys: {old_log_prob.batch.keys()}')
                    # print(f'old_log_prob non_tensor_batch keys: {old_log_prob.non_tensor_batch.keys()}')
                    # print(f'old_log_prob meta_info: {old_log_prob.meta_info}')

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob_input_batch = batch.select(
                                batch_keys=['input_ids', 'attention_mask', 'position_ids'])
                            ref_log_prob_input_batch.batch['responses'] = batch.batch['input_ids'][:, 1:]
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(ref_log_prob_input_batch)
                            batch = batch.union(ref_log_prob)

                        # ref_log_probs will has one less column as it's p(a|s)
                        # print(f"ref_log_prob ref_log_prob shape: {ref_log_prob.batch['ref_log_prob'].shape} / {batch.batch['input_ids'].shape}")
                        # print(f'ref_log_prob batch keys: {ref_log_prob.batch.keys()}')
                        # print(f'ref_log_prob non_tensor_batch keys: {ref_log_prob.non_tensor_batch.keys()}')
                        # print(f'ref_log_prob meta_info: {ref_log_prob.meta_info}')

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values_input_batch = batch.select(
                                batch_keys=['input_ids', 'attention_mask', 'position_ids'])
                            values_input_batch.batch['responses'] = batch.batch['input_ids'][:, 1:]
                            values = self.critic_wg.compute_values(values_input_batch)
                            batch = batch.union(values)

                        # Values will has one less column as it is only on the states
                        # print(f'values values shape: {values.batch["values"].shape} / {batch.batch["input_ids"].shape}')
                        # print(f'values batch keys: {values.batch.keys()}')
                        # print(f'values non_tensor_batch keys: {values.non_tensor_batch.keys()}')
                        # print(f'values meta_info: {values.meta_info}')

                    with _timer('adv', timing_raw):
                        # compute scores. The raw reward are computed provided by the environment, and is already stored in the batch
                        # `tokenwise_reward` is the rewards on the last tokens of each model generated (policy) turn. `tokenwise_reward` has the same size as input_ids
                        # `reward_by_action_turn` is rewards for each policy turn.
                        # `model_generated_mask` is a boolean mask of the same size as input_ids, indicating the tokens that are generated by the model.

                        # rename it for the ray trainer convention
                        # I didn't directly name it `token_level_scores` to test any potential issues when integrating with the functions originally designed in ray trainer (for single turn of response at last turn).
                        batch.batch['token_level_scores'] = batch.batch['tokenwise_reward'][:, 1:]

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl_in_reward,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            # get an offset to align with the log_prob etc.
                            # see more details in apply_kl_penalty
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        # compute advantages, executed on the driver process
                        batch = compute_advantage_agent(batch,
                                                        adv_estimator=self.config.algorithm.adv_estimator,
                                                        gamma=self.config.algorithm.gamma,
                                                        lam=self.config.algorithm.lam,
                                                        num_repeat=self.config.actor_rollout_ref.rollout.n)
                    # print(f'batch batch keys: {batch.batch.keys()}')
                    # print(f'batch non_tensor_batch keys: {batch.non_tensor_batch.keys()}')
                    # print(f'batch meta_info: {batch.meta_info}')

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_input_batch = batch.select(
                                batch_keys=['input_ids', 'attention_mask', 'position_ids', 'values', 'returns'])
                            critic_input_batch.batch['responses'] = batch.batch['input_ids'][:, 1:]
                            critic_input_batch.batch['response_mask'] = batch.batch['model_generated_mask'][:, 1:]
                            critic_output = self.critic_wg.update_critic(critic_input_batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)
                        print(f'critic_output_metrics: {critic_output_metrics}')

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            actor_input_batch = batch.select(batch_keys=[
                                'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'ref_log_prob',
                                'advantages'
                            ])
                            actor_input_batch.batch['responses'] = batch.batch['input_ids'][:, 1:]
                            actor_input_batch.batch['response_mask'] = batch.batch['model_generated_mask'][:, 1:]
                            actor_output = self.actor_rollout_wg.update_actor(actor_input_batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)
                        print(f'actor_output_metrics: {actor_output_metrics}')

                    # validate
                    if self.config.trainer.test_freq > 0 and \
                        (is_last_step or  self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and ( is_last_step or \
                            self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f'Final validation metrics: {last_val_metrics}')
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
