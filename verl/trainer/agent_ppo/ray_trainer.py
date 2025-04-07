"""
Agent-specific implementation of PPO Trainer.
This trainer extends the base RayPPOTrainer with agent-specific functionality.
"""

from typing import Dict, Type, Optional
from tqdm import tqdm
import uuid
import numpy as np
from copy import deepcopy
from pprint import pprint
from omegaconf import OmegaConf, open_dict

import torch
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup
from verl.utils.dataset.rl_agent_dataset import RLAgentDataset, collate_fn
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer, 
    ResourcePoolManager, 
    Role, 
    WorkerType,
    AdvantageEstimator,
    # apply_kl_penalty, 
    # compute_advantage,
    # compute_response_mask,
    _timer
)


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
            reward_fn = None,       # Reward is provided by the environment
            val_reward_fn = None    # Reward is provided by the environment
        )
        assert self.use_rm is False, "Reward model is not needed for agent PPO. The reward is provided by the environment"      
    
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
        
    def _create_dataloader(self):
        # TODO: we have to make sure the batch size is divisible by the dp size
        self.train_dataset = RLAgentDataset(
            environment_endpoint=self.config.env.environment_endpoint,
            parquet_files=self.config.data.train_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            max_prompt_length=self.config.data.max_prompt_length,
            truncation=self.config.data.get('truncation', 'error'),
        )
        assert self.train_dataset.truncation == self.config.data.get(
            'truncation', 'error'
        ), f'dataset truncation {self.train_dataset.truncation} must be the same as config {self.config.data.get("truncation", "error")}'
        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get('seed', 1))
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)
        
        self.train_dataloader = StatefulDataLoader(dataset=self.train_dataset,
                                                   batch_size=self.config.data.train_batch_size,
                                                   num_workers=8,
                                                   drop_last=True,
                                                   collate_fn=collate_fn,
                                                   sampler=sampler)
        
        self.val_dataset = RLAgentDataset(
            environment_endpoint=self.config.env.environment_endpoint,
            parquet_files=self.config.data.val_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            max_prompt_length=self.config.data.max_prompt_length,
            truncation=self.config.data.get('truncation', 'error'),
        )
        assert self.val_dataset.truncation == self.config.data.get(
            'truncation', 'error'
        ), f'dataset truncation {self.val_dataset.truncation} must be the same as config {self.config.data.get("truncation", "error")}'
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            # Validation datasets are sent to inference engines as a whole batch,
            # which will schedule the memory themselves.
            batch_size=len(self.val_dataset),
            num_workers=8,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn)
        
        assert len(self.train_dataloader) >= 1
        assert len(
            self.val_dataloader
        ) == 1, "Validation dataloader must have a single batch, which inference engines will schedule the memory themselves."

        print(f'Size of train dataloader: {len(self.train_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps
        
    def _validate(self):
        reward_tensor_lst = []
        data_source_lst = []

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n,
                                           interleave=True)

            # Store original inputs
            input_ids = test_batch.batch['input_ids']
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            test_gen_batch = test_batch.pop(
                batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                non_tensor_batch_keys=['raw_prompt_ids'],
            )

            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                'validate': True,
            }
            print(f'test_gen_batch meta info: {test_gen_batch.meta_info}')

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print('validation generation end')

            # Store generated outputs
            output_ids = test_output_gen_batch.batch['responses']
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            
            # TODO: Remove the debug print
            # print("test_batch batch: ", test_batch.batch[0, ...])
            # print("test_batch non_tensor_batch: ", {
            #     k: v[0] for k, v in test_batch.non_tensor_batch.items()
            # })
            # print("test_batch meta_info: ", test_batch.meta_info)
            # print("test_gen_batch batch: ", test_gen_batch.batch[0, ...])
            # print("test_output_gen_batch batch: ", test_output_gen_batch.batch[0, ...])
            # response_tokens = test_output_gen_batch.batch[0,...]['responses']
            # response = self.tokenizer.decode(response_tokens, skip_special_tokens=False)
            # print("response: ", response)
            # print("input: ", self.tokenizer.decode(test_gen_batch.batch[0,...]['input_ids'], skip_special_tokens=False))
            return {}

            # evaluate using reward_function
            reward_tensor = self.val_reward_fn(test_batch)

            # Store scores
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)

        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)

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

        # # add tqdm
        # progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # # we start from step 1
        # self.global_steps += 1
        # last_val_metrics = None

        # for epoch in range(self.config.trainer.total_epochs):
        #     for batch_dict in self.train_dataloader:
        #         metrics = {}
        #         timing_raw = {}

        #         batch: DataProto = DataProto.from_single_dict(batch_dict)

        #         # pop those keys for generation
        #         if 'multi_modal_inputs' in batch.non_tensor_batch.keys():
        #             gen_batch = batch.pop(
        #                 batch_keys=['input_ids', 'attention_mask', 'position_ids'],
        #                 non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
        #             )
        #         else:
        #             gen_batch = batch.pop(
        #                 batch_keys=['input_ids', 'attention_mask', 'position_ids'],
        #                 non_tensor_batch_keys=['raw_prompt_ids'],
        #             )

        #         is_last_step = self.global_steps >= self.total_training_steps

        #         with _timer('step', timing_raw):
        #             # generate a batch
        #             with _timer('gen', timing_raw):
        #                 gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

        #             if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
        #                 with _timer('gen_max', timing_raw):
        #                     gen_baseline_batch = deepcopy(gen_batch)
        #                     gen_baseline_batch.meta_info['do_sample'] = False
        #                     gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

        #                     batch = batch.union(gen_baseline_output)
        #                     reward_baseline_tensor = self.reward_fn(batch)
        #                     reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

        #                     batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

        #                     batch.batch['reward_baselines'] = reward_baseline_tensor

        #                     del gen_baseline_batch, gen_baseline_output

        #             batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
        #                                                      dtype=object)
        #             # repeat to align with repeated responses in rollout
        #             batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
        #             batch = batch.union(gen_batch_output)

        #             batch.batch['response_mask'] = compute_response_mask(batch)
        #             # balance the number of valid tokens on each dp rank.
        #             # Note that this breaks the order of data inside the batch.
        #             # Please take care when you implement group based adv computation such as GRPO and rloo
        #             if self.config.trainer.balance_batch:
        #                 self._balance_batch(batch, metrics=metrics)

        #             # compute global_valid tokens
        #             batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

        #             # recompute old_log_probs
        #             with _timer('old_log_prob', timing_raw):
        #                 old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
        #                 batch = batch.union(old_log_prob)

        #             if self.use_reference_policy:
        #                 # compute reference log_prob
        #                 with _timer('ref', timing_raw):
        #                     ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
        #                     batch = batch.union(ref_log_prob)

        #             # compute values
        #             if self.use_critic:
        #                 with _timer('values', timing_raw):
        #                     values = self.critic_wg.compute_values(batch)
        #                     batch = batch.union(values)

        #             with _timer('adv', timing_raw):
        #                 # compute scores. Support both model and function-based.
        #                 # We first compute the scores using reward model. Then, we call reward_fn to combine
        #                 # the results from reward model and rule-based results.
        #                 if self.use_rm:
        #                     # we first compute reward model score
        #                     reward_tensor = self.rm_wg.compute_rm_score(batch)
        #                     batch = batch.union(reward_tensor)

        #                 # we combine with rule-based rm
        #                 reward_tensor = self.reward_fn(batch)
        #                 batch.batch['token_level_scores'] = reward_tensor

        #                 # compute rewards. apply_kl_penalty if available
        #                 if self.config.algorithm.use_kl_in_reward:
        #                     batch, kl_metrics = apply_kl_penalty(batch,
        #                                                          kl_ctrl=self.kl_ctrl_in_reward,
        #                                                          kl_penalty=self.config.algorithm.kl_penalty)
        #                     metrics.update(kl_metrics)
        #                 else:
        #                     batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

        #                 # compute advantages, executed on the driver process
        #                 batch = compute_advantage(batch,
        #                                           adv_estimator=self.config.algorithm.adv_estimator,
        #                                           gamma=self.config.algorithm.gamma,
        #                                           lam=self.config.algorithm.lam,
        #                                           num_repeat=self.config.actor_rollout_ref.rollout.n)

        #             # update critic
        #             if self.use_critic:
        #                 with _timer('update_critic', timing_raw):
        #                     critic_output = self.critic_wg.update_critic(batch)
        #                 critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
        #                 metrics.update(critic_output_metrics)

        #             # implement critic warmup
        #             if self.config.trainer.critic_warmup <= self.global_steps:
        #                 # update actor
        #                 with _timer('update_actor', timing_raw):
        #                     actor_output = self.actor_rollout_wg.update_actor(batch)
        #                 actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
        #                 metrics.update(actor_output_metrics)

        #             # validate
        #             if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
        #                 (is_last_step or  self.global_steps % self.config.trainer.test_freq == 0):
        #                 with _timer('testing', timing_raw):
        #                     val_metrics: dict = self._validate()
        #                     if is_last_step:
        #                         last_val_metrics = val_metrics
        #                 metrics.update(val_metrics)

        #             if self.config.trainer.save_freq > 0 and ( is_last_step or \
        #                     self.global_steps % self.config.trainer.save_freq == 0):
        #                 with _timer('save_checkpoint', timing_raw):
        #                     self._save_checkpoint()

        #         # collect metrics
        #         metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
        #         metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
        #         # TODO: implement actual tflpo and theoretical tflpo
        #         n_gpus = self.resource_pool_manager.get_n_gpus()
        #         metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

        #         # TODO: make a canonical logger that supports various backend
        #         logger.log(data=metrics, step=self.global_steps)

        #         if is_last_step:
        #             pprint(f'Final validation metrics: {last_val_metrics}')
        #             progress_bar.close()
        #             return

        #         progress_bar.update(1)
        #         self.global_steps += 1
