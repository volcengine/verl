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
"""
An naive implementation of split placment example
"""
import os
from pprint import pprint
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl import DataProto
from verl.trainer.ppo.ray_trainer import compute_advantage, apply_kl_penalty, reduce_metrics, compute_data_metrics, Role, create_colocated_worker_cls
from codetiming import Timer


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

    global_steps = 0

    # perform validation before training
    # currently, we only support validation using the reward_function.
    if self.val_reward_fn is not None:
        val_metrics = self._validate()
        pprint(f'Initial validation metrics: {val_metrics}')

    for epoch in range(self.config.trainer.total_epochs):
        for batch_dict in self.train_dataloader:
            metrics = {}

            batch: DataProto = DataProto.from_single_dict(batch_dict)
            # batch = batch.to('cuda')

            # pop those keys for generation
            gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])

            # generate a batch
            with Timer(name='gen', logger=None) as timer:
                gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
            metrics['timing/gen'] = timer.last

            batch = batch.union(gen_batch_output)

            if self.use_reference_policy:
                # compute reference log_prob
                with Timer(name='ref', logger=None) as timer:
                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                    batch = batch.union(ref_log_prob)
                metrics['timing/ref'] = timer.last

            # compute values
            with Timer(name='values', logger=None) as timer:
                values = self.critic_wg.compute_values(batch)
                batch = batch.union(values)
            metrics['timing/values'] = timer.last

            with Timer(name='adv', logger=None) as timer:
                # compute scores. Support both model and function-based.
                # We first compute the scores using reward model. Then, we call reward_fn to combine
                # the results from reward model and rule-based results.
                if self.use_rm:
                    # we first compute reward model score
                    reward_tensor = self.rm_wg.compute_rm_score(batch)
                    batch = batch.union(reward_tensor)

                # we combine with rule-based rm
                reward_tensor = self.reward_fn(batch)
                batch.batch['token_level_scores'] = reward_tensor

                # compute rewards. apply_kl_penalty if available
                batch, kl_metrics = apply_kl_penalty(batch,
                                                     kl_ctrl=self.kl_ctrl,
                                                     kl_penalty=self.config.algorithm.kl_penalty)
                metrics.update(kl_metrics)

                # compute advantages, executed on the driver process
                batch = compute_advantage(batch,
                                          self.config.algorithm.gamma,
                                          self.config.algorithm.lam,
                                          adv_estimator=self.config.algorithm.adv_estimator)
            metrics['timing/adv'] = timer.last

            # update critic
            if self.use_critic:
                with Timer(name='update_critic_call', logger=None) as timer:
                    critic_output = self.critic_wg.update_critic(batch)
                metrics['timing/update_critic_call'] = timer.last

            # implement critic warmup
            if self.config.trainer.critic_warmup <= global_steps:
                # update actor
                with Timer(name='update_actor_call', logger=None) as timer:
                    actor_output = self.actor_rollout_wg.update_actor(batch)
                metrics['timing/update_acto_call'] = timer.last

            # NOTE: make sure you set blocking=False in update_actor and update_crtic in the worker class
            with Timer(name='update_actor_critic', logger=None) as timer:
                # NOTE: get the DataProtoFuture
                critic_output = critic_output.get()
                critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                metrics.update(critic_output_metrics)

                # NOTE: get the DataProtoFuture
                actor_output = actor_output.get()
                actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                metrics.update(actor_output_metrics)
            metrics['timing/update_actor_critic'] = timer.last

            # validate
            if self.val_reward_fn is not None and (global_steps + 1) % self.config.trainer.test_freq == 0:
                with Timer(name='testing', logger=None) as timer:
                    val_metrics: dict = self._validate()
                    val_metrics = {f'val/{key}': val for key, val in val_metrics.items()}
                metrics['timing/testing'] = timer.last
                metrics.update(val_metrics)

            # collect metrics
            data_metrics = compute_data_metrics(batch=batch)
            metrics.update(data_metrics)

            # TODO: make a canonical logger that supports various backend
            logger.log(data=metrics, step=global_steps)

            if self.config.trainer.save_freq > 0 and (global_steps + 1) % self.config.trainer.save_freq == 0:
                actor_local_path = os.path.join(self.config.trainer.default_local_dir, 'actor',
                                                f'global_step_{global_steps}')
                actor_remote_path = os.path.join(self.config.trainer.default_hdfs_dir, 'actor')
                self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path)

                if self.use_critic:
                    critic_local_path = os.path.join(self.config.trainer.default_local_dir, 'critic',
                                                     f'global_step_{global_steps}')
                    critic_remote_path = os.path.join(self.config.trainer.default_hdfs_dir, 'critic')
                    self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path)

            global_steps += 1

    # perform validation after training
    if self.val_reward_fn is not None:
        val_metrics = self._validate()
        pprint(f'Final validation metrics: {val_metrics}')
