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
This is the default ActorEnvironment that performs the rollout and actor update.
"""

import uuid
import numpy as np
from copy import deepcopy


class ActorEnvironment:

    def __init__(self):
        pass

    def post_init(self, config, actor_rollout_wg, reward_fn, tokenizer, processor):
        """A post init function that adds the essentials for the training loop
        after initializing the ActorEnvironment class.

        Args:
            config: The veRL config.
            actor_rollout_wg: The actor rollout worker group.
            reward_fn: The reward function.
            tokenizer: The tokenizer.
            processor: The processor (multimodal).
            
        Returns:
            Batch: Enriched batch containing generated sequences, rewards, and metadata
        """
        self.config = config
        self.actor_rollout_wg = actor_rollout_wg
        self.reward_fn = reward_fn
        self.tokenizer = tokenizer
        self.processor = processor

    def step(self, batch, gen_batch, timing_raw):
        """Performs a rollout step to generate sequences and compute rewards.
        
        Generates sequences using the actor model, computes rewards, and prepares training data.
        For REMAX advantage estimation, additionally generates baseline trajectories without
        sampling to compute baseline rewards. Handles batching and adds unique identifiers.
        
        Args:
            batch: Input batch containing initial data for rollout
            gen_batch: Configuration batch for sequence generation
            timing_raw: Dictionary to store timing metrics
            
        Returns:
            Batch: Enriched batch containing generated sequences, rewards, and metadata
        """
        from verl.trainer.ppo.ray_trainer import _timer, AdvantageEstimator

        # generate a batch
        with _timer('gen', timing_raw):
            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
            with _timer('gen_max', timing_raw):
                gen_baseline_batch = deepcopy(gen_batch)
                gen_baseline_batch.meta_info['do_sample'] = False
                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                batch = batch.union(gen_baseline_output)
                reward_baseline_tensor = self.reward_fn(batch)
                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                batch.batch['reward_baselines'] = reward_baseline_tensor

                del gen_baseline_batch, gen_baseline_output

        batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
        # repeat to align with repeated responses in rollout
        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
        batch = batch.union(gen_batch_output)

        return batch

    def update(self, timing_raw, batch, metrics):
        """Updates the actor model using collected rollout data.
        
        Executes the actor network update based on the provided batch of experiences.
        Merges training metrics from the update process into the provided metrics dictionary.
        
        Args:
            timing_raw: Dictionary to store timing metrics
            batch: Training data batch containing experiences from rollout
            metrics: Dictionary to accumulate training metrics
            
        Returns:
            actor_output: Output from actor update containing updated model and metrics
        """
        from verl.trainer.ppo.ray_trainer import _timer, reduce_metrics

        with _timer('update_actor', timing_raw):
            actor_output = self.actor_rollout_wg.update_actor(batch)
        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
        metrics.update(actor_output_metrics)

        return actor_output
