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
from .ray_trainer import _timer, reduce_metrics, AdvantageEstimator

class ActorEnvironment:
    def __init__(self, config, actor_rollout_wg, reward_fn):
        self.config = config
        self.actor_rollout_wg = actor_rollout_wg
        self.reward_fn = reward_fn

    def step(self, timing_raw, gen_batch):
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

        batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                    dtype=object)
        # repeat to align with repeated responses in rollout
        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
        batch = batch.union(gen_batch_output)

    def update(self, timing_raw, batch, metrics):
        with _timer('update_actor', timing_raw):
            actor_output = self.actor_rollout_wg.update_actor(batch)
        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
        metrics.update(actor_output_metrics)
