# Copyright 2025 Meituan Ltd. and/or its affiliates
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

import logging

import ray
from ray.util.collective import collective

logger = logging.getLogger(__name__)


@ray.remote
class ParameterSynchronizer:
    """
    统一的参数同步器，负责在actor和rollout之间同步模型参数
    基于one_step_off_policy的成熟同步模式实现
    合并了原有的多个同步器类的功能
    """

    def __init__(self, config, actor_wg, rollout_wg):
        """
        初始化统一参数同步器

        Args:
            config: 配置对象
            actor_wg: trainer actor引用（用于async模式）
            rollout_wg: rollouter actor引用（用于async模式）
        """
        self.config = config
        self.actor_wg = actor_wg
        self.rollout_wg = rollout_wg

        # 基础属性
        self.weights_info = None
        self.sync_group_initialized = False
        self.sync_group_name = "actor_rollout"

        # 统计信息
        self.current_version = 0

        self._init_weights_info()
        self._init_sync_group()

    def get_current_param_version(self) -> int:
        """获取当前参数版本号"""
        return self.current_version

    def get_weights_info(self):
        """获取权重信息"""
        return self.weights_info

    def _init_weights_info(self):
        self.weights_info = self.actor_wg.get_actor_weights_info()[0]
        self.rollout_wg.set_actor_weights_info(self.weights_info)

    def _init_sync_group(self):
        print("Initializing parameter synchronization group...")
        actor_rollout_workers = self.actor_wg.workers + self.rollout_wg.workers
        collective.create_collective_group(
            actor_rollout_workers,
            len(actor_rollout_workers),
            list(range(0, len(actor_rollout_workers))),
            backend="nccl",
            group_name=self.sync_group_name,
        )

    def sync_weights(self, version):
        self.current_version = version
        logger.debug(f"Starting weight synchronization (version {self.current_version})...")
        self.actor_wg.sync_rollout_weights()
        ray.get(self.rollout_wg.sync_rollout_weights())
