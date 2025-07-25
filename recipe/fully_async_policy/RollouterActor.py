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
import ray

from recipe.fully_async_policy.rollouter import Rollouter


@ray.remote
class RollouterActor:
    """Rollouter的Ray Actor包装器"""

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping,
        resource_pool_manager,
        ray_worker_group_cls,
        processor=None,
        train_dataset=None,
        collate_fn=None,
        train_sampler=None,
        device_name="cuda",
    ):
        self.rollouter = Rollouter(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            processor=processor,
            train_dataset=train_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=device_name,
        )

    def init_workers(self):
        """初始化worker"""
        return self.rollouter.init_workers()

    def set_message_queue_client(self, message_queue_client):
        """设置消息队列客户端"""
        return self.rollouter.set_message_queue_client(message_queue_client)

    def set_parameter_synchronizer(self, param_synchronizer):
        """设置参数同步器"""
        return self.rollouter.set_parameter_synchronizer(param_synchronizer)

    def update_rollout_weights(self, param_version: int):
        """更新rollout权重"""
        return self.rollouter.update_rollout_weights(param_version)

    def fit(self):
        """开始生成循环"""
        return self.rollouter.fit()

    def shutdown(self):
        """关闭rollouter"""
        return self.rollouter.shutdown()

    def get_statistics(self):
        """获取统计信息"""
        return self.rollouter.get_statistics()
