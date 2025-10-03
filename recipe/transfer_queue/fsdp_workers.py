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
The main entry point to run the PPO algorithm
"""

import verl.workers.fsdp_workers as workers
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.transferqueue_utils import create_transferqueue_client


class ActorRolloutRefWorker(workers.ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=True)
    def create_transferqueue_client(self, controller_infos, storage_infos):
        create_transferqueue_client(
            client_id=f"worker_{self.rank}",
            controller_infos=controller_infos,
            storage_infos=storage_infos,
        )

 
class CriticWorker(workers.CriticWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=True)
    def create_transferqueue_client(self, controller_infos, storage_infos):
        create_transferqueue_client(
            client_id=f"worker_{self.rank}",
            controller_infos=controller_infos,
            storage_infos=storage_infos,
        )


# TODO(sgm): we may need to extract it to dp_reward_model.py
class RewardModelWorker(workers.RewardModelWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=True)
    def create_transferqueue_client(self, controller_infos, storage_infos):
        create_transferqueue_client(
            client_id=f"worker_{self.rank}",
            controller_infos=controller_infos,
            storage_infos=storage_infos,
        )
   

# ================================= Async related workers =================================
class AsyncActorRolloutRefWorker(workers.AsyncActorRolloutRefWorker):
    pass
