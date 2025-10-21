# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
import os

from hydra import compose, initialize_config_dir

from verl.workers.fsdp_workers import ActorRolloutRefWorker


def test_actor_rollout_ref_worker_actor_ref_model():
    """Test specifying different reference/actor model"""
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "8888"

    with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config")):
        config = compose("ppo_trainer")

    config.actor_rollout_ref.model.path = os.path.expanduser("~/models/Qwen/Qwen2.5-0.5B-Instruct")
    config.actor_rollout_ref.ref.model = {"path": os.path.expanduser("~/models/Qwen/Qwen2.5-1.5B-Instruct")}

    actor_rollout_ref_worker = ActorRolloutRefWorker(config.actor_rollout_ref, role="ref")
    actor_rollout_ref_worker.init_model()

    model_config = actor_rollout_ref_worker.ref_module_fsdp._fsdp_wrapped_module.config
    assert model_config.hidden_size == 1536

    # set ref.model to null, fallback to default case where actor is the same as reference
    config.actor_rollout_ref.ref.model = None
    actor_rollout_ref_worker = ActorRolloutRefWorker(config.actor_rollout_ref, role="ref")
    actor_rollout_ref_worker.init_model()

    model_config = actor_rollout_ref_worker.ref_module_fsdp._fsdp_wrapped_module.config
    assert model_config.hidden_size == 896
