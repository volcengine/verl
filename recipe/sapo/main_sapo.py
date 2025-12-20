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
"""SAPO entry point using the standard PPO trainer."""

import hydra

from verl.trainer.main_ppo import run_ppo
from verl.utils.device import auto_set_ascend_device_name


@hydra.main(config_path="config", config_name="sapo_trainer", version_base=None)
def main(config):
    auto_set_ascend_device_name(config)
    run_ppo(config)


if __name__ == "__main__":
    main()
