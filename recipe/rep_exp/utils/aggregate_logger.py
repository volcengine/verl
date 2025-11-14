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
A Ray logger will receive logging info from different processes.
"""

import json
import os


class JsonEvalLogger:
    """
    A logger that logs to a json file.
    Args:
        save_path: The path to the checkpoint to resume from.
        task: The task name, used to name the experiment.
    """

    def __init__(self, save_path: str, task: str):
        self.root = "eval"
        if save_path is not None and save_path != "":
            self.experiment_name = save_path.split("/")[-2]
            self.checkpoint_type = save_path.split("/")[-1]
        else:
            self.experiment_name = f"{task}_untrained"
            self.checkpoint_type = ""

    def flush(self):
        pass

    def log(self, data, step):
        # Create eval folder
        save_folder = os.path.join(self.root, self.experiment_name, self.checkpoint_type)
        os.makedirs(save_folder, exist_ok=True)

        # Save to json
        with open(os.path.join(save_folder, "eval.json"), "w") as f:
            json.dump(data, f)
