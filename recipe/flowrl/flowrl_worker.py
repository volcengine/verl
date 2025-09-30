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

"""FlowRL specific worker implementations."""

import torch
from verl.workers.fsdp_workers import BaseFSDPWorker
from recipe.flowrl.flowrl_actor import ProjZModule


class FlowRLFSDPWorker(BaseFSDPWorker):
    """FSDP Worker with FlowRL-specific model modifications."""

    def _post_init_model(self):
        """Add projection network after model initialization."""
        super()._post_init_model()

        # Add projection network for log Z estimation
        if hasattr(self.actor_module, 'config'):
            if hasattr(self.actor_module.config, 'hidden_size'):
                hidden_size = self.actor_module.config.hidden_size
            else:
                # Fallback for different architectures
                hidden_size = getattr(self.actor_module.config, 'd_model', 4096)

            proj_layers = getattr(self.config.actor, 'proj_layer', 3)
            proj_dropout = getattr(self.config.actor, 'proj_dropout', 0.1)

            self.actor_module.proj_z = ProjZModule(
                hidden_size=hidden_size,
                num_layers=proj_layers,
                dropout=proj_dropout
            )

    def save_checkpoint(self, checkpoint_dir):
        """Save checkpoint with special handling for proj_z parameters."""

        # Save full checkpoint including proj_z
        full_checkpoint_path = super().save_checkpoint(checkpoint_dir)

        # Also save inference version without proj_z for deployment
        inference_state_dict = {}
        full_state_dict = self.actor_module.state_dict()

        for name, param in full_state_dict.items():
            if not name.startswith("proj_z"):
                inference_state_dict[name] = param

        # Save inference-ready checkpoint
        inference_checkpoint_path = os.path.join(
            checkpoint_dir,
            f"inference_{os.path.basename(full_checkpoint_path)}"
        )

        torch.save({
            'model_state_dict': inference_state_dict,
            'config': self.config
        }, inference_checkpoint_path)

        return full_checkpoint_path, inference_checkpoint_path


def create_flowrl_model_update_function():
    """Create a model update function that filters proj_z parameters for vLLM."""

    def filter_proj_z_for_vllm(model_updates):
        """Filter out proj_z parameters when loading to vLLM."""
        filtered_updates = {}

        for name, param in model_updates.items():
            if not name.startswith("proj_z"):
                filtered_updates[name] = param

        return filtered_updates

    return filter_proj_z_for_vllm