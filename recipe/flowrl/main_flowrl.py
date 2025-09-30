#!/usr/bin/env python3
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

"""Main training script for FlowRL algorithm."""

import os
import sys
from omegaconf import OmegaConf

# Add VERL to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from verl.trainer.ppo import PPOTrainer
from verl.utils.config import load_config_from_file
from recipe.flowrl.flowrl_actor import FlowRLActor


class FlowRLTrainer(PPOTrainer):
    """FlowRL Trainer that extends PPOTrainer with FlowRL-specific components."""

    def __init__(self, config):
        super().__init__(config)

    def _create_actor_worker(self, config):
        """Create FlowRL actor worker instead of standard PPO actor."""
        return FlowRLActor(config)

    def _filter_proj_z_params(self, model_state_dict):
        """Filter out proj_z parameters when loading to vLLM for inference."""
        filtered_params = {}
        for name, param in model_state_dict.items():
            if not name.startswith("proj_z"):
                filtered_params[name] = param
        return filtered_params

    def _save_checkpoint(self, step):
        """Override to handle proj_z parameters in checkpointing."""
        # Save full model including proj_z
        checkpoint = {
            'step': step,
            'model_state_dict': self.actor_worker.actor_module.state_dict(),
            'optimizer_state_dict': self.actor_worker.optimizer.state_dict(),
            'config': self.config
        }

        checkpoint_path = os.path.join(self.config.trainer.save_dir, f'checkpoint_{step}.pt')
        torch.save(checkpoint, checkpoint_path)

        # Also save inference-ready version without proj_z
        inference_state_dict = self._filter_proj_z_params(
            self.actor_worker.actor_module.state_dict()
        )
        inference_checkpoint = {
            'step': step,
            'model_state_dict': inference_state_dict,
            'config': self.config
        }

        inference_path = os.path.join(self.config.trainer.save_dir, f'inference_checkpoint_{step}.pt')
        torch.save(inference_checkpoint, inference_path)

        return checkpoint_path


def main():
    """Main training function."""

    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'flowrl_config.yaml')
    if not os.path.exists(config_path):
        # Fallback to a default PPO config as base
        config_path = 'examples/ppo_trainer/config/qwen2_ppo.yaml'

    config = load_config_from_file(config_path)

    # Override config for FlowRL specific settings
    if not hasattr(config.algorithm, 'tb_coef'):
        config.algorithm.tb_coef = 15.0

    if not hasattr(config.actor, 'proj_layer'):
        config.actor.proj_layer = 3

    if not hasattr(config.actor, 'proj_dropout'):
        config.actor.proj_dropout = 0.1

    # Set algorithm type for logging
    config.algorithm.name = 'FlowRL'

    # Initialize trainer
    trainer = FlowRLTrainer(config)

    # Start training
    trainer.fit()


if __name__ == '__main__':
    main()