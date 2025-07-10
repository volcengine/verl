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
The abstract base class defining the interface for model training engines.
"""


class BaseEngine:
    """
    Abstract base class defining the interface for model training engines.

    Engine implementations must subclass BaseEngine and provide concrete behavior for all methods.
    """

    def __init__(self, config):
        """
        Initialize the BaseEngine.

        Args:
            config: Configuration object containing parameters for engine setup.
        """
        raise NotImplementedError

    def init_model(self):
        """
        Instantiate or load the model, optimizer, and learning rate scheduler.

        Should prepare all components necessary for training or evaluation.
        """
        raise NotImplementedError

    def train_mode(self):
        """
        Context manager entry for switching the engine and model into training mode.

        Usage:
            with engine.train_mode():
                # runs in training mode
        """
        raise NotImplementedError

    def eval_mode(self):
        """
        Context manager entry for switching the engine and model into evaluation mode.

        Usage:
            with engine.eval_mode():
                # runs in evaluation mode
        """
        raise NotImplementedError

    def infer_batch(self, batch, processor=None):
        """
        Execute a forward pass over a batch of data.

        Args:
            batch: Raw batch data (e.g., tensors or mappings) to process.
            ctx: Optional context dict passed to preprocess/postprocess functions.
            preprocess_fn: Function(batch, ctx) -> (inputs, ctx), applied before model call.
            postprocess_fn: Function(outputs, ctx) -> (predictions, ctx), applied after model call.

        Returns:
            (predictions, ctx)
        """
        raise NotImplementedError

    def train_batch(self, batch, metrics, processor=None):
        """
        Execute a forward pass and backward pass over a batch of data.

        Args:
            batch: Raw batch data (e.g., tensors or mappings) to process.
            ctx: Optional context dict passed to preprocess/postprocess functions.
            preprocess_fn: Function(batch, ctx) -> (inputs, ctx), applied before model call.
            postprocess_fn: Function(outputs, ctx) -> (predictions, ctx), applied after model call.

        Returns:
            (predictions, loss, ctx)
        """
        raise NotImplementedError

    def optimizer_zero_grad(self):
        """
        Zero out gradients of all parameters before starting a new backward pass.
        """
        raise NotImplementedError

    def optimizer_step(self):
        """
        Perform an optimization step to update model parameters based on accumulated gradients.

        Returns:
            grad_norm (float): The norm of the gradients before clipping or update.
        """
        raise NotImplementedError

    def lr_scheduler_step(self):
        """
        Advance the learning rate scheduler by one step.

        Returns:
            current_lr (float or list[float]): Updated learning rate(s).
        """
        raise NotImplementedError

    def shard_data(self, data):
        """
        Shard or partition data for distributed training or parallel execution.

        Args:
            data: Data structure to be sharded across devices/workers.

        Returns:
            Sharded data in the same format as input.
        """
        raise NotImplementedError

    def unshard_data(self, data):
        """
        Reconstruct or gather sharded data back to a unified format.

        Args:
            data: Sharded data structure to reconstruct.

        Returns:
            Unsharded, combined data.
        """
        raise NotImplementedError

    def set_loss_fn(self, loss_fn):
        """
        Set the loss function to be used during training.

        Args:
            loss_fn: Callable(data, predictions, ctx) -> (loss_tensor, new_ctx)
        """
        raise NotImplementedError

    def to(self, device: str, model: bool = True, optimizer: bool = True):
        """
        Move model parameters, optimizer states, or both to the specified device.

        Args:
            device: Target device identifier (e.g., "cuda" or "cpu").
            model: If True, move the model.
            optimizer: If True, move the optimizer states.
        """
        raise NotImplementedError

    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        """
        Save model, optimizer, and scheduler states to a checkpoint.

        Args:
            local_path: Local filesystem path to save checkpoint.
            hdfs_path: Optional HDFS path to copy checkpoint.
            global_step: Integer training step number for naming.
            max_ckpt_to_keep: Maximum number of recent checkpoints to retain.
        """
        raise NotImplementedError

    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=True):
        """
        Load model, optimizer, and scheduler states from a checkpoint.

        Args:
            local_path: Local filesystem path of the checkpoint.
            hdfs_path: Optional HDFS path where checkpoint is stored.
            del_local_after_load: Whether to delete local copy after loading.
        """
        raise NotImplementedError
