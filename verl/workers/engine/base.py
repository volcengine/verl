from contextlib import nullcontext

"""
TODO:
- checkpointing
"""

class BaseEngine(object):
    def __init__(self, config):
        raise NotImplementedError

    def init_model_and_optimizer(self):
        raise NotImplementedError

    def train_mode(self):
        raise NotImplementedError

    def eval_mode(self):
        raise NotImplementedError

    def forward_backward_step(self, batch, forward_only=False, ctx=None):
        """
        return
        - preds
        - loss
        - out_ctx
        """
        raise NotImplementedError

    def optimizer_zero_grad(self):
        raise NotImplementedError

    def optimizer_step(self):
        raise NotImplementedError

    def lr_scheduler_step(self):
        raise NotImplementedError

    def shard_data(self, data):
        raise NotImplementedError

    def unshard_data(self, data):
        raise NotImplementedError

    def set_preprocess_fn(self, preprocess_fn):
        """
        preprocess_fn(data, ctx) -> inputs, ctx
        """
        raise NotImplementedError


    def set_postprocess_fn(self, postprocess_fn):
        """
        postprocess_fn(outputs, ctx) -> preds, ctx
        """
        raise NotImplementedError
        

    def set_loss_fn(self, loss_fn):
        """
        loss_fn(data, preds, ctx) -> loss, out_ctx
        """
        raise NotImplementedError

    def to():
        raise NotImplementedError