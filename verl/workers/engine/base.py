class BaseEngine(object):
    def __init__(self, config):
        """
        Initialize the BaseEngine instance.

        This method serves as the constructor for the BaseEngine class. BaseEngine 
        is likely an abstract base class.

        Args:
            config: A configuration object containing parameters and settings
                    required for initializing the engine.
        """
        raise NotImplementedError

    def init_model(self):
        """
        Initialize the model and its corresponding optimizer.
        """
        raise NotImplementedError

    def train_mode(self):
        """
        Set the engine to training mode. This method is designed to switch the engine 
        and its associated model(s) into training mode. 
        """
        raise NotImplementedError

    def eval_mode(self):
        """
        Set the engine to eval mode. This method is designed to switch the engine 
        and its associated model(s) into eval mode. 
        """
        raise NotImplementedError

    def forward_backward_step(self, 
                              batch, 
                              ctx=None, 
                              forward_only=False, 
                              preprocess_fn=None, 
                              postprocess_fn=None):
        """
        inputs, ctx = preprocess_fn(batch, ctx)
        preds, ctx = postprocess_fn(outputs, ctx)
        """
        raise NotImplementedError

    def optimizer_zero_grad(self):
        """
        Zero out the gradients of all the parameters optimized by the optimizer.
        Before starting a new round of gradient computation for a fresh batch of data,
        it's necessary to zero out the gradients to prevent the gradients from being
        added to the previous ones.
        """
        raise NotImplementedError

    def optimizer_step(self):
        """
        Perform a single optimization step.
        This method is designed to update the model's parameters based on the computed gradients.
        """
        raise NotImplementedError

    def lr_scheduler_step(self):
        """
        Perform a single step of the learning rate scheduler.

        In the process of model training, the learning rate scheduler is used to adjust 
        the learning rate dynamically over time, which can help the model converge better.
        This method is designed to trigger the learning rate adjustment mechanism of the scheduler.
        For example, it can be used to decay the learning rate after a certain number of epochs 
        or when the validation loss plateaus.
        """
        raise NotImplementedError

    def shard_data(self, data):
        raise NotImplementedError

    def unshard_data(self, data):
        raise NotImplementedError
        

    def set_loss_fn(self, loss_fn):
        """
        loss_fn(data, preds, ctx) -> loss, ctx
        """
        raise NotImplementedError

    def to(self, device: str, model: bool = True, optimizer: bool = True):
        raise NotImplementedError


    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        raise NotImplementedError


    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=True):
        raise NotImplementedError