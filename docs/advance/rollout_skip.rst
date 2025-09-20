RolloutSkip Function Usage Documentation
========================================

Last updated: 08/31/2025.

Applicable Scenarios
--------------------

The RolloutSkip functionality is designed to accelerate the rollout process in reinforcement learning training by caching and reusing previously generated sequences. This feature is particularly useful when:

1. You need to repeatedly run experiments with the same configuration

2. You want to save time by avoiding redundant sequence generation to come close to the optimal policy


API and Usage Example
----------------------

2.1 Trainer Adaptation
~~~~~~~~~~~~~~~~~~~~~~

Both`RayDAPOTrainer()` (in `verl/recipe/dapo/dapo_ray_trainer.py`) and `RayPPOTrainer()`(in `verl/trainer/ppo/ray_trainer.py``) have already been adapted.

This is an example of how to patch rollout_skip in RayPPOTrainer.

.. code-block:: python

    #* Import the RolloutSkip class
    from verl.utils.rollout_skip import RolloutSkip

    ...
    class RayPPOTrainer:
        ...
        def fit(self):
            ...

            #* Add code as follow:
            rollout_skip = RolloutSkip(self.config)
            rollout_skip.wrap_generate_sequences(self.actor_rollout_wg)

            ...

            for epoch in range(self.config.trainer.total_epochs):
                for batch_dict in self.train_dataloader:
                    ...

2.2 Basic Configuration
~~~~~~~~~~~~~~~~~~~~~~~

Then, you should add the following parameters to your config to enable the RolloutSkip feature:

.. code-block:: bash

    actor_rollout_ref.rollout.skip.enable=True \
    actor_rollout_ref.rollout.skip.dump_step=1 \
    actor_rollout_ref.rollout.skip.post_dump_action="repeat" \
    actor_rollout_ref.rollout.skip.dump_dir="/tmp/rollout_dump" \


Note:

1. The `dump_dir` is the directory where the cached sequences will be stored. Ensure that this directory is writable and accessible by your training process. And make sure that `skip_dump_dir` is not relative path because ray will store the data in `/tmp/ray/session_<session_id>/` and the relative path will not be found in the worker.
2. The dumped data path follows this naming pattern `{experiment_name}_{project_name}/GBS{gen_batch_size}_N{n}_in{max_prompt_length}_out{max_response_length}`, once you change the `experiment_name`, `project_name`, `gen_gbs`, `n`, `max_prompt_length` or `max_response_length`, the cached data will be stored in a new directory.
