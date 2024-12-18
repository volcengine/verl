.. _quickstart:

=========================================================
Quickstart: Post-train a LLM using PPO with GSM8K dataset
=========================================================

Post-train a LLM using GSM8K dataset
===================================================================

Introduction
------------

.. _hf_dataset_gsm8k: https://huggingface.co/datasets/gsm8k

In this example, we train an LLM to tackle the `GSM8k <hf_dataset_gsm8k>`_ task with function-based rewards[1]_.

Prerequisite:

- the latest version of ``verl`` and its dependencies installed following the installation guide. Using the docker image is recommended.

- an GPU with at least 32 GB memory


Dataset Introduction
--------------------

GSM8k is a math problem dataset. The prompt is an elementary school
problem. The LLM model is asked to solve the math problem. Below is an example:

Prompt

   Katy makes coffee using teaspoons of sugar and cups of water in the
   ratio of 7:13. If she used a total of 120 teaspoons of sugar and cups
   of water, calculate the number of teaspoonfuls of sugar she used.

Solution

   The total ratio representing the ingredients she used to make the
   coffee is 7+13 = <<7+13=20>>20 Since the fraction representing the
   number of teaspoons she used is 7/20, she used 7/20\ *120 =
   <<7/20*\ 120=42>>42 #### 42

Step 1: Prepare the dataset
----------------------------

We preprocess the dataset in parquet format so that (1) it contains necessary fields for computing RL rewards and (2) is faster to read.

.. code:: bash

   python3 examples/data_preprocess/gsm8k.py --local_dir ~/data/gsm8k

Step 2: Download a model for post-training
-------------------------------------------

Usually we recommend starting with an "instruct" model variant so that the model follows instructions. In this example, we start with the ``Qwen2.5-0.5B-Instruct`` model.

If you start from a "base" model variant, doing SFT before RL is recommended. Refer to the `sft directory <https://github.com/volcengine/verl/blob/main/examples/gsm8k/sft/>`_ and `SFT Trainer <https://github.com/volcengine/verl/blob/main/verl/trainer/fsdp_sft_trainer.py>`_ for further details.

.. code:: bash

   python3 -c "import transformers; transformers.pipeline('text-generation', model='Qwen/Qwen2.5-0.5B-Instruct')"

Step 3: Perform PPO training with the instruct model
----------------------------------------------------------------------

**Reward Model/Function**

We use a pre-defined rule-based reward model. We force the model to produce a final
answer following 4 “#” as shown in the solution. We extract the final
answer from both the solution and model's output using regular
expression matching. We assign a reward of 1 to correct
answer, 0.1 to incorrect answer and 0 to no answer. 

For mode details, please refer to `verl/utils/reward_score/gsm8k.py <https://github.com/volcengine/verl/blob/v0.1/verl/utils/reward_score/gsm8k.py>`_.

**Training Script**

Now let's run PPO training with the dataset and model above[2]_.

.. code:: bash

   bash examples/ppo_trainer/run_deepseek7b_llm.sh

The script of `run_deepseek7b_llm.sh`

- Prepare your own run.sh script. Here's an example for GSM8k dataset
  and deepseek-llm-7b-chat model.
- Users could replace the ``data.train_files`` ,\ ``data.val_files``,
  ``actor_rollout_ref.model.path`` and ``critic.model.path`` based on
  their environment.
- See :doc:`examples/config` for detailed explaination of each config field.

.. code:: bash

   python3 -m verl.trainer.main_ppo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=512 \
    data.val_batch_size=1312 \
    data.max_prompt_length=512 \
    data.max_response_length=256 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    critic.optim.lr=1e-5 \
    critic.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    critic.ppo_micro_batch_size=1 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger=['console'] \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=15 $@ 2>&1 | tee verl_demo.log


# checkpoints/${trainer.project_name}/${trainer.experiment_name}

# trainer.logger=['console','wandb']
# trainer.project_name='verl_post_training' \
# trainer.experiment_name='gsm8k_function_rm' \

# actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
# critic.model.fsdp_config.optimizer_offload=False \



.. [1] The original paper (https://arxiv.org/pdf/2110.14168) mainly focuses on training a verifier (a reward model) to solve math problems via Best-of-N sampling. In this example, we train an RL agent using a rule-based reward model.
.. [2] More training script examples for FSDP and Megatron-LM backend are stored in `examples/ppo_trainer <https://github.com/volcengine/verl/tree/main/examples/ppo_trainer>`_ directory.