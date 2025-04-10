SGLang Backend
==============
Author: `Yongan Xiang <https://github.com/BearBiscuit05>`_, `Chenyang Zhao <https://github.com/zhaochenyang20>`_, `Junrong Lin <https://github.com/ocss884>`_

Introduction
------------
`SGLang <https://github.com/sgl-project/sglang>`_ is an open-source state-of-the-art inference service engine, fully adopted by xAI to support all inference needs of Grok during research and serving processes.

Currently, verl fully supports using SGLang as the inference engine during the rollout phase. As a rollout engine, SGLang provides the same feature coverage as vLLM., including memory saving and multi-node rollout features. After installing verl and SGLang, simply add ``actor_rollout_ref.rollout.name=sglang`` at startup to seamlessly switch between the two inference frameworks.

In addition, the SGLang team is actively working on supporting features such as Multi-Turn Agentic RL, VLM RLHF, Server-Based RLHF, and Partial Rollout. You can track the related development progress in the `Tracking Roadmap <https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/issues/74>`_.

Installation
------------
First, follow the requirements outlined in `Install SGLang as rollout backend <https://verl.readthedocs.io/en/latest/start/install.html#install-sglang-as-rollout-backend>`_ for installation, and ensure that the version requirements are met. Generally, using the latest `SGLang <https://github.com/sgl-project/sglang>`_ from the main branch will allow stable training startup without needing to target a specific version.

.. code-block:: bash

    # Currently 0.4.5, subject to updates at any time, please refer to the latest version
    pip install "sglang[all]>=0.4.5" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python

Using SGLang as the Inference Backend for PPO Training on a Single Machine
-------------------------------------------------------------------------
We use Qwen/Qwen2-7B-Instruct on the gsm8k dataset for a simple test.

1. Run the following command to prepare the gsm8k dataset:

.. code-block:: bash

    python3 examples/data_preprocess/gsm8k.py

2. Run the following script to conduct a PPO experiment on a single machine with 4 GPUs:

.. code-block:: bash

    PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
        data.train_files=$HOME/data/gsm8k/train.parquet \
        data.val_files=$HOME/data/gsm8k/test.parquet \
        data.train_batch_size=4096 \
        data.max_prompt_length=4096 \
        data.max_response_length=4096 \
        actor_rollout_ref.rollout.name=sglang \
        actor_rollout_ref.model.path=Qwen/Qwen2-7B-Instruct \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.ppo_mini_batch_size=64 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
        critic.optim.lr=1e-5 \
        critic.model.path=Qwen/Qwen2-7B-Instruct \
        critic.ppo_micro_batch_size_per_gpu=4 \
        critic.model.fsdp_config.param_offload=True \
        critic.model.fsdp_config.optimizer_offload=True \
        algorithm.kl_ctrl.kl_coef=0.001 \
        trainer.logger=['console'] \
        trainer.val_before_train=False \
        trainer.default_hdfs_dir=null \
        trainer.n_gpus_per_node=4 \
        trainer.nnodes=1 \
        trainer.save_freq=-1 \
        trainer.test_freq=10 \
        trainer.total_epochs=15 2>&1 | tee verl_demo.log

Using SGLang as the Inference Backend for PPO Training Across Multiple Machines
------------------------------------------------------------------------------
SGLang also supports running verl's RAY-based cross-machine inference in IPv4 and IPv6 scenarios. In the script below, we use TP=16 for cross-machine inference. Suppose we have two interconnected machines: node0 with IP 10.94.16.4 and node1 with IP 10.94.16.5.

1. Start Ray on node0:

.. code-block:: bash

    ray start --head --dashboard-host=0.0.0.0

You will see the following prompt:

.. code-block:: bash

    Usage stats collection is enabled. To disable this, add `--disable-usage-stats` to the command that starts the cluster, or run the following command: `ray disable-usage-stats` before starting the cluster. See https://docs.ray.io/en/master/cluster/usage-stats.html for more details.

    Local node IP: 10.94.16.4

    --------------------
    Ray runtime started.
    --------------------

    Next steps
    To add another node to this Ray cluster, run
        ray start --address='10.94.16.4:6379'

2. Have node1 join the Ray cluster:

Run the following command on node1:

.. code-block:: bash

    ray start --address='10.94.16.4:6379'

Run the following command to confirm that the Ray cluster now has two nodes:

.. code-block:: bash

    ray status

You can see that the cluster has two nodes with 16 GPUs:

.. code-block:: bash

    ======== Autoscaler status: 2025-04-09 09:25:37.694016 ========
    Node status
    ---------------------------------------------------------------
    Active:
     1 node_ef382ffd687d8f6b060c1b68e63ada7341b936fe5b1901dd04de1027
     1 node_1eb4d7d07e793114c23a89d1a41f1f76acf6ef5b35af844a4ee8e4ba
    Pending:
     (no pending nodes)
    Recent failures:
     (no failures)

    Resources
    ---------------------------------------------------------------
    Usage:
     0.0/360.0 CPU
     0.0/16.0 GPU
     0B/3.39TiB memory
     0B/372.53GiB object_store_memory

3. Run the following script to train meta-llama/Llama-3.1-8B-Instruct with TP=16 across 2 machines using 16 GPUs:

.. code-block:: bash

    DATA_DIR=$HOME/data/gsm8k

    python3 -m verl.trainer.main_ppo \
        actor_rollout_ref.rollout.name=sglang \
        data.train_files=$DATA_DIR/train.parquet \
        data.val_files=$DATA_DIR/test.parquet \
        data.train_batch_size=4096 \
        data.max_prompt_length=4096 \
        data.max_response_length=4096 \
        actor_rollout_ref.model.path=meta-llama/Llama-3.1-8B-Instruct \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=64 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=16 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
        actor_rollout_ref.rollout.free_cache_engine=True \
        actor_rollout_ref.ref.log_prob_micro_batch_size=16 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        critic.optim.lr=1e-5 \
        critic.model.use_remove_padding=True \
        critic.model.path=meta-llama/Llama-3.1-8B-Instruct \
        critic.model.enable_gradient_checkpointing=True \
        critic.ppo_micro_batch_size=16 \
        critic.model.fsdp_config.param_offload=True \
        critic.model.fsdp_config.optimizer_offload=True \
        algorithm.kl_ctrl.kl_coef=0.001 \
        trainer.critic_warmup=0 \
        trainer.logger=['console'] \
        trainer.val_before_train=True \
        trainer.default_hdfs_dir=null \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=2 \
        trainer.save_freq=-1 \
        trainer.test_freq=10 \
        trainer.total_epochs=15 2>&1 | tee verl_demo.log
