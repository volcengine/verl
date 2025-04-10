SGLang Backend
==============
Author: `Yongan Xiang <https://github.com/BearBiscuit05>`_, `Chenyang Zhao <https://github.com/zhaochenyang20>`_, `Junrong Lin <https://github.com/ocss884>`_

介绍
----
`SGLang <https://github.com/sgl-project/sglang>`_ 是开源 SOTA 的推理服务引擎，被 xAI 全面采用，用于支持 grok 在研究和 serving 过程中的所有推理需求。

目前，veRL 全面支持采用 SGLang 作为 rollout 阶段的推理引擎。作为 rollout engine，目前 SGLang 和 vLLM 完全一致，包括 memory save 和 multi-node rollout。安装完成 veRL 和 SGLang 后，在启动时添加 ``actor_rollout_ref.rollout.name=sglang``，即可在两个推理框架之间顺利切换。

此外，SGLang 团队正在全力支持 Multi-Turn Agentic RL、VLM RLHF、Server-Based RLHF 以及 Partial Rollout 等功能，相关的开发进度可以参考此处的 `Tracking Roadmap <https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/issues/74>`_。

安装
----
首先需要按照 `Install SGLang as rollout backend <https://verl.readthedocs.io/en/latest/start/install.html#install-sglang-as-rollout-backend>`_ 里的要求进行安装，并且注意版本要求是否匹配。基本上，采用 main branch 最新的 `SGLang <https://github.com/sgl-project/sglang>`_ 就可以稳定启动训练，不用追求特定的版本。

.. code-block:: bash

    # 目前是 0.4.5，随时可能更新，请参考最新的版本
    pip install "sglang[all]>=0.4.5" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python

选择SGLang为推理后端在单机上进行PPO训练
--------------------------------------
我们使用 Qwen/Qwen2-7B-Instruct 在 gsm8k 上训练来进行简单的测试。

1. 运行下面的命令来准备 gsm8k 数据集：

.. code-block:: bash

    python3 examples/data_preprocess/gsm8k.py

2. 运行下面的脚本在单机上使用4卡进行PPO实验：

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

选择SGLang为推理后端在多机上进行PPO训练
--------------------------------------
SGLang 同样支持在 IPv4 和 IPv6 的场景下运行 veRL 中基于 RAY 的跨机推理。下面的脚本中我们使用了 TP=16 来进行跨机推理。现假设我们有两台互联的机器，node0 的 ip 为 10.94.16.4，node1 的 ip 为 10.94.16.5。

1. 在 node0 启动 ray：

.. code-block:: bash

    ray start --head --dashboard-host=0.0.0.0

可以看到下面的提示：

.. code-block:: bash

    Usage stats collection is enabled. To disable this, add `--disable-usage-stats` to the command that starts the cluster, or run the following command: `ray disable-usage-stats` before starting the cluster. See https://docs.ray.io/en/master/cluster/usage-stats.html for more details.

    Local node IP: 10.94.16.4

    --------------------
    Ray runtime started.
    --------------------

    Next steps
    To add another node to this Ray cluster, run
        ray start --address='10.94.16.4:6379'

2. 令 node1 加入 ray cluster：

在 node1 上运行下面的命令：

.. code-block:: bash

    ray start --address='10.94.16.4:6379'

运行下面的命令确认此时 ray cluster 里有两个节点：

.. code-block:: bash

    ray status

可以看到 cluster 上有两个节点，16 张 GPU：

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

3. 运行下面的脚本在2台机器上使用16张卡TP16训练 meta-llama/Llama-3.1-8B-Instruct：

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
