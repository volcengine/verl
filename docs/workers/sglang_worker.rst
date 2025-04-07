SGLang Backend
==============
Author: `Yongan Xiang <https://github.com/BearBiscuit05>`_, `Chenyang Zhao <https://github.com/zhaochenyang20>`_

介绍
----
`SGLang <https://github.com/sgl-project/sglang>`_ 是开源 SOTA 的推理服务引擎，被 xAI 全面采用，用于支持 grok 在研究和 serving 过程中的所有推理需求。

目前，veRL 全面支持采用 SGLang 作为 rollout 阶段的推理引擎。作为 rollout engine，目前 SGLang 和 vllm 完全一致，包括 memory save 和 multi-node rollout。安装完成 veRL 和 SGLang 后，在启动时添加 ``actor_rollout_ref.rollout.name=sglang``，即可在两个推理框架之间顺利切换。

此外，SGLang 团队正在全力支持 Multi-Turn Agentic RL，VLM RLHF，Sever-Based RLHF 以及 Partial Rollout 等功能，相关的开发进度可以参考此处的 `Tracking Roadmap <https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/issues/74>`_。

安装
----
首先需要按照 `Install SGLang as rollout backend <https://verl.readthedocs.io/en/latest/start/install.html#install-sglang-as-rollout-backend>`_ 里的要求进行安装，并且注意版本要求是否匹配。基本上，采用 main branch 最新的 `SGLang <https://github.com/sgl-project/sglang>`_ 就可以稳定启动训练，不用追求特定的版本。

.. code-block:: bash
    # 目前是 0.4.5，随时可能更新，请参考最新的版本
    pip install "sglang[all]>=0.4.5" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python

SGLang 在单机运行
------------------
我们使用 Qwen-0.5B 来进行简单的测试，有关数据集和模型的下载，你可以参照 `Quickstart <https://verl.readthedocs.io/en/latest/start/quickstart.html#step-1-prepare-the-dataset>`_ 来安装模型以及数据集。对于测试 SGLang 是否有效执行，最直接的方法是运行 ``main_generation.py`` 来进行测试。

.. code-block:: bash

    python3 -m verl.trainer.main_generation \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=2 \
        data.path=~/data/rlhf/gsm8k/test.parquet \
        data.prompt_key=prompt \
        data.n_samples=1 \
        data.output_path=~/data/rlhf/math/deepseek_v2_lite_gen_test.parquet \
        model.path=Qwen/Qwen2.5-0.5B-Instruct \
        +model.trust_remote_code=True \
        rollout.temperature=1.0 \
        rollout.name=sglang \
        rollout.top_k=50 \
        rollout.top_p=0.7 \
        rollout.prompt_length=2048 \
        rollout.response_length=1024 \
        rollout.tensor_model_parallel_size=2 \
        rollout.gpu_memory_utilization=0.8

如果想测试 RL 相关算法，可以测试以下代码：

.. code-block:: bash

    PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
        data.train_files=$HOME/data/gsm8k/train.parquet \
        data.val_files=$HOME/data/gsm8k/test.parquet \
        data.train_batch_size=256 \
        data.max_prompt_length=512 \
        data.max_response_length=256 \
        actor_rollout_ref.model.path=deepseek-ai/deepseek-llm-7b-chat \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.ppo_mini_batch_size=64 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.rollout.name=sglang \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
        critic.optim.lr=1e-5 \
        critic.model.path=deepseek-ai/deepseek-llm-7b-chat \
        critic.ppo_micro_batch_size_per_gpu=4 \
        algorithm.kl_ctrl.kl_coef=0.001 \
        trainer.logger=['console'] \
        trainer.val_before_train=False \
        trainer.default_hdfs_dir=null \
        trainer.n_gpus_per_node=1 \
        trainer.nnodes=1 \
        trainer.save_freq=10 \
        trainer.test_freq=10 \
        trainer.total_epochs=15 2>&1 | tee verl_demo.log

SGLang 在多机下运行
-------------------
SGLang 同样支持在 IPv4 和 IPv6 的场景下运行 verl 中基于 RAY 的多机推理。下面的脚本是在 TP 设置大于一台机器中总卡数的情况下来进行的测试，用于验证 SGLang 的多机推理。

.. code-block:: bash

    python3 -m verl.trainer.main_generation \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=8 \
        data.path=~/data/rlhf/gsm8k/test.parquet \
        data.prompt_key=prompt \
        data.n_samples=1 \
        data.output_path=~/data/rlhf/math/deepseek_v2_lite_gen_test.parquet \
        model.path=deepseek-ai/deepseek-llm-7b-chat \
        +model.trust_remote_code=True \
        rollout.temperature=1.0 \
        rollout.name=sglang \
        rollout.top_k=50 \
        rollout.top_p=0.7 \
        rollout.prompt_length=2048 \
        rollout.response_length=1024 \
        rollout.tensor_model_parallel_size=16 \
        rollout.gpu_memory_utilization=0.8
