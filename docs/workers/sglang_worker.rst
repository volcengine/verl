SGLang Backend
==============
Author: `Yongan Xiang <https://github.com/BearBiscuit05>`_

介绍
----
SGLang 是一个用于大型语言模型和视觉语言模型的快速服务框架。它通过协同设计后端运行时和前端语言，让用户与模型的交互更快、更可控。
我们目前支持 SGLang 后端用于实现 rollout 阶段的 response 生成。目前也基于 `FSDPSGLangShardingManager <https://github.com/volcengine/verl/blob/main/verl/workers/sharding_manager/fsdp_sglang.py>`_ 实现了 SGLang 与 FSDP 之间的 resharding 功能。这表明我们可以在 verl 上基于 FSDP 和 SGLang 来进行 RL 训练任务。
目前的 SGLang 已经支持 memory save 和多机推理，并且在 verl 中对于推理框架参数的设置与 vLLM 相同，因此在两个推理框架之间的无缝切换仅需要修改 ``actor_rollout_ref.rollout.name=sglang`` 即可。

安装
----
首先需要按照 `Install SGLang as rollout backend 文档 <https://verl.readthedocs.io/en/latest/start/install.html#install-sglang-as-rollout-backend>`_ 里的要求进行安装，并且注意版本要求是否匹配。

.. code-block:: bash

    pip install -e "pip install "sglang[all]>=0.4.4.post4" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python

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

TODO
----
veRL-SGLang 项目目前正在快速推进，有多个功能正在支持中，相关开发可以查看 `[链接]()`。