verl x Ascend
===================================

Last updated: 06/17/2025.

我们在 verl 上增加对华为昇腾设备的支持。

硬件支持
-----------------------------------

Atlas 200T A2 Box16

Atlas 900 A2 PODc


安装
-----------------------------------

基础环境准备
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+-----------+-------------+
| software  | version     |
+-----------+-------------+
| Python    | == 3.10     |
+-----------+-------------+
| CANN      | == 8.1.RC1  |
+-----------+-------------+
| torch     | == 2.5.1    |
+-----------+-------------+
| torch_npu | == 2.5.1.RC1|
+-----------+-------------+


vllm & vllm-ascend
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

为了能够在 verl 中正常使用 vllm，需使用以下命令编译安装 vllm 和 vllm-ascend。请注意根据机器类型区分安装方式。

.. code-block:: bash
    
    # vllm
    git clone -b v0.7.3 --depth 1 https://github.com/vllm-project/vllm.git
    cd vllm
    pip install -r requirements-build.txt

    # for Atlas 200T A2 Box16
    VLLM_TARGET_DEVICE=empty pip install -e . --extra-index https://download.pytorch.org/whl/cpu/
    
    # for Atlas 900 A2 PODc
    VLLM_TARGET_DEVICE=empty pip install -e .

.. code-block:: bash
    
    # vllm-ascend
    git clone -b v0.7.3.post1 --depth 1 https://github.com/vllm-project/vllm-ascend.git
    cd vllm-ascend
    export COMPILE_CUSTOM_KERNELS=1
    python setup.py install

安装verl
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    git clone https://github.com/volcengine/verl.git
    cd verl
    pip install -r requirements-npu.txt
    pip install -e .

其他三方库说明
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+--------------+---------------+
| software     | description   |
+--------------+---------------+
| transformers | v4.52.4       |
+--------------+---------------+
| flash_attn   | not supported |
+--------------+---------------+
| liger-kernel | not supported |
+--------------+---------------+
| tensordict   | 0.8.3 (ARM)   |
+--------------+---------------+

1. 支持通过 transformers 使能 --flash_attention_2， transformers 需大于等于 4.52.0版本。
2. 不支持通过 flash_attn 使能 flash attention 加速。
3. 不支持 liger-kernel 使能。
4. 针对 ARM 服务器，tensordict 要求 0.8.3，可在依赖安装完成后再手动安装 tensordict。
5. 针对 x86 服务器，需要安装 cpu 版本的 torchvision。

.. code-block:: bash

    pip install torchvision==0.20.1+cpu --index-url https://download.pytorch.org/whl/cpu


快速开始
-----------------------------------
正式使用前，建议您通过对Qwen2.5-0.5B GRPO的训练尝试以检验环境准备和安装的正确性。

1.下载数据集并将数据集预处理为parquet格式，以便包含计算RL奖励所需的必要字段

.. code-block:: bash

    python3 examples/data_preprocess/gsm8k.py --local_dir ~/data/gsm8k

2.执行训练

.. code-block:: bash

    set -x

    export VLLM_ATTENTION_BACKEND=XFORMERS

    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.train_files=$HOME/data/gsm8k/train.parquet \
        data.val_files=$HOME/data/gsm8k/test.parquet \
        data.train_batch_size=128 \
        data.max_prompt_length=512 \
        data.max_response_length=128 \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
        actor_rollout_ref.actor.optim.lr=5e-7 \
        actor_rollout_ref.model.use_remove_padding=False \
        actor_rollout_ref.actor.entropy_coeff=0.001 \
        actor_rollout_ref.actor.ppo_mini_batch_size=64 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=20 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=40 \
        actor_rollout_ref.rollout.enable_chunked_prefill=False \
        actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        actor_rollout_ref.rollout.n=5 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=40 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.kl_ctrl.kl_coef=0.001 \
        trainer.critic_warmup=0 \
        trainer.logger=console \
        trainer.project_name='verl_grpo_example_gsm8k' \
        trainer.experiment_name='qwen2_7b_function_rm' \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=1 \
        trainer.save_freq=-1 \
        trainer.test_freq=5 \
        trainer.total_epochs=1 \
        trainer.device=npu $@

MindSpeed 训练后端
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1. 参考 `MindSpeed Readme <https://gitee.com/ascend/MindSpeed>`_ 说明安装 MindSpeed 加速库。

2. 使能 Verl worker 模型 ``strategy`` 配置为 ``megatron`` ，例如 ``actor_rollout_ref.actor.strategy=megatron``。

3. MindSpeed 自定义入参可通过 ``override_transformer_config`` 参数传入，例如对 actor 模型开启 FA 特性可使用 ``+actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True``。

4. 更多特性信息可参考 `MindSpeed Verl 文档 <https://gitee.com/ascend/MindSpeed/blob/master/docs/user-guide/verl.md>`_ 。

支持现状
-----------------------------------

+-----------+-------------------------+-------------+-------------------+----------------------+
| algorithm |         model           | rewards mae |  throughput ratio |        hardware      |
+-----------+-------------------------+-------------+-------------------+----------------------+
|   GRPO    | Qwen2.5-7B-instruct     |    0.38%    |        0.588      |  Atlas 200T A2 Box16 |
+-----------+-------------------------+-------------+-------------------+----------------------+
|   GRPO    | Qwen2.5-32B-instruct    |    0.30%    |        0.685      |  Atlas 200T A2 Box16 |
+-----------+-------------------------+-------------+-------------------+----------------------+
|   GRPO    | Qwen2.5-VL-3B-instruct  |    3.14%    |        0.470      |  Atlas 200T A2 Box16 |
+-----------+-------------------------+-------------+-------------------+----------------------+
|   GRPO    | Qwen2.5-VL-7B-instruct  |    3.30%    |        0.380      |  Atlas 200T A2 Box16 |
+-----------+-------------------------+-------------+-------------------+----------------------+
|   GRPO    | Qwen2.5-VL-32B-instruct |    0.79%    |        0.568      |  Atlas 200T A2 Box16 |
+-----------+-------------------------+-------------+-------------------+----------------------+
|   DAPO    | Qwen2.5-7B-instruct     |    3.83%    |        pending    |  Atlas 200T A2 Box16 |
+-----------+-------------------------+-------------+-------------------+----------------------+
|   DAPO    | Qwen3-8B-base           |    5.3%     |        pending    |  Atlas 200T A2 Box16 |
+-----------+-------------------------+-------------+-------------------+----------------------+
|   DAPO    | Qwen3-14B-base          |    5.9%     |        pending    |  Atlas 200T A2 Box16 |
+-----------+-------------------------+-------------+-------------------+----------------------+
|  SFT-PEFT | Qwen2.5-0.5B-instruct   |    0.06%    |        0.305      |  Atlas 900 A2 PODc   |
+-----------+-------------------------+-------------+-------------------+----------------------+

精度对比说明
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

对于 SFT 类算法，我们期望在相同配置下华为昇腾设备与 A100 的 loss 平均绝对误差<= 2%。计算方式如下图。更多信息请参考 `精度计算说明 <https://www.hiascend.com/document/detail/zh/Pytorch/600/ptmoddevg/trainingmigrguide/LMaccuracy_0001.html>`_。

.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/loss_comparison.png?raw=true
   :alt: loss_comparison

根据经验，对于 GRPO 等 RL 类算法，我们期望在相同配置下华为昇腾设备与 A100 的 rewards 平均绝对误差<= 4%，计算方式参考上图。


吞吐对比说明
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Ascend npu 和 A100 分别取日志中前4个 step 的 "perf/throughput" 做平均， throughput ratio = npu 平均值 / A100 平均值。 


计划
-----------------------------------

查看 `roadmap <https://github.com/volcengine/verl/discussions/900>`_ 获取更多特性的支持进度。



声明
-----------------------------------
verl中提供的ascend支持代码皆为参考样例，商业使用请通过官方正式途径沟通，谢谢。
