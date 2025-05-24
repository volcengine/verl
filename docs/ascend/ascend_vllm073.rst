verl x Ascend
========

我们在 verl 上增加对华为昇腾设备的支持。

硬件支持
=======

* Atlas 800T A2

* Atlas 200T A2 Box16

安装
=======

环境准备
------

+-----------+-------------+
| software  | version     |
+-----------+-------------+
| Python    | == 3.10     |
+-----------+-------------+
| torch     | == 2.5.1    |
+-----------+-------------+
| torch_npu | == 2.5.1rc1 |
+-----------+-------------+
| CANN      | == 8.1.RC1  |
+-----------+-------------+

1. 为了能够在 ASCEND NPU 上正常使能 flash_attention_2， transformers 版本需要大于等于 4.52.0。
2. 目前支持 SFT 与 LLM 模型的 GRPO 训练，VLM模型的 GRPO 训练因为 vllm-ascend 的问题将会在后续支持，涉及到的issue为：

    https://github.com/vllm-project/vllm-ascend/issues/809

    https://github.com/vllm-project/vllm-ascend/issues/825

源码安装
------

.. code-block:: bash

    git clone https://github.com/volcengine/verl.git
    cd verl
    pip install -r requirements-npu.txt
    pip install -e .

vLLM
------

为了保证能够在 verl 上正常使用 vLLM，需要使用以下命令编译安装 vLLM 和 vLLM Ascend 插件（`vllm-ascend`）。

.. code-block:: bash

    git clone -b v0.7.3 --depth 1 https://github.com/vllm-project/vllm.git
    cd vllm
    pip install -r requirements-build.txt
    VLLM_TARGET_DEVICE=empty pip install -e . --extra-index https://download.pytorch.org/whl/cpu/

.. code-block:: bash
    
    git clone -b v0.7.3 --depth 1 https://github.com/vllm-project/vllm-ascend.git
    cd vllm-ascend
    export COMPILE_CUSTOM_KERNELS=1
    python setup.py install

其他第三方库说明
------

+--------------+---------------+
| software     | description   |
+--------------+---------------+
| flash_attn   | not supported |
+--------------+---------------+
| liger-kernel | not supported |
+--------------+---------------+

精度对比
------

根据经验，对于SFT等微调算法，我们期望在相同配置下，在华为昇腾设备上的 Loss 与英伟达 GPU 的 Loss 平均绝对误差小于等于 2%，具体计算方式如下：

.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/loss_comparison.png?raw=true
   :alt: loss_comparison

其中，N 表示训练的步数。更多信息请参考[精度计算说明](https://www.hiascend.com/document/detail/zh/Pytorch/600/ptmoddevg/trainingmigrguide/LMaccuracy_0001.html)。

根据经验，对于GRPO等强化学习算法，我们期望在相同配置下，在华为昇腾设备上的 reward 与英伟达 GPU 的 reward 平均绝对误差小于等于 4%，具体计算参考 Loss 计算。

进展
------

+-----------+-------------+
| algorithm | description |
+-----------+-------------+
|    SFT    |  supported  |
+-----------+-------------+
|    GRPO   |  supported  |
+-----------+-------------+
