Ascend Quickstart with SGLang Backend
===================================

Last updated: 09/25/2025.

我们在 verl 上增加对华为昇腾设备的支持。

硬件支持
-----------------------------------

Atlas 200T A2 Box16

Atlas 900 A2 PODc

Atlas 800T A3


安装
-----------------------------------

基础环境准备
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+-----------+-------------+
| software  | version     |
+-----------+-------------+
| Python    | == 3.11     |
+-----------+-------------+
| CANN      | == 8.3.RC1  |
+-----------+-------------+
| HDK       | == 25.3.RC1 |
+-----------+-------------+
| torch     | == 2.6.0    |
+-----------+-------------+
| torch_npu | == 2.6.0    |
+-----------+-------------+

**目前verl框架中sglang npu后端仅支持上述HDK、CANN和PTA版本, 商发可用版本预计2025年10月发布**

为了能够在 verl 中正常使用 sglang，需使用以下命令安装sglang、torch_memory_saver和verl。

sglang
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: bash
    
    # sglang
    git clone https://github.com/sgl-project/sglang.git
    cd sglang
    mv python/pyproject.toml python/pyproject.toml.backup
    mv python/pyproject_other.toml python/pyproject.toml
    pip install -e "python[srt_npu]"

安装torch_memory_saver
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: bash
    
    # torch_memory_saver
    git clone https://github.com/sgl-project/sgl-kernel-npu.git
    cd sgl-kernel-npu
    bash build.sh  -a memory-saver
    pip install output/torch_memory_saver*.whl

安装verl
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    git clone https://github.com/volcengine/verl.git
    cd verl
    pip install --no-deps -e .
    pip install -r requirements-npu.txt 


其他三方库说明
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+--------------+---------------+
| software     | description   |
+--------------+---------------+
| transformers | v4.56.1       |
+--------------+---------------+
| triton_ascend| v3.2.0        |
+--------------+---------------+

1. sglang依赖 transformers v4.56.1
2. sglang依赖triton_ascend v3.2.0
3. 暂不支持多模态模型，卸载相关安装包torchvision、timm

.. code-block:: bash
    
    pip uninstall torchvision
    pip uninstall timm
    pip uninstall triton
    
    pip install transformers==4.56.1
    pip install -i https://test.pypi.org/simple/ triton-ascend==3.2.0.dev20250925


快速开始
-----------------------------------
正式使用前，建议您通过对Qwen3-8B GRPO的训练尝试以检验环境准备和安装的正确性。

1.下载数据集并将数据集预处理为parquet格式，以便包含计算RL奖励所需的必要字段

.. code-block:: bash

    python3 examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k

2.执行训练

.. code-block:: bash

    bash verl/examples/grpo_trainer/run_qwen3_8b_grpo_sglang_1k_npu.sh