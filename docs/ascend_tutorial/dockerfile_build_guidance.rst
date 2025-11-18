Ascend Dockerfile Build Guidance
===================================

Last updated: 11/14/2025.

我们在verl上增加对华为昇腾镜像构建的支持。


硬件支持
-----------------------------------

Atlas 200T A2 Box16

Atlas 900 A2 PODc

Atlas 800T A3


组件版本信息
----------------

=========== ============
组件        版本
=========== ============
基础镜像    Ubuntu 22.04
Python      3.11
CANN        8.3.RC1
torch       2.7.1
torch_npu   2.7.1
vLLM        0.11.0
vLLM-ascend 0.11.0rc1
Megatron-LM v0.12.1
MindSpeed   (f2b0977e)
=========== ============

二、 Dockerfile 构建镜像脚本
---------------------------

============== ============== ==============
设备类型         基础镜像版本     参考文件
============== ============== ==============
A2              8.2.RC1       `Dockerfile.ascend_8.2.rc1_a2 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_8.2.rc1_a2>`_
A2              8.3.RC1       `Dockerfile.ascend_8.3.rc1_a2 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_8.3.rc1_a2>`_
A3              8.2.RC1       `Dockerfile.ascend_8.2.rc1_a3 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_8.2.rc1_a3>`_
A3              8.3.RC1       `Dockerfile.ascend_8.3.rc1_a3 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_8.3.rc1_a3>`_
============== ============== ==============


三、镜像构建命令示例
--------------------

.. code:: bash

   # Navigate to the directory containing the Dockerfile 
   cd /verl/docker
   # Build the image (specified tag: ascend-verl:cann82rc1_vllm091) 
   docker build -f Dockerfile.ascend.vllm-0.9.1 -t verl-ascend-vllm:cann8.2.rc1-vllm-0.9.1 .
