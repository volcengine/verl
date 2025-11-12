Ascend Dockerfile Build Guidance
===================================

Last updated: 10/31/2025.

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
CANN        8.2.RC1
torch       2.5.1
torch_npu   2.5.1
vLLM        0.9.1
vLLM-ascend 0.9.1
Megatron-LM v0.12.1
MindSpeed   (f2b0977e)
=========== ============

Dockerfile构建镜像脚本
---------------------------

============== ============== ==============
设备类型         基础镜像版本     参考文件
============== ============== ==============
A2              8.2.RC1       `Dockerfile.ascend_8.2.rc1_a2 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_8.2.rc1_a2>`_
A3              8.2.RC1       `Dockerfile.ascend_8.2.rc1_a3 <https://github.com/volcengine/verl/blob/main/docker/ascend/Dockerfile.ascend_8.2.rc1_a3>`_
============== ============== ==============


镜像构建命令示例
--------------------

.. code:: bash

   # Navigate to the directory containing the Dockerfile 
   cd {verl-root-path}/docker/ascend
   # Build the image
   docker build -f Dockerfile.ascend_8.2.rc1_a2 -t verl-ascend:8.2.rc1-a2 .


声明
--------------------
verl中提供的ascend相关Dockerfile、镜像皆为参考样例，可用于尝鲜体验，如在生产环境中使用请通过官方正式途径沟通，谢谢。