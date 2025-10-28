一、组件版本信息
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
MindSpeed   main
=========== ============

二、 Dockerfile 构建镜像脚本
---------------------------

Dockerfile 脚本请参照 `此处 <https://github.com/volcengine/verl/blob/main/docker/Dockerfile.ascend_vllm-0.9.1>`_ 。


三、镜像构建命令示例
--------------------

.. code:: bash

   # Navigate to the directory containing the Dockerfile 
   cd /verl/docker
   # Build the image (specified tag: ascend-verl:cann82rc1_vllm091) 
   docker build -f Dockerfile.ascend.vllm-0.9.1 -t verl-ascend-vllm:cann8.2.rc1-vllm-0.9.1 .
