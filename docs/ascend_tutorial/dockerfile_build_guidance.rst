一、组件版本信息
----------------

=========== ============
组件        版本
=========== ============
CANN        8.2.RC1
vLLM        0.9.1
vLLM-ascend 0.9.1
Megatron-LM core_v0.12.1
Python      3.11
基础镜像    Ubuntu 22.04
=========== ============

二、 Dockerfile 构建镜像脚本
---------------------------

Dockerfile 脚本请参照 `此处 <https://github.com/songyy29/verl/blob/main/docker/Dockerfile.ascend_vllm-0.9.1>`_ 。


三、镜像构建命令示例
--------------------

.. code:: bash

   # Navigate to the directory containing the Dockerfile 
   cd /verl/docker
   # Build the image (specified tag: ascend-verl:cann82rc1_vllm091) 
   docker build -f Dockerfile.ascend_vllm-0.9.1 -t ascend-verl:cann82rc1_vllm091 .
