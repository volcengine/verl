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

二、镜像差异说明
----------------

ARM 架构与 X86 架构镜像的核心差异如下：

1. **pip 源配置差异**

   x86 架构需额外配置镜像源，ARM 架构无需此步骤：

   .. code:: bash

      pip config set global.extra-index-url "https://download.pytorch.org/whl/cpu/ https://mirrors.huaweicloud.com/ascend/repos/pypi"

2. **LD_LIBRARY_PATH 路径差异**

   -  ARM 架构：LD_LIBRARY_PATH 指向 aarch64 库路径

      .. code:: bash

         export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/8.2.RC1/aarch64-linux/devlib/linux/aarch64:$LD_LIBRARY_PATH

   -  x86 架构：LD_LIBRARY_PATH`指向 x86_64 库路径

      .. code:: bash

         export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/8.2.RC1/x86_64-linux/devlib/linux/x86_64/:$LD_LIBRARY_PATH

3. 必配环境变量

   两种架构均需配置以下环境变量，否则会导致安装vllm_ascend报错：

   .. code:: bash

      # Configuring the Dynamic Link Library Path
      export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/8.2.RC1/[Architecture Directory]/devlib/linux/[Architecture Directory]:$LD_LIBRARY_PATH
      # Load Ascend toolchain environment variables 
      source /usr/local/Ascend/ascend-toolkit/set_env.sh

三、ARM 架构镜像 与 X86 架构镜像 Dockerfile 构建
---------------------------

.. code:: dockerfile

   # 1. Base Image
   FROM swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.2.rc1-910b-ubuntu22.04-py3.11

   # 2. Set up a network proxy (add as needed)
   # ENV ip_addr=xx.xxx.x.xxx
   # ENV http_proxy="http://p_atlas:proxy%40123@$ip_addr:8080"
   # ENV https_proxy=$http_proxy
   # ENV no_proxy=127.0.0.1,localhost,local,.local
   # ENV GIT_SSL_NO_VERIFY=1

   # 3. Install system dependencies
   RUN apt-get update -y && apt-get install -y --no-install-recommends \
       gcc g++ cmake libnuma-dev wget git curl jq vim build-essential \
       && rm -rf /var/lib/apt/lists/*

   # 4. Pre-installation foundation vllm with architecture echo
   RUN echo "===== Current system architecture check =====" && \
       arch=$(uname -m) && \
       echo "Detected architecture: $arch" && \
       echo "============================================" && \
       git clone --depth 1 --branch v0.9.1 https://github.com/vllm-project/vllm \
       && if [ "$arch" = "x86_64" ]; then \
            echo "===== Entering x86_64 branch: Setting pip extra index url ====="; \
            pip config set global.extra-index-url "https://download.pytorch.org/whl/cpu/ https://mirrors.huaweicloud.com/ascend/repos/pypi"; \
          else \
            echo "===== Entering non-x86_64 branch: No extra pip index url set ====="; \
          fi \
       && cd vllm \
       && VLLM_TARGET_DEVICE=empty pip install -v -e . \
       && cd ..

   # 5. Install vllm_ascend
   RUN git clone --depth 1 --branch v0.9.1 https://github.com/vllm-project/vllm-ascend.git \
       && cd vllm-ascend \
       && arch=$(uname -m) && \
       echo "===== Configuring LD_LIBRARY_PATH for $arch =====" && \
       if [ "$arch" = "aarch64" ]; then \
            export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/8.2.RC1/aarch64-linux/devlib/linux/aarch64:$LD_LIBRARY_PATH; \
          elif [ "$arch" = "x86_64" ]; then \
            export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/8.2.RC1/x86_64-linux/devlib/linux/x86_64/:$LD_LIBRARY_PATH; \
          fi \
       && source /usr/local/Ascend/ascend-toolkit/set_env.sh \
       && source /usr/local/Ascend/nnal/atb/set_env.sh \
       && pip install -v -e . \
       && cd ..

   # 6. Install verl
   RUN git clone https://github.com/volcengine/verl.git \
       && cd verl \
       && pip install -r requirements-npu.txt \
       && pip install -e . \
       && cd ..

   # 7. Install MindSpeed
   RUN git clone https://gitee.com/ascend/MindSpeed.git \
       && pip install -e MindSpeed

   # 8. Install Megatron-LM and configure PYTHONPATH
   RUN git clone https://github.com/NVIDIA/Megatron-LM.git \
       && cd Megatron-LM \
       && git checkout core_v0.12.1 \
       && cd .. \
       && echo "export PYTHONPATH=\$PYTHONPATH:/Megatron-LM" >> ~/.bashrc

   # Clear pip cache to reduce image size
   RUN pip cache purge

   # Setting Default Commands
   CMD ["/bin/bash"]

四、镜像构建命令示例
--------------------

1. ARM 与 X86 架构镜像构建
~~~~~~~~~~~~~~~~~~~

.. code:: bash

   # Navigate to the directory containing the Dockerfile 
   cd /path/to/arm-dockerfile
   # Build the image (specified tag: ascend-verl:[x86_64/aarch64]_cann82rc1_vllm091) 
   docker build -f [created Dockerfile] -t ascend-verl:[x86_64/aarch64]_cann82rc1_vllm091 .
