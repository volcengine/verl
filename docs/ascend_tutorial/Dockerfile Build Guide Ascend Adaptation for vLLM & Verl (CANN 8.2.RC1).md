## 一、组件版本信息

| 组件        | 版本         |
| ----------- | ------------ |
| CANN        | 8.2.RC1      |
| vLLM        | 0.9.1        |
| vLLM-ascend | 0.9.1        |
| Megatron-LM | core_v0.12.1 |
| Python      | 3.11         |
| 基础镜像    | Ubuntu 22.04 |



## 二、镜像差异说明

ARM 架构与 x86 架构镜像的核心差异如下：

1. **pip 源配置差异**

   x86 架构需额外配置镜像源，ARM 架构无需此步骤：

   ```bash
   pip config set global.extra-index-url "https://download.pytorch.org/whl/cpu/ https://mirrors.huaweicloud.com/ascend/repos/pypi"
   ```

2. **LD_LIBRARY_PATH 路径差异**

   - ARM 架构：LD_LIBRARY_PATH 指向 aarch64 库路径

     ```bash
     export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/8.2.RC1/aarch64-linux/devlib/linux/aarch64:$LD_LIBRARY_PATH
     ```

   - x86 架构：LD_LIBRARY_PATH`指向 x86_64 库路径

     ```bash
     export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/8.2.RC1/x86_64-linux/devlib/linux/x86_64/:$LD_LIBRARY_PATH
     ```

3. 必配环境变量

   两种架构均需配置以下环境变量，否则会导致安装vllm_ascend报错：

   ```bash
   # 配置动态链接库路径
   export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/8.2.RC1/[架构目录]/devlib/linux/[架构目录]:$LD_LIBRARY_PATH
   # 加载昇腾工具链环境变量
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ```

   

## 三、ARM 架构镜像 Dockerfile

```dockerfile
# 1. 基础镜像
FROM swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.2.rc1-910b-ubuntu22.04-py3.11

# 2. 设置网络代理（按需添加）
# ENV ip_addr=xx.xxx.x.xxx
# ENV http_proxy="http://p_atlas:proxy%40123@$ip_addr:8080"
# ENV https_proxy=$http_proxy
# ENV no_proxy=127.0.0.1,localhost,local,.local
# ENV GIT_SSL_NO_VERIFY=1

# 3. 安装系统依赖
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    gcc g++ cmake libnuma-dev wget git curl jq vim build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. 前置安装基础vllm
RUN git clone --branch v0.9.1 https://github.com/vllm-project/vllm \
    && cd vllm \
    && VLLM_TARGET_DEVICE=empty pip install -v -e . \
    && cd ..

# 5. 安装vllm_ascend
RUN git clone --depth 1 --branch v0.9.1 https://github.com/vllm-project/vllm-ascend.git \
    && cd vllm-ascend \
    && export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/8.2.RC1/aarch64-linux/devlib/linux/aarch64:$LD_LIBRARY_PATH \
    && source /usr/local/Ascend/ascend-toolkit/set_env.sh \
    && source /usr/local/Ascend/nnal/atb/set_env.sh \
    && pip install -v -e . \
    && cd ..

# 6. 安装verl
RUN git clone https://github.com/volcengine/verl.git \
    && cd verl \
    && pip install -r requirements-npu.txt \
    && pip install -e . \
    && cd ..

# 7. 安装MindSpeed
RUN git clone https://gitee.com/ascend/MindSpeed.git \
    && pip install -e MindSpeed

# 8. 安装Megatron-LM并配置PYTHONPATH
RUN git clone https://github.com/NVIDIA/Megatron-LM.git \
    && cd Megatron-LM \
    && git checkout core_v0.12.1 \
    && cd .. \
    # 配置Megatron-LM路径到环境变量
    && echo "export PYTHONPATH=\$PYTHONPATH:/Megatron-LM" >> ~/.bashrc

# 清理pip缓存，减小镜像体积
RUN pip cache purge

# 设置默认命令
CMD ["/bin/bash"]
```



## 四、x86 架构镜像 Dockerfile

```dockerfile
# 1. 基础镜像
FROM swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.2.rc1-910b-ubuntu22.04-py3.11

# 2. 设置网络代理（按需添加）
# ENV ip_addr=xx.xxx.x.xxx
# ENV http_proxy="http://p_atlas:proxy%40123@$ip_addr:8080"
# ENV https_proxy=$http_proxy
# ENV no_proxy=127.0.0.1,localhost,local,.local
# ENV GIT_SSL_NO_VERIFY=1

# 3. 安装系统依赖
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    gcc g++ cmake libnuma-dev wget git curl jq vim build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. 前置安装基础vllm（x86需额外配置pip源）
RUN git clone --depth 1 --branch v0.9.1 https://github.com/vllm-project/vllm \
    && pip config set global.extra-index-url "https://download.pytorch.org/whl/cpu/ https://mirrors.huaweicloud.com/ascend/repos/pypi" \
    && cd vllm \
    && VLLM_TARGET_DEVICE=empty pip install -v -e . \
    && cd ..

# 5. 安装vllm_ascend
RUN git clone --depth 1 --branch v0.9.1 https://github.com/vllm-project/vllm-ascend.git \
    && cd vllm-ascend \
    && export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/8.2.RC1/x86_64-linux/devlib/linux/x86_64/:$LD_LIBRARY_PATH \
    && source /usr/local/Ascend/ascend-toolkit/set_env.sh \
    && source /usr/local/Ascend/nnal/atb/set_env.sh \
    && pip install -v -e . \
    && cd ..

# 6. 安装verl
RUN git clone https://github.com/volcengine/verl.git \
    && cd verl \
    && pip install -r requirements-npu.txt \
    && pip install -e . \
    && cd ..

# 7. 安装MindSpeed
RUN git clone https://gitee.com/ascend/MindSpeed.git \
    && pip install -e MindSpeed

# 8. 安装Megatron-LM并配置PYTHONPATH
RUN git clone https://github.com/NVIDIA/Megatron-LM.git \
    && cd Megatron-LM \
    && git checkout core_v0.12.1 \
    && cd .. \
    && echo "export PYTHONPATH=\$PYTHONPATH:/Megatron-LM" >> ~/.bashrc

# 清理pip缓存，减小镜像体积
RUN pip cache purge

# 设置默认命令
CMD ["/bin/bash"]
```



## 五、镜像构建命令示例

### 1. ARM 架构镜像构建

```bash
# 进入Dockerfile所在目录
cd /path/to/arm-dockerfile
# 构建镜像（指定标签：ascend-verl:arm_cann82rc1_vllm091）
docker build -f [创建的Dockerfile文件] -t ascend-verl:arm_cann82rc1_vllm091 .
```



### 2. x86 架构镜像构建

```bash
# 进入Dockerfile所在目录
cd /path/to/x86-dockerfile
# 构建镜像（指定标签：ascend-verl:x86_cann82rc1_vllm091）
docker build -f [创建的Dockerfile文件] -t ascend-verl:x86_cann82rc1_vllm091 .
```