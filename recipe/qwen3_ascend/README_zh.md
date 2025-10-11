# Qwen3-235B-A22B RL训练优化实践样例

## 概述
本样例针对Qwen3-235B-A22B模型，基于[veRL开源框架](https://github.com/volcengine/verl)，使用veRL原生支持的MindSpeed和vllm-ascend框架，完成RL训练全流程的优化适配。

# 环境准备

##  镜像创建

使用vLLM-Ascend提供的镜像，可以快速配置环境：
```shell
镜像下载命令：docker pull quay.io/ascend/vllm-ascend:v0.10.1rc1-a3
```
镜像使用：
```shell
# 执行以下脚本创建容器，请传入容器名称，如your_docker_name
bash run_container.sh your_docker_name
```

##  软件包安装

1、安装依赖的python库。
```
pip3 install -r requirements.txt
```

2、准备源码，本样准备源码的步骤如下：
```shell
# veRL (commit-id:ac2f7)
git clone https://github.com/volcengine/verl.git
git fetch origin pull/3427/head && git cherry-pick FETCH_HEAD
cd verl
cd ..

# vLLM
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout v0.10.1
cp -r vllm ../verl
cd ..

# vLLM-Ascend
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git checkout af62af
git fetch origin pull/2869/head && git cherry-pick FETCH_HEAD
git fetch origin pull/3005/head && git cherry-pick FETCH_HEAD
cp -r vllm_ascend ../verl
cd ..

# MindSpeed
git clone https://gitee.com/ascend/MindSpeed.git
cd MindSpeed
git checkout 7ff81
cp -r mindspeed ../verl
cd ..

# Megatron-LM.core and others
pip install git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.12.1
pip install mathruler
```

## 准备训练数据集与模型
数据集放入 ./data, 数据集准备参考: [veRL官方文档](https://verl.readthedocs.io/en/latest/preparation/prepare_data.html)
模型放入 ./Qwen3-235B-A22B 模型下载地址：[Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B)

## 执行RL后训练
```shell
# 本sample目录下启动Qwen3-235B-A22B的RL后训练
bash ./ray_start_grpo_npu.sh # 基于真实权重的训练脚本 
```

## 性能数据
基于Atlas 900 A3 SuperPoD超节点64卡集群，加载真实权重，Prefill/Decode阶段长度分别为1K与3K，系统吞吐达到89tps/卡。
| 模型                  | 机器型号     | GBS | n_samples | max_prompt_length | max_tokens | 端到端 tps | 
|---------------------|----------|-----|-----------|-------------------|------------|---------| 
| Qwen3-235B-A22B    | Atlas 900 A3 SuperPoD | 256 | 16        | 4096              | 3072       | 89     |