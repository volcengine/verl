# DeepSeek-R1-Zero on Ascend NPU
本recipe是基于Deepseek-V3-Base模型在NPU上进行RLHF后训练的样例，基于GRPO与规则奖励，使用deepscaler数据集。

## 实现细节
为了在Ascend NPU上实现DeepSeek模型的RL训练，本样例中补充了一些代码，如下所示：
- 为了能够在更有挑战性的deepscaler数据集上进行训练，我们参考`verl/utils/reward_score/gsm8k.py`实现了`deepscaler.py`
- NPU上vLLM的sleep可能存在内存卸载不干净的问题，因此添加了一些patch，手动实现NPU上Rollout模型的卸载与加载。相关代码在`megatron_vllm.py`, `megatron_workers.py`
- 为了实现vLLM利用所有卡进行专家并行，需要支持vLLM的数据并行。为此添加了一些patch构建正确的DP通信域。相关代码在`vllm_parallel_state.py`, `vllm_rollout_spmd.py`, 此外还需要正确配置`VLLM_DP_SIZE`环境变量
- NPU的MindSpeed训练框架会将torch.compile无效化来规避训练侧的compile失败，但这会使推理侧无法利用torch.compile加速。为了解决该问题，本样例添加了一些patch，使推理时可以compile，训练时不compile。相关代码`megatron_workers.py`, `patch_compile.py`
- RL训练过程中，NPU上vLLM多次KVcache调度可能引发申请内存不一致导致内存踩踏问题，修复patch在`engine_core.py`


## 训练细节
### 训练超参

本样例基于DeepSeek-671B Base模型在deepscaler数据集上训练，使用简单的结果准确率奖励，训练超参如下：

|  迭代  | 学习率 |  gbs  |  采样数 | 温度 |  kl-coef | 输入长度 | 输出长度 | 规则奖励 | 奖励模型 |
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| 100 | 1e-6 (constant) |  512  |  16  |  0.9  |  0.001  |  1024  |  3072  |  base_acc  | - |

### 训练资源与性能
本样例在昇腾Atlas 800T A3超节点服务器上进行训练，使用了128张A3 NPU，等效于256张加速卡。具体的部署方式如下：

| Rollout部署 | Actor部署 | Reference部署 | Offload策略 |
|:----:|:----:|:----:|:----:|
|  TP2 EP256  |  EP32 PP8  |  同Actor  |  全offload，优化器使用[Mindspeed卸载特性](https://gitee.com/ascend/MindSpeed/blob/master/docs/features/swap-optimizer.md)  |

得到一步的训练性能如下：
|  Step  | 平均问题长度 |  平均回复长度  |  单步总耗时(s) | 吞吐(tps/A3) | generate_seq耗时(s) |  reshard耗时(s) | reward耗时(s) | old耗时(s) | ref耗时(s) | update耗时(s) |
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| 27 | 75.8 |  2553.0  |  1338.3  | 125.7 |  495.5  |  67.6  |  8.5  |  134.1  |  116.7  | 495.2 |


### 训练过程记录

**Reward曲线：** TODO


**Response length mean曲线：** TODO


**问答样例：** TODO


## 快速开始

### 环境准备
veRL上的NPU环境准备，可参考[ascend_quick_start.rst](../../docs/ascend_tutorial/ascend_quick_start.rst)进行配置。

此外，也可使用我们提供的Dockerfile在本地构建项目运行环境：`docker build -f Dockerfile.vllm_ascend.mindspeed.deepseekV3 -t REPOSITORY:TAG ./`

本样准备源码的步骤如下：
```shell
# veRL (commit-id:55e3c)
git clone https://github.com/volcengine/verl.git
cd verl
cd ..

# vLLM (v0.9.1)
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout v0.9.1
cp -r vllm ../verl
cd ..

# vLLM-Ascend (v0.9.1rc2)
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git checkout v0.9.1rc2
cp -r vllm-ascend ../verl
cd ..

# MindSpeed (commit-id: 7ff81)
git clone https://gitee.com/ascend/MindSpeed.git
cd MindSpeed
git checkout 55e3c
cd ..

# Megatron-LM.core and others
pip install git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.12.1
pip install mathruler
```

### 准备训练数据集与模型权重
- 数据集放入`./data`, 数据集准备参考: [veRL官方文档](https://verl.readthedocs.io/en/latest/preparation/prepare_data.html)。本样例使用了[deepscaler数据集](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset/blob/main/deepscaler.json)。

- 模型放入`./DeepSeek-V3-hf`, 模型下载地址：[DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3)。

- 分布式权重切分使用`verl/scripts/converter_hf_to_mcore.py`完成。实践中我们发现2T的CPU内存不足以完成671B模型的权重切分处理，为此我们对该脚本进行了专家并行的适配，并在64块NPU上用EP8 PP8分布式策略对权重进行了切分。

### 执行RL后训练
```shell
# verl目录下启动DeepSeekV3的RL后训练
bash ./recipe/r1_ascend/ray_start_grpo_npu.sh
```
