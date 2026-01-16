# DeepSeek-R1-Zero on Ascend NPU
本recipe是基于Deepseek-V3-Base模型在NPU上进行RLHF后训练的样例，基于GRPO与规则奖励，使用deepscaler数据集。

## 实现细节
为了在Ascend NPU上实现DeepSeek模型的RL训练，本样例中补充了一些代码，如下所示：
- 我们参考`verl/utils/reward_score/gsm8k.py`，在`deepscaler.py`中实现了一个简单的规则奖励函数。
- 我们提供了数据集文件转换脚本`json_to_parquet.py`，在数据文件格式转换的同时给prompt增加了激发模型思考的模板。
- NPU上vLLM的sleep可能存在内存卸载不干净的问题，因此添加了一些patch，手动实现NPU上Rollout模型与KVcache的卸载与加载。相关代码在`vllm_rollout_spmd.py`以及 `megatron_workers.py`中。
- 为了实现vLLM利用所有卡进行专家并行，需要支持vLLM的数据并行。为此添加了一些patch构建正确的DP通信域。相关代码在`vllm_parallel_state.py`以及`vllm_rollout_spmd.py`中。此外还需要正确配置`VLLM_DP_SIZE`环境变量为`world_size / vllm_tp_size`。
- NPU的MindSpeed训练框架会将torch.compile无效化来规避训练侧的compile失败，但这会使推理侧无法利用torch.compile加速。为了解决该问题，本样例添加了一些patch，使推理时可以compile，训练时不compile。相关代码`megatron_workers.py`中。
- RL训练过程中，NPU上vLLM多次KVcache调度可能引发申请内存不一致导致内存踩踏问题，修复patch在`engine_core.py`中。

通过全局搜索`# NPU-ADAPTATION`，可以看到patch代码所做的实际改动。

更多技术细节可参考[技术报告](https://gitcode.com/cann/cann-recipes-train/blob/master/docs/deepseek/deepseek_rl_train_optimization.md)。

## 训练细节
### 训练超参

本样例基于DeepSeek-671B Base模型在deepscaler数据集上训练，使用简单的格式奖励和结果准确率奖励，训练超参如下：

|  迭代  | 学习率 |  gbs  |  采样数 | 温度 |  kl-coef | 输入长度 | 输出长度 | 规则奖励 | 奖励模型 |
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| 70 | 1e-6 (constant) |  512  |  16  |  1.0  |  0.001  |  1024  |  2048  |  format + acc  | - |

### 训练资源与性能
本样例在昇腾Atlas 800T A3超节点服务器上进行训练，使用了128张A3 NPU，等效于256张加速卡。具体的部署方式如下：

| Rollout部署 | Actor部署 | Reference部署 | Offload策略 |
|:----:|:----:|:----:|:----:|
|  TP2 EP256  |  EP32 PP8  |  同Actor  |  全offload，优化器使用[Mindspeed卸载特性](https://gitee.com/ascend/MindSpeed/blob/master/docs/features/swap-optimizer.md)  |

得到一步的训练性能如下（吞吐会随着训练中模型输出长度变化而改变）：
|  step  | 平均问题长度 |  平均回复长度  |  单步总耗时(s) | 吞吐(tps/A3) | gen耗时(s) | reward耗时(s) | old_prob耗时(s) | ref_prob耗时(s) | update耗时(s) |
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| 2 | 175.1 |  1385.0  |  1044.8  | 95.5 |  482.2  |  20.4  |  105.5  |  92.7  | 342.9 |

### 训练过程记录
<div align="center">
  <img src="./figures/rewards.png" width="33%" />
  <img src="./figures/response_len.png" width="33%" />
  <img src="./figures/val_score.png" width="33%" />
</div>

## 快速开始

### 环境准备
verl上的NPU环境准备，可参考[ascend_quick_start.rst](../../docs/ascend_tutorial/ascend_quick_start.rst)进行配置。

此外，也可使用我们提供的Dockerfile在本地构建项目运行环境：`docker build -f Dockerfile.vllm_ascend.mindspeed.deepseekV3 -t REPOSITORY:TAG ./`

本样准备源码的步骤如下：
```bash
# verl
git clone https://github.com/volcengine/verl.git

# vLLM (v0.9.1)
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout v0.9.1
cp -r vllm ../verl
cd ..

# vLLM-Ascend (v0.9.1-dev)
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git checkout 8c7bc45
cp -r vllm_ascend ../verl
cd ..

# MindSpeed (commit-id: f6688)
git clone https://gitcode.com/Ascend/MindSpeed.git
cd MindSpeed
git checkout f6688
cp -r mindspeed ../verl
cd ..

# Megatron-LM.core and others
pip install git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.12.1
pip install mathruler
```

### 准备训练数据集
本样例使用deepscaler数据集。准备方式如下：
- 下载数据集[json文件](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset/blob/main/deepscaler.json)。
- 获取`train.parquet`与`test.parquet`文件并放入`./data/deepscaler`路径：

    ```bash
    # 在verl项目目录执行
    python recipe/r1_ascend/json_to_parquet.py --output_dir ./data/deepscaler --json_path path/to/deepscaler.json --train_data_ratio 0.9
    ```
    
    训练中经过处理的prompt将包含模板，例如：`A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. Put your final answer within \boxed{}. <｜User｜>{problem}<｜Assistant｜>`

### 准备模型权重
DeepSeek-V3-Base模型权重准备步骤如下：
- 需要将模型配置相关文件（不含权重）放入`./DeepSeek-V3-hf`目录，并且`config.json`需要进行替换以去除量化和MTP。该步骤可参考[此链接](https://gitcode.com/cann/cann-recipes-train/blob/master/rl_train/deepseek/README.md#%E6%A8%A1%E5%9E%8B%E6%9D%83%E9%87%8D%E5%87%86%E5%A4%87)。
- 模型FP8权重下载：[HuggingFace地址](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base)，[ModelScope地址](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-V3-Base)。此步骤需要目录所在磁盘有650GB以上空间。
- 将FP8权重转为BF16权重，可参考[此链接](https://gitcode.com/cann/cann-recipes-train/blob/master/rl_train/deepseek/README.md#%E6%A8%A1%E5%9E%8B%E6%9D%83%E9%87%8D%E5%87%86%E5%A4%87)。此步骤需要目录所在磁盘有1300GB以上空间。

本样例使用了预先切分的分布式权重，因此还要执行以下的切分权重操作：
- 分布式权重需存储至`ckpts/DeepseekV3-dist-ckpts`。
- 使用`verl/scripts/converter_hf_to_mcore.py`对原始的BF16权重切分得到分布式权重。实践中我们发现2T的CPU内存不足以完成671B模型的权重切分处理，为此我们对该脚本进行了专家并行的适配，并在64块NPU上用EP8 PP8分布式策略对权重进行了切分。

### 其他代码修改
实践中为了得到以上on-policy训练的结果，我们将 `verl/workers/actor/megatron_actor.py` 中的代码段 `old_log_prob = data["old_log_probs"]` 替换为如下代码：
```python
on_policy = self.config.ppo_epochs == 1
if on_policy:
    old_log_prob = log_prob.detach()    # 确保二者数值完全相等
else:
    old_log_prob = data["old_log_probs"]
```

### 执行RL后训练
```bash
# verl目录下启动DeepSeekV3的RL后训练
bash ./recipe/r1_ascend/ray_start_grpo_npu.sh
```