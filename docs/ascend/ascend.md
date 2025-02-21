# veRL x Ascend

我们在 verRL 上增加对华为昇腾设备的支持，在华为昇腾设备上使用 veRL 与在英伟达 GPU 上使用几乎相同。

## 硬件支持

* Atlas 800T A2

## 安装

### 环境准备

| 软件      | 版本        |
| --------- | ----------- |
| Python    | >= 3.10     |
| torch     | == 2.5.1    |
| torch_npu | >= 2.5.1rc1 |
| CANN      | >= 8.1.RC1 (**Not Released**)   |

> 当前版本在 CANN 8.1.RC1 上进行测试，该版本通常预计在三月底正式发布。
> 为了保证能够正常使用 vLLM，我们建议上述配套软件的安装遵循 vllm-ascend 的[安装教程](https://vllm-ascend.readthedocs.io/en/v0.7.1rc1/installation.html)。
### 源码安装

```shell
git clone -b vllm-0.7-npu https://github.com/as12138/verl.git
cd verl
pip install -r requirements-npu.txt
pip install -e .
```

### vLLM

为了保证能够在 veRL 上正常使用 vLLM，需要安装 vLLM Ascend 插件（`vllm-ascend`）。关于在华为昇腾上支持的 vLLM 版本以及和 vLLM Ascend 的配套关系请参考[安装教程](https://vllm-ascend.readthedocs.io/en/v0.7.1rc1/installation.html)。

> hybrid engine 功能依赖 vLLM >= 0.7.2，当前 vllm-ascend 不支持该版本，该功能尚未支持。
### Ray

训练脚本提交方式与英伟达 GPU 相同。

### 其他第三方库说明

| 软件           | 说明 |
|:--------------|:--|
| flash_attn   | 不支持 |
| liger-kernel | 不支持 |

## 支持的算法

### 精度对比

根据经验，我们期望在相同配置下，在华为昇腾设备上的 Loss 与英伟达 GPU 的 Loss 平均误差小于 2%，具体计算方式如下：

![loss_comparison](./images/loss_comparison.png)

其中，N 表示训练的步数。更多信息请参考[精度计算说明](https://www.hiascend.com/document/detail/zh/Pytorch/600/ptmoddevg/trainingmigrguide/LMaccuracy_0001.html)。

### 进展

| 算法   | 进展                                                        |
|------|---------------------------------------------------------------|
| SFT  | 已支持                                                           |
| PPO  | 已支持                                                        |
| GRPO | 已支持                                                        |


> 补充说明：
>
> 1. 由于适配问题，vllm 0.7.1 版本会出现`Invalid device`错误，按照 https://github.com/volcengine/verl/blob/main/docs/README_vllm0.7.md 修改即可。
> 2. `decord`库在ASCEND NPU上需要编译安装，如果暂时使用不到这个库，可以修改vllm/multimodel/video.py
`import decord` 为 `decord = PlaceholderModule("decord")`