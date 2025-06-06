# Training DeepSeek 671b

verl integrates Megatron to support large MoE models such as `Qwen3-235B-A22B` and `deepseek-ai/DeepSeek-V3`. This is an ongoing community effort.

In the journey the community added the following features and optimizations that enable verl with larger models:
- per tensor weight resharding between rollout and training
- context parallelism and expert parallelism enabled via megatron
- dynamic batch size (sequence balance) for megatron
- reduced ray-related serialization overhead
- optimizer offloading
- various debugging metrics and utils

and the megatron backend now has a wider list of models supported:
- DeepSeek-V3
- Moonlight
- Qwen3
- Qwen2.5-VL (to be merged soon)
- Qwen2

# Getting Started

## DeepSeek 671b


For checkpoint loading, we rely on megatron dist-ckpt for resharding. 

## Qwen3 236b

For Qwen3-236b, please refer to [examples/grpo_trainer/run_qwen3-236b_megatron.sh](https://github.com/volcengine/verl/blob/main/examples/grpo_trainer/run_qwen3-236b_megatron.sh), which runs on 128 H20(96GB) GPUs.

Acknowledgement: @vermouth1992 @ISEEKYAN @ETOgaosion @yzlnew @ShareLer @BearBiscuit05 @ccclyu @ann-qin-lu @SwordFaith @zzong2006 @zhaochenyang20 @ocss884 @eric-haibin-lin
