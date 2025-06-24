# Training DeepSeek 671b

verl integrates Megatron to support large MoE models such as `Qwen3-235B-A22B` and `deepseek-ai/DeepSeek-V3`. This is an ongoing community effort.

In the journey the community added the following features and optimizations that enable verl with larger models:
- per tensor weight resharding between rollout and training
- context parallelism and expert parallelism enabled via megatron
- dynamic batch size (sequence balance) for megatron
- reduced ray-related serialization overhead
- optimizer offloading, recomputation, and efficient kernels
- various debugging metrics and utils

and the megatron backend now has a wider list of models supported:
- DeepSeek-V3
- Moonlight
- Qwen3
- Qwen2.5-VL (to be merged soon)
- Qwen2
- Mixtral

## Getting Started

### DeepSeek 671b

The recommended image with pre-built megatron dependency is `whatcanyousee/verl:ngc-cu124-vllm0.8.5-sglang0.4.6.post5-mcore0.12.1-te2.3-deepseekv3`, built with the Dockerfile in [docker/Dockerfile.vllm.sglang.megatron.deepseek](https://github.com/volcengine/verl/blob/main/docker/Dockerfile.vllm.sglang.megatron.deepseek).

For checkpoint loading, we rely on megatron dist-ckpt for resharding. A converted dist-ckpt for DeepSeek-V3 is available from [huggingface BearBiscuit05/dpsk-v3-671B-BF16-dist_ckpt](https://huggingface.co/BearBiscuit05/dpsk-v3-671B-BF16-dist_ckpt/tree/main).

To run end-to-end training on the DAPO dataset, run [recipe/dapo/test_dapo_dspk_671b_megatron.sh](https://github.com/volcengine/verl/blob/main/recipe/dapo/test_dapo_dspk_671b_megatron.sh). It runs on 512 H20(96GB) GPUs with the following setup:
- vllm rollout with TP=32, bfloat16
- megatron training with attention DP, MoE EP=32, PP=16, bfloat16

MTP is disabled during RL training.

### Qwen3 236b

For Qwen3-236b, please refer to [examples/grpo_trainer/run_qwen3-236b_megatron.sh](https://github.com/volcengine/verl/blob/main/examples/grpo_trainer/run_qwen3-236b_megatron.sh), which runs on 128 H20(96GB) GPUs.

## Upcoming Optimizations

The community continue to optimize large MoE models further, ongoing efforts include:
- further optimizing memory consumption, and provide recommended/tuned configurations with various machine types
- optimizing long context RL training performance
- performance improvement with SGLang x Megatron

We invite the community to try and improve verl together. Get connected with us on [slack](https://join.slack.com/t/verlgroup/shared_invite/zt-2w5p9o4c3-yy0x2Q56s_VlGLsJ93A6vA)/[wechat](https://raw.githubusercontent.com/eric-haibin-lin/verl-community/refs/heads/main/WeChat.JPG)/[Github issues](https://github.com/volcengine/verl/issues/708)!

## Acknowledgement
@vermouth1992 @ISEEKYAN @ETOgaosion @yzlnew @ShareLer @BearBiscuit05 @ccclyu @ann-qin-lu @SwordFaith @zzong2006 @zhaochenyang20 @ocss884 @eric-haibin-lin
