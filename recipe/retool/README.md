# Retool
[ReTool: Reinforcement Learning for Strategic Tool Use in LLMs](https://arxiv.org/abs/2504.11536)

## Overview
- Base model: [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct)
- SFT dataset: [JoeYing/ReTool-SFT](https://huggingface.co/datasets/JoeYing/ReTool-SFT)
- RL dataset: [BytedTsinghua-SIA/DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k)
- Val dataset: [yentinglin/aime_2025](https://huggingface.co/datasets/yentinglin/aime_2025)

## SFT
1. Data preparation
```bash
python3 recipe/retool/retool_sft_preprocess.py
```

2. Training
```bash
bash recipe/retool/run_qwen2-32b_sft.sh
```

After 6 epoches, validation metrics:
- val-core/aime_2025/acc/mean@30: 0.24
- val-aux/num_turns/mean: 7.2

## RL

### GRPO
```bash
bash recipe/retool/run_qwen2-32b_dapo.sh
```

After 150 steps, validation metrics:
- val-core/aime_2025/acc/mean@30: 0.6
- val-aux/num_turns/mean: 10

### PPO

```bash
bash recipe/retool/run_qwen2-32b_ppo.sh
```

After 250 steps, validation metrics:
- val-core/aime_2025/acc/mean@30: 0.55
- val-aux/num_turns/mean: 8.3
