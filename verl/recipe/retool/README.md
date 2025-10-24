# Retool
[ReTool: Reinforcement Learning for Strategic Tool Use in LLMs](https://arxiv.org/abs/2504.11536)

## Overview
- Base model: [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct)
- SFT dataset: [JoeYing/ReTool-SFT](https://huggingface.co/datasets/JoeYing/ReTool-SFT)
- RL dataset: [BytedTsinghua-SIA/DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k)
- Val dataset: [yentinglin/aime_2025](https://huggingface.co/datasets/yentinglin/aime_2025)

## How it works

 Retool's workflow is divided into two key phases:

1.  Cold Start and Supervised Fine Tuning (SFT)
    
     The data generation pipeline builds a high-quality dataset containing code-enhanced inference trajectories, and supervised fine-tuning enables the model to master basic Tool call (e.g., code execution) and analysis of the execution results.
    
2.  Dynamic Interaction and Policy Optimization (RL).
    
     With the verl Reinforcement Learning framework, the model dynamically inserts code blocks during inference and interacts with the sandbox environment in real-time, generating a hybrid trajectory of natural language thinking and code snippets, sending the code to the sandbox for asynchronous execution when code termination markers are detected, and the execution results (success outputs/errors) are fed back to the model for guiding the subsequent inference. This "think-execute-feedback" cycle, together with the design of rewards based on the accuracy of the final answer, enables the model to independently optimize the Tool call strategy, and improves the reasoning efficiency and computational accuracy.

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
