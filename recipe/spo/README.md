# Single-stream Policy Optimization (SPO)

[![arXiv](https://img.shields.io/badge/arXiv-2509.13232-b31b1b.svg)](https://arxiv.org/abs/2509.13232)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A novel reinforcement learning algorithm for Large Language Models that eliminates the fundamental inefficiencies of group-based methods like GRPO while achieving superior performance on challenging reasoning tasks.

## 🎯 Key Highlights

- **+3.4 pp improvement** on average maj@32 across 5 math benchmarks compared to GRPO
- **Superior scalability** for tool-integrated and multi-turn reasoning tasks

## 📖 Overview

Single-stream Policy Optimization (SPO) addresses critical inefficiencies in group-based RL methods for LLMs:

### Problems with Group-based Methods (GRPO)
- **Degenerate Groups**: Sample groups yield zero learning signal when all responses have identical rewards
- **Synchronization Bottlenecks**: Entire groups must wait for the slowest member, severely limiting scalability
- **High Variance**: Small-group baselines create noisy, unstable learning signals

### SPO Solution
SPO returns to the classic single-stream paradigm with three key innovations:

1. **KL-Adaptive Value Tracker**: Persistent Bayesian tracker that maintains success probability estimates across policy updates
2. **Global Advantage Normalization**: Stable normalization across entire batches instead of noisy per-group statistics
3. **Prioritized Sampling**: Adaptive curriculum focusing on high-uncertainty prompts

## 🏆 Performance Results

**Math Competition Benchmarks (Qwen3-8B)**
| Method | AIME 24 | AIME 25 | BeyondAIME | BRUMO 25 | HMMT 25 | Average |
|--------|---------|---------|------------|----------|---------|---------|
| GRPO   | 83.3    | 72.1    | 45.6       | 56.7     | 44.2    | 60.4    |
| **SPO** | **84.0** | **76.5** | **46.9** | **64.0** | **47.5** | **63.8** |
| **Gain** | **+0.7** | **+4.4** | **+1.3** | **+7.3** | **+3.3** | **+3.4** |

## 🎯 Quick Start

### Prerequisites
**Required verl version:**

This project requires a specific verl commit for SPO implementation. Please use:
```bash
git clone https://github.com/volcengine/verl.git
cd verl
git checkout 04726dbf12da5352aafbd550bff5093b89ead8c1
pip install -e .
```

### Basic Training

**Generate offline values file:**

If you want to generate offline values file, you can just change the val_files to the training files, and set the following params:
```bash
trainer.val_before_train=True \
trainer.val_only=True \
trainer.validation_data_dir=$validation_data_dir
```

**Training commands:**
```bash
# Enable SPO training mode
export SPO_ENABLE=True
export SPO_OFFLINE_VALUES="dingzihan737/SPO_Qwen3-8B_DAPO_16k_ReTool_Binary"
export EXP_NAME="spo_experiment"

# Run SPO training
bash run_spo.sh
```

## 📚 Citation

If you use SPO in your research, please cite:

```bibtex
@article{xu2025single,
	title={Single-stream Policy Optimization},
	author={Xu, Zhongwen and Ding, Zihan},
	year={2025},
	journal={arXiv preprint arXiv:2509.13232},
}
```
