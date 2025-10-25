<h1 align="center" style="color:#1976D2; font-size:42px; font-weight:bold; margin-bottom:0;">
  FlowRL
</h1>

<p align="center" style="color:#42A5F5; font-size:16px; margin-top:0;">
  Matching Reward Distributions via Flow Balance
</p>
<p align="center" style="color:#42A5F5; font-size:15px; margin-top:4px;">
  <a href="https://arxiv.org/abs/2509.15207" target="_blank">📄 arXiv Paper</a> |
  <a href="https://huggingface.co/papers/2509.15207" target="_blank">🤗 #1 Paper of the Day</a>
</p>
<p align="center" style="color:#42A5F5; font-size:14px; margin-top:4px;">
  <a href="https://x.com/RoverHM/status/1969113890878259518" target="_blank">𝕏 Post 1</a> |
  <a href="https://x.com/zdhnarsil/status/1969049940774023428" target="_blank">𝕏 Post 2</a> |
  <a href="https://x.com/_akhaliq/status/1968901977376505929" target="_blank">𝕏 Post 3</a>
</p>

<p align="center">
  <img src="figures/flowrl.png" alt="FlowRL Overview" width="95%"/>
</p>

## FlowRL Objective:

$$
\mathcal{L}_{\text{FlowRL}} = w \cdot \left( \log Z_{\phi}(x) + \frac{1}{|y|} \log \pi_{\theta}(y \mid x) - \beta \hat{r}(x, y) - \frac{1}{|y|} \log \pi_{\text{ref}}(y \mid x) \right)^2
$$

FlowRL is a flow-balanced reinforcement learning method that matches full reward distributions instead of maximizing rewards, promoting diverse exploration and generalizable reasoning trajectories in LLMs.

## 🚀 Quick Start

There are two ways to run FlowRL:

### Option 1: Original Paper Reproduction

For exact reproduction of results from the paper, use the original repository:

👉 **Original Code:** [https://github.com/Xuekai-Zhu/FlowRL](https://github.com/Xuekai-Zhu/FlowRL)

Follow the instructions in the original repository for paper reproduction.

### Option 2: FlowRL Recipe in verl

For running FlowRL using the verl framework:

#### Step 1: Prepare Data and Model

```bash
# Prepare dataset
bash recipe/flowrl/prepare/prepare_data.sh

# Prepare model
bash recipe/flowrl/prepare/prepare_model.sh
```

#### Step 2: Run Training

```bash
# Train FlowRL with Qwen2.5-7B
bash recipe/flowrl/run_flowrl_qwen2.5_7b.sh
```

## 🔧 Implementation Guide

If you want to implement FlowRL in your own codebase, we provide a simple guideline:

⚙️ [FlowRL Implementation Guide](FLOWRL_SIMPLE_GUIDE.md)

## 📚 Additional Resources

### Installation

Install [verl](https://github.com/volcengine/verl) first before using FlowRL.

### Data Preparation (Alternative Methods)

```bash
# Option A: Download our pre-processed datasets directly
bash preprocess/down_load_dataset.sh
# Move data to default directory
mv data/xuekai/flowrl-data-collection/math_data data/math_data
mv data/xuekai/flowrl-data-collection/code_data data/code_data
```

```bash
# Option B: Process data from original sources
# For detailed processing instructions, see data/README.md
```

### Model Preparation

For Math Tasks: `Qwen/Qwen2.5-7B` (default in script) ; `Qwen/Qwen2.5-32B`

For Code Tasks: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`

```bash
# Download default model (Qwen2.5-7B for math)
bash preprocess/down_load_model.sh

# For other models, modify MODEL_NAME in the script before running
```

### Training Scripts

```bash
cd verl_FlowRL

# For 7B math training
bash command/training/math/flowrl_7B_math.sh

# For 32B math training
bash command/training/math/flowrl_32B_math.sh

# For 7B code training
bash command/training/code/flowrl_7B_code.sh
```

### Testing

```bash
cd verl_Test

# First merge the model
bash command/eval/merge_model.sh

# For math testing
bash command/eval/math/flowrl_math_test.sh

# For code testing
bash command/eval/code/flowrl_code_test.sh
```

## 📝 Citation

If you think this repo helps you, please kindly consider citing our paper:

```bibtex
@article{zhu2025flowrl,
  title={FlowRL: Matching Reward Distributions for LLM Reasoning},
  author={Zhu, Xuekai and Cheng, Daixuan and Zhang, Dinghuai and Li, Hengli and Zhang, Kaiyan and Jiang, Che and Sun, Youbang and Hua, Ermo and Zuo, Yuxin and Lv, Xingtai and others},
  journal={arXiv preprint arXiv:2509.15207},
  year={2025}
}
```
