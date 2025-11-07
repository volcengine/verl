# VERL Jupyter Notebooks

Complete Jupyter notebook interface for training language models with **verl** (Volcano Engine Reinforcement Learning).

## 📚 Notebooks

| Notebook | Description | Use Case |
|----------|-------------|----------|
| **1_verl_complete_training.ipynb** | Main training interface with all RL algorithms | Train models with GRPO, PPO, REINFORCE++, RLOO, ReMax |
| **2_data_preprocessing.ipynb** | Dataset preparation and formatting | Prepare GSM8K, MATH, HH-RLHF, or custom datasets |
| **3_model_evaluation.ipynb** | Model evaluation and benchmarking | Evaluate trained models, compute metrics, generate samples |

## 🚀 Quick Start

### 1. Installation (First Time Only)

Open `1_verl_complete_training.ipynb` and run Section 0:

```python
# Choose one:
!pip install verl[vllm,gpu,math]      # For vLLM backend
!pip install verl[sglang,gpu,math]    # For SGLang backend
!pip install verl[vllm,sglang,gpu,math]  # For both (recommended)
```

**Note**: Installation takes ~5-10 minutes.

### 2. Prepare Data

Open `2_data_preprocessing.ipynb` and run the section for your dataset:
- Section 1: GSM8K (math problems)
- Section 2: MATH (competition math)
- Section 3: HH-RLHF (dialogue)
- Section 4: Custom datasets

### 3. Train Your Model

Open `1_verl_complete_training.ipynb`:

1. **Run Section 1**: Detect your hardware (GPUs, CUDA, bf16 support)
2. **Run Section 1.5**: Choose backend (vLLM or SGLang)
3. **Edit Section 2**: Configure cluster (single GPU / multi-GPU / multi-node)
4. **Edit Section 3**: Set data paths and model
5. **Run ONE algorithm section** (4-8):
   - Section 4: GRPO
   - Section 5: PPO
   - Section 6: REINFORCE++
   - Section 7: RLOO
   - Section 8: ReMax
6. **Section 11**: Upload to HuggingFace

### 4. Evaluate Your Model

Open `3_model_evaluation.ipynb`:
- Load your trained checkpoint
- Generate sample outputs
- Benchmark on test set
- Compute metrics

---

## 🔧 Backend Selection: vLLM vs SGLang

### Quick Comparison

| Feature | vLLM | SGLang |
|---------|------|--------|
| **Maturity** | ⭐⭐⭐⭐⭐ Very stable | ⭐⭐⭐⭐ Newer |
| **Speed** | ⭐⭐⭐⭐ Fast | ⭐⭐⭐⭐⭐ Faster |
| **Multi-turn** | ⭐⭐⭐ Good | ⭐⭐⭐⭐⭐ Excellent |
| **Memory** | PagedAttention | RadixAttention (better caching) |
| **Model Support** | ⭐⭐⭐⭐⭐ Wide | ⭐⭐⭐⭐ Growing |
| **Documentation** | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐⭐ Good |

### When to Use Each

**Use vLLM if**:
- You want maximum stability
- You're working with less common models
- You need extensive documentation
- Your team prefers mature, battle-tested tools

**Use SGLang if**:
- You want maximum speed
- You're doing multi-turn conversations
- You need better KV cache reuse
- You're comfortable with newer tools

### Switching Backends

You can easily switch backends between training runs:

1. Open `1_verl_complete_training.ipynb`
2. In Section 1.5, change `BACKEND = 'vllm'` to `BACKEND = 'sglang'` (or vice versa)
3. Re-run Section 1.5
4. Re-run your algorithm section

**Note**: Checkpoints are backend-agnostic (standard PyTorch), so you can train with one backend and evaluate with another.

---

## 💡 Hardware Auto-Detection

The notebooks automatically detect and configure based on your hardware:

### GPU Detection
- **Type**: A100, H100, V100, RTX 4090, etc.
- **Count**: Single GPU, 8x GPUs, multi-node
- **VRAM**: Adjusts batch sizes based on available memory
- **Compute Capability**: Enables bf16 on Ampere+ (compute ≥ 8.0)

### Automatic Optimizations

| GPU VRAM | Micro Batch Size | Recommended Train Batch | Offloading |
|----------|------------------|------------------------|-----------|
| **80GB** (A100-80, H100) | 32 | 1024 | No |
| **40GB** (A100-40) | 16 | 512 | No |
| **24GB** (RTX 4090, A5000) | 8 | 256 | Yes |
| **<24GB** | 4 | 128 | Yes |

The notebooks automatically set these values, but you can override them.

---

## 📖 Notebook Details

### 1_verl_complete_training.ipynb

#### Sections Overview

| Section | Name | What It Does |
|---------|------|--------------|
| **0** | Installation | Install verl with your chosen backend |
| **1** | Hardware Detection | Auto-detect GPUs, CUDA, bf16 support |
| **1.5** | Backend Selection | Choose vLLM or SGLang |
| **2** | Cluster Config | Configure single/multi GPU/node setup |
| **3** | Data & Model Config | Set paths and hyperparameters |
| **4** | GRPO | Train with GRPO algorithm |
| **5** | PPO | Train with PPO algorithm |
| **6** | REINFORCE++ | Train with REINFORCE++ |
| **7** | RLOO | Train with RLOO |
| **8** | ReMax | Train with ReMax |
| **9** | Monitoring | TensorBoard, metrics visualization |
| **10** | Checkpoints | Manage training checkpoints |
| **11** | HuggingFace Upload | Share your model |
| **12** | Cleanup | Shutdown Ray, clear GPU memory |

#### Algorithm Comparison

| Algorithm | Type | Requires Critic | Best For | Sample Efficiency |
|-----------|------|----------------|----------|-------------------|
| **GRPO** | On-policy | ❌ No | Quick experiments, lower memory | ⭐⭐⭐ |
| **PPO** | On-policy | ✅ Yes | Stable training, good baseline | ⭐⭐⭐⭐ |
| **REINFORCE++** | On-policy | ❌ No | Simple implementation | ⭐⭐⭐ |
| **RLOO** | On-policy | ❌ No | Low variance | ⭐⭐⭐⭐ |
| **ReMax** | On-policy | ❌ No | Direct reward maximization | ⭐⭐⭐⭐ |

### 2_data_preprocessing.ipynb

#### Supported Datasets

| Dataset | Type | Samples | Difficulty | Preprocessing Time |
|---------|------|---------|------------|-------------------|
| **GSM8K** | Math | 8.5K | Grade school | ~1 min |
| **MATH** | Math | 12.5K | Competition | ~2 min |
| **HH-RLHF** | Dialogue | 160K+ | N/A | ~5 min |
| **Custom** | Any | User-defined | N/A | Varies |

#### Output Format

All datasets are converted to **Parquet** with the following structure:

```python
{
    'data_source': 'gsm8k',           # Dataset identifier
    'prompt': 'What is 25 * 37?',     # Input prompt
    'ability': 'math',                 # Task category (optional)
    'reward_model': {                  # Reward config (optional)
        'style': 'rule',
        'ground_truth': '925'
    },
    'extra_info': {                    # Additional metadata
        'answer': '925',
        'difficulty': 'easy'
    }
}
```

### 3_model_evaluation.ipynb

#### Features

- **Load Checkpoints**: From local paths or HuggingFace
- **Backend Support**: Use vLLM or SGLang for inference
- **Batch Inference**: Efficient evaluation on large test sets
- **Metrics**: Accuracy, rewards, custom metrics
- **Comparison**: Compare multiple checkpoints side-by-side
- **Export**: Save results to Parquet for analysis

---

## 🎯 Cluster Configurations

### Single GPU

For testing or small models:

```python
CLUSTER_CONFIG = {
    'trainer.n_gpus_per_node': 1,
    'trainer.nnodes': 1,
    'actor_rollout_ref.rollout.tensor_model_parallel_size': 1,
}
```

### Single Node, Multiple GPUs

Most common setup (e.g., 8x A100):

```python
CLUSTER_CONFIG = {
    'trainer.n_gpus_per_node': 8,
    'trainer.nnodes': 1,
    'actor_rollout_ref.rollout.tensor_model_parallel_size': 2,
}
```

### Multi-Node, Multi-GPU

For very large models (70B, 671B):

```python
CLUSTER_CONFIG = {
    'trainer.n_gpus_per_node': 8,
    'trainer.nnodes': 2,  # 2 nodes × 8 GPUs = 16 GPUs total
    'actor_rollout_ref.rollout.tensor_model_parallel_size': 4,
    'ray_kwargs.ray_init.address': '192.168.1.100:6379',  # Head node IP
}
```

**Important for multi-node**: You MUST set the Ray head node address.

---

## 🔍 Troubleshooting

### Installation Issues

**Problem**: `ImportError: No module named 'verl'`

```bash
# Solution: Install verl
pip install verl[vllm,gpu,math]
```

**Problem**: `CUDA out of memory`

```python
# Solution: Reduce batch sizes or enable offloading
RECOMMENDED_CONFIG['ppo_micro_batch_size_per_gpu'] = 8  # Lower this
RECOMMENDED_CONFIG['param_offload'] = True  # Enable offloading
```

### Backend Issues

**Problem**: `sglang not found`

```bash
# Solution: Install SGLang
pip install verl[sglang]
```

**Problem**: `vllm version mismatch`

```bash
# Solution: Install compatible vLLM version
pip install "vllm>=0.8.5,<=0.11.0"
```

### Training Issues

**Problem**: Ray cluster won't start

```python
# Solution: Shutdown existing Ray and restart
import ray
ray.shutdown()
# Then re-run your training cell
```

**Problem**: bf16 not supported

The notebooks automatically detect this and fall back to fp16. If you see warnings:
- Your GPU compute capability < 8.0 (pre-Ampere)
- fp16 will be used instead (still works fine)

**Problem**: Multi-node setup not working

```python
# On head node:
ray start --head --port=6379

# On worker nodes:
ray start --address='<HEAD_NODE_IP>:6379'

# In notebook:
CLUSTER_CONFIG['ray_kwargs.ray_init.address'] = '<HEAD_NODE_IP>:6379'
```

### Data Issues

**Problem**: `File not found` when loading data

```python
# Solution: Use absolute paths
DATA_CONFIG['train_files'] = '/home/user/data/gsm8k/train.parquet'
# Or expand ~ properly:
import os
DATA_CONFIG['train_files'] = os.path.expanduser('~/data/gsm8k/train.parquet')
```

**Problem**: Data format errors

Run the validation cell in `2_data_preprocessing.ipynb`, Section 5 to check your data.

---

## 📊 Example Workflows

### Workflow 1: Quick Math Model Training

1. **Prepare data** (`2_data_preprocessing.ipynb`):
   ```python
   # Section 1: Process GSM8K
   GSM8K_CONFIG['output_dir'] = '~/data/gsm8k'
   # Run cell → data saved
   ```

2. **Train** (`1_verl_complete_training.ipynb`):
   ```python
   # Section 1.5: Choose backend
   BACKEND = 'sglang'  # or 'vllm'

   # Section 3: Set paths
   DATA_CONFIG['train_files'] = '~/data/gsm8k/train.parquet'
   MODEL_CONFIG['model_path'] = 'Qwen/Qwen3-8B'

   # Section 4: Run GRPO training
   # Run the cell → training starts
   ```

3. **Evaluate** (`3_model_evaluation.ipynb`):
   ```python
   # Section 2: Load checkpoint
   CHECKPOINT_CONFIG['checkpoint_path'] = './checkpoints/epoch_15'

   # Section 5: Benchmark
   TEST_CONFIG['test_file'] = '~/data/gsm8k/test.parquet'
   # Run cell → get accuracy
   ```

### Workflow 2: Multi-Dataset Training

1. **Prepare multiple datasets** (`2_data_preprocessing.ipynb`):
   - Section 1: GSM8K
   - Section 2: MATH
   - Section 6: Merge datasets

2. **Train** (`1_verl_complete_training.ipynb`):
   ```python
   DATA_CONFIG['train_files'] = '~/data/merged/train.parquet'
   # Run PPO for better sample efficiency
   ```

### Workflow 3: Large Model on Multi-Node

1. **Setup cluster**:
   ```bash
   # On head node (192.168.1.100):
   ray start --head --port=6379

   # On worker node (192.168.1.101):
   ray start --address='192.168.1.100:6379'
   ```

2. **Configure** (`1_verl_complete_training.ipynb`):
   ```python
   # Section 2:
   CLUSTER_CONFIG = {
       'trainer.n_gpus_per_node': 8,
       'trainer.nnodes': 2,
       'actor_rollout_ref.rollout.tensor_model_parallel_size': 4,
       'ray_kwargs.ray_init.address': '192.168.1.100:6379',
   }

   # Section 3:
   MODEL_CONFIG['model_path'] = 'Qwen/Qwen3-70B'  # Large model
   ```

---

## 🆘 Getting Help

- **verl Documentation**: https://verl.readthedocs.io/
- **GitHub Issues**: https://github.com/volcengine/verl/issues
- **Slack**: https://join.slack.com/t/verl-project/...
- **Paper**: [HybridFlow](https://arxiv.org/abs/2409.19256)

## 📝 Best Practices

### 1. Start Small
- Test with single GPU and small dataset first
- Use `TEST_CONFIG['num_samples'] = 100` for quick evaluation
- Scale up after validating the pipeline

### 2. Monitor Training
- Use TensorBoard (Section 9) to watch metrics
- Check for divergence (exploding rewards, NaN losses)
- Save checkpoints frequently (`trainer.save_freq=20`)

### 3. Backend Selection
- Try both vLLM and SGLang on your specific model
- Benchmark inference speed (Section 4 in evaluation notebook)
- Stick with vLLM if you encounter issues with SGLang

### 4. Memory Management
- If OOM: reduce batch sizes, enable offloading
- If slow: disable offloading, increase batch sizes
- Use gradient checkpointing for large models

### 5. Data Quality
- Always validate data (Section 5 in preprocessing notebook)
- Check prompt/response length distributions
- Filter out extremely long sequences

---

## 🎉 Success Stories

The verl framework has been used to train:
- **Doubao-1.5-pro**: SOTA reasoning model (70.0 on AIME)
- **Seed-Thinking-v1.5**: 86.7 on AIME, 55.0 on Codeforces
- **DAPO**: 50 points on AIME 2024

See the main [README](../README.md) for more awesome projects using verl!

---

## 📄 License

Apache 2.0 - Same as verl

## 🙏 Acknowledgments

These notebooks are built on top of the amazing verl framework by ByteDance Seed Team and the verl community.

---

**Happy Training! 🚀**
