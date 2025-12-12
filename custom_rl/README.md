# Custom SFT Training Scripts

This directory contains custom Supervised Fine-Tuning (SFT) scripts for training models before applying reinforcement learning algorithms.

## Directory Structure

```
custom_sft/
├── README.md                # This file - training plans and documentation
├── configs/                 # Custom configuration files for SFT
├── scripts/                 # Training scripts for different models
├── data_preprocessing/      # Data preparation scripts
└── checkpoints/            # Directory for saving SFT model checkpoints
```

## Training Plan

### 1. Data Preparation
- [ ] Prepare dataset in the required format
- [ ] Create data preprocessing scripts
- [ ] Validate data quality and format

### 2. SFT Training
- [ ] Configure model and training parameters
- [ ] Run SFT training for baseline model
- [ ] Monitor training metrics (loss, validation scores)
- [ ] Save best checkpoint for RL training

### 3. Evaluation
- [ ] Evaluate SFT model performance
- [ ] Compare with base model
- [ ] Decide if ready for RL training

## Key Files for SFT in verl

### Core SFT Implementation
- **Main SFT Trainer**: `verl/trainer/fsdp_sft_trainer.py` - The core FSDP-based SFT implementation
- **SFT Configuration**: `verl/trainer/config/sft_trainer.yaml` - Default configuration template
- **Main Entry Point**: `verl/trainer/main_sft.py` - Entry point for SFT training

### Example Scripts to Reference
- **Simple Example**: `recipe/char_count/train_sft.sh` - Beginner-friendly example
- **GSM8K Examples**: `examples/sft/gsm8k/` - Various model examples:
  - `run_deepseek_7b.sh` - DeepSeek 7B model
  - `run_llama3_8b.sh` - Llama 3 8B model
  - `run_qwen_05.sh` - Qwen 0.5B model
- **Advanced Example**: `recipe/retool/run_qwen2-32b_sft.sh` - Large model with advanced configs

### Data Processing Examples
- `examples/data_preprocess/gsm8k/prepare_gsm8k_sft_data.py` - GSM8K data preparation
- `recipe/char_count/create_dataset.py` - Simple dataset creation example

## Quick Start

1. **Prepare your data**:
   ```bash
   # Reference existing examples or create your own
   python data_preprocessing/prepare_data.py
   ```

2. **Configure training**:
   - Copy `verl/trainer/config/sft_trainer.yaml` to `configs/sft_config.yaml`
   - Adjust hyperparameters as needed
   - Reference example configs in `examples/sft/gsm8k/`

3. **Run SFT training**:
   ```bash
   # Create your script based on examples
   bash scripts/train_sft.sh
   ```

4. **Evaluate results**:
   ```bash
   python scripts/evaluate_sft.py --checkpoint-path checkpoints/best_model
   ```

## Next Steps: RL Training

After successful SFT training, proceed to RL training using your SFT checkpoint.

## Understanding RL Training in verl

### What is RL Training for LLMs?

Reinforcement Learning (RL) training for LLMs optimizes the model to generate better responses based on reward signals. Unlike SFT which learns from fixed examples, RL training:
- **Generates** responses from the model
- **Evaluates** them using a reward function
- **Updates** the model to maximize rewards

### Key Components

1. **Actor Model**: Your SFT-trained model that generates responses
2. **Critic Model**: Estimates the value/quality of generated responses (not needed for GRPO)
3. **Reward Model**: Scores how good a response is (can be external API, human feedback, or another model)
4. **Reference Model**: Original model used to prevent the actor from deviating too much (KL regularization)

### Supported RL Algorithms

- **PPO (Proximal Policy Optimization)**: Most popular, stable training with separate critic model
- **GRPO (Group Relative Policy Optimization)**: More efficient, critic-free variant
- **REINFORCE++**: Simple policy gradient method
- **RLOO**: Leave-one-out baseline method
- **Others**: ReMax, PRIME, DAPO, DrGRPO

## Key Files for RL Training

### Core RL Implementation
- **Main Entry Point**: `verl/trainer/main_ppo.py` - Orchestrates the entire RL training
- **PPO Trainer**: `verl/trainer/ppo/ray_trainer.py` - Core PPO training logic
- **PPO Algorithm**: `verl/single_controller/ppo_algo.py` - PPO algorithm implementation
- **Configuration**: `verl/trainer/config/ppo_trainer.yaml` - Default PPO configuration

### Worker Components
- **Actor-Rollout Worker**: `verl/workers/fsdp_workers.py` - Manages policy model and generation
- **Critic Worker**: `verl/workers/fsdp_workers.py` - Value estimation
- **Reward Worker**: `verl/workers/reward_worker.py` - Reward computation

### Example Scripts
- **PPO Examples**: `examples/ppo_trainer/`
  - `run_deepseek_7b.sh` - DeepSeek 7B model
  - `run_qwen_05b.sh` - Qwen 0.5B model
- **GRPO Example**: `recipe/char_count/train_grpo.sh` - Simple GRPO training
- **Advanced Examples**: `recipe/gsm8k/`, `recipe/retool/`

## Quick Start RL Training

### 1. Prepare RL Dataset
RL datasets need prompts (without responses):
```python
# Format: {"prompt": "Question: What is 2+2?"}
# The model will generate responses during training
```

### 2. Configure RL Training
Key parameters to adjust:
```yaml
# Model path - use your SFT checkpoint
actor_rollout_ref.model.path: /path/to/sft/checkpoint

# Training parameters
data.train_batch_size: 128
actor_rollout_ref.actor.optim.lr: 1e-6  # Lower than SFT
actor_rollout_ref.actor.ppo_mini_batch_size: 16

# Algorithm settings
algorithm.adv_estimator: grpo  # or "gae" for standard PPO
actor_rollout_ref.actor.use_kl_loss: true  # Prevent overfitting
actor_rollout_ref.actor.kl_loss_coef: 0.1
```

### 3. Run RL Training

**For PPO**:
```bash
python -m verl.trainer.main_ppo \
    actor_rollout_ref.model.path=/your/sft/checkpoint \
    data.train_files=/path/to/rl/train.parquet \
    data.val_files=/path/to/rl/val.parquet
```

**For GRPO** (simpler, more stable):
```bash
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=/your/sft/checkpoint \
    # ... other parameters
```

## RL Training Workflow

1. **Rollout Phase**: Generate responses for prompts
2. **Reward Phase**: Score the generated responses
3. **Advantage Estimation**: Calculate how much better/worse each action was
4. **Policy Update**: Update actor model using PPO/GRPO
5. **Value Update**: Update critic model
6. **Repeat**: Continue for multiple epochs

## Key Differences from SFT

| Aspect | SFT | RL Training |
|--------|-----|-------------|
| **Data** | Prompts + Responses | Prompts only |
| **Learning** | Supervised (fixed targets) | Reinforcement (exploration) |
| **Batch Size** | Larger (256+) | Smaller (32-128) |
| **Learning Rate** | Higher (1e-5) | Lower (1e-6) |
| **Training Time** | Faster | Slower (generation needed) |
| **Memory Usage** | Lower | Higher (multiple models) |

## Tips for RL Training

1. **Start with GRPO**: Simpler and more stable than PPO
2. **Use KL regularization**: Prevents model from deviating too much
3. **Monitor rewards**: Ensure rewards are increasing but not exploding
4. **Smaller batches**: RL is more unstable, smaller batches help
5. **Lower learning rate**: Typically 10x lower than SFT
6. **Checkpoint frequently**: RL can be unstable

## Common Issues and Solutions

- **Out of Memory**: Reduce batch size or use LoRA
- **Reward Hacking**: Model finds shortcuts - improve reward function
- **Unstable Training**: Lower learning rate, increase KL coefficient
- **Slow Training**: Use vLLM for faster generation

## Notes

- SFT provides a good initialization for RL training
- RL training typically runs for fewer epochs than SFT (1-2 epochs)
- Monitor both rewards and response quality during training
- The SFT → RL pipeline is the standard RLHF approach

## Deep Dive: GRPO (Group Relative Policy Optimization)

### What is GRPO?

GRPO is a simplified RL algorithm that eliminates the need for a separate critic (value) model. Instead of using a critic to estimate the value of states, GRPO uses group-based sampling and relative rewards within each group.

### How GRPO Works

1. **Group Sampling**: For each prompt, generate multiple (n>1) responses using the current policy
2. **Reward Computation**: Calculate rewards for all responses in the group
3. **Baseline Calculation**: Use the group's mean reward as the baseline
4. **Advantage Estimation**: Normalize each response's reward: `(reward - mean) / (std + epsilon)`
5. **Policy Update**: Update the policy using these normalized advantages

### GRPO Architecture

```
Input Prompts → Actor Model → Multiple Responses per Prompt
                                        ↓
                              Reward Model → Rewards
                                        ↓
                              Group Statistics (mean, std)
                                        ↓
                              Normalized Advantages
                                        ↓
                              Policy Update (PPO-style)
```

### Key Implementation Details

#### Core Algorithm (`verl/trainer/ppo/core_algos.py`):
- `compute_grpo_outcome_advantage`: Computes group-relative advantages
- Groups responses by prompt index
- Calculates per-group statistics
- Normalizes rewards within each group

#### Configuration Parameters:
```yaml
# Essential GRPO settings
algorithm.adv_estimator: grpo              # Enable GRPO
actor_rollout_ref.rollout.n: 5             # Samples per prompt (must be >1)
actor_rollout_ref.actor.use_kl_loss: true  # KL regularization in loss
algorithm.use_kl_in_reward: false          # No KL in reward for GRPO

# Hyperparameters
actor_rollout_ref.actor.kl_loss_coef: 0.001
algorithm.norm_adv_by_std_in_grpo: true    # Normalize by std (set false for DrGRPO)
```

### GRPO vs PPO Comparison

| Feature | PPO | GRPO |
|---------|-----|------|
| **Critic Model** | Required | Not needed |
| **Memory Usage** | High (2 models) | Low (1 model) |
| **Training Speed** | Slower | Faster |
| **Sampling** | 1 response/prompt | n responses/prompt |
| **Baseline** | Critic prediction | Group mean |
| **Stability** | Good | Better |
| **Complexity** | Higher | Lower |

### Example GRPO Script

```bash
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=Qwen/Qwen3-8B \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    algorithm.use_kl_in_reward=false \
    data.train_batch_size=1024 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.optim.lr=1e-6
```

### DrGRPO Variant

DrGRPO addresses optimization bias in GRPO by:
- Disabling standard deviation normalization
- Using different loss aggregation to eliminate length bias

```yaml
# DrGRPO specific settings
algorithm.norm_adv_by_std_in_grpo: false
actor_rollout_ref.actor.loss_agg_mode: seq-mean-token-sum-norm
actor_rollout_ref.actor.use_kl_loss: false
```

### When to Use GRPO

**Use GRPO when:**
- You want simpler, more stable training
- Memory is limited (no critic model)
- You're new to RL training
- Your reward function is reliable

**Use PPO when:**
- You need fine-grained value estimates
- Your reward is sparse or noisy
- You have complex environments
- Memory is not a constraint

### GRPO Best Practices

1. **Group Size**: Use n=5-10 samples per prompt
2. **Batch Size**: Can use larger batches (1024) due to no critic
3. **Learning Rate**: Start with 1e-6, adjust based on stability
4. **KL Coefficient**: Start with 0.001, increase if model diverges
5. **Epochs**: Usually 10-20 epochs work well
6. **Monitoring**: Watch reward distribution within groups

### Common GRPO Issues

- **All responses identical**: Increase temperature or top_p during sampling
- **High variance in groups**: Increase group size or reduce learning rate
- **Model not improving**: Check reward function, may need adjustment
- **Length bias**: Try DrGRPO variant