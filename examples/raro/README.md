# RARO Implementation for VERL

This directory contains the implementation of **RARO (Relativistic Adversarial Reasoning Optimization)** based on the paper "Escaping the Verifier: Learning to Reason via Demonstrations" using the VERL framework.

## Overview

RARO is an Inverse Reinforcement Learning approach that trains:
- **Policy**: Generates reasoning chains that attempt to fool a Critic
- **Critic**: Learns to distinguish between Expert and Policy answers

Unlike standard RLHF which uses external verifiers, RARO uses a single model with shared weights that switches roles via system prompts.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Single LLM (Shared Weights)              │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐              ┌──────────────┐             │
│  │   Policy     │              │   Critic     │             │
│  │   (Generator)│              │ (Discriminator)│            │
│  │              │              │              │             │
│  │ Input:       │              │ Input:       │             │
│  │   Question   │              │   Question   │             │
│  │              │              │   Answer 1   │             │
│  │ Output:      │              │   Answer 2   │             │
│  │   Solution   │              │              │             │
│  │   + Answer   │              │ Output:      │             │
│  │              │              │   Analysis   │             │
│  │ Goal: Fool   │              │   + Label    │             │
│  │   Critic     │              │              │             │
│  └──────────────┘              │ Goal: Spot   │             │
│                                │   Expert     │             │
│                                └──────────────┘             │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   Reward Matrix       │
              ├───────────────────────┤
              │ Critic    │ Pol │ Crit│
              │ Prediction│ Reward│ Reward│
              ├───────────┼─────┼─────┤
              │ Expert ✓  │ 0.0 │ 1.0 │
              │ Policy ✗  │ 1.0 │ 0.0 │
              │ Tie ~     │ τ_pol│ τ_crit│
              └───────────────────────┘
```

## Key Components

### 1. Reward Manager (`verl/workers/reward_manager/raro.py`)

The main `RARORewardManager` class handles:
- Dual-pass rollout orchestration
- Reward computation based on adversarial outcome
- Replay buffer management

### 2. Prompt Templates (`verl/workers/reward_manager/raro_prompts.py`)

Defines system prompts for:
- **Policy Role**: Standard assistant prompt for generating solutions
- **Critic Role**: Adversarial judge prompt for comparing answers

### 3. Replay Buffer (`verl/workers/reward_manager/raro_replay_buffer.py`)

Stores historical (question, expert_answer, policy_answer) triplets to prevent catastrophic forgetting in the Critic. Supports:
- **FIFO Buffer**: Standard first-in-first-out queue
- **Reservoir Buffer**: Uniform sampling from stream

### 4. Joint Loss (`verl/trainer/ppo/raro_utils.py`)

Implements the weighted joint objective:
```
J(θ) = λ_pol × J_pol(θ) + λ_crit × J_crit(θ) - β × D_KL
```

## Usage

### Basic Training

```python
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.workers.reward_manager import RARORewardManager
from transformers import AutoTokenizer
from omegaconf import OmegaConf

# Load configuration
cfg = OmegaConf.load("verl/trainer/config/raro.yaml")

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-7B-Instruct")

# Create RARO reward manager
reward_manager = RARORewardManager(
    tokenizer=tokenizer,
    tau_pol=0.6,           # Policy tie reward
    tau_crit=0.55,         # Critic tie reward
    replay_buffer_size=10000,
    replay_buffer_ratio=0.5,
)

# Initialize trainer
trainer = RayPPOTrainer(
    config=cfg,
    tokenizer=tokenizer,
    reward_manager=reward_manager,
)

# Train
trainer.fit()
```

### Using the Configuration File

The `verl/trainer/config/raro.yaml` file contains all RARO-specific settings:

```yaml
reward_manager:
  name: raro
  tau_pol: 0.6
  tau_crit: 0.55
  replay_buffer_size: 10000
  replay_buffer_ratio: 0.5
  max_response_length: 4096
  shuffle_answers: true

algorithm:
  adv_estimator: grpo
  raro:
    lambda_pol: 0.111  # 1/9
    lambda_crit: 0.889  # 8/9
    n_rollouts: 16
```

Run training with:

```bash
python -m verl.trainer.main_ppo \
    --config-path verl/trainer/config \
    --config-name raro \
    data.train_files=path/to/train.json \
    model.path=Qwen/Qwen2.5-Math-7B-Instruct
```

## Configuration Parameters

### Reward Manager

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tau_pol` | 0.6 | Policy reward for Tie outcome |
| `tau_crit` | 0.55 | Critic reward for Tie outcome |
| `replay_buffer_size` | 10000 | Maximum samples in replay buffer |
| `replay_buffer_ratio` | 0.5 | Ratio of historical to fresh samples |
| `max_response_length` | 4096 | Maximum allowed response length |
| `shuffle_answers` | true | Shuffle answer order for Critic robustness |
| `buffer_type` | fifo | Buffer type ('fifo' or 'reservoir') |

### Algorithm

| Parameter | Default | Description |
|-----------|---------|-------------|
| `adv_estimator` | grpo | Advantage estimator (must be 'grpo') |
| `lambda_pol` | 0.111 | Policy loss weight (typically 1/9) |
| `lambda_crit` | 0.889 | Critic loss weight (typically 8/9) |
| `n_rollouts` | 16 | Number of rollouts for GRPO |

## Training Loop

The RARO training loop follows this sequence:

1. **Policy Rollout**
   - Input: Batch of questions
   - Prompt: Policy system prompt
   - Output: Policy answers with reasoning traces
   - Store: (q, a_exp, a_pol) in replay buffer

2. **Data Mixing**
   - Create Critic batch with 50% fresh + 50% historical samples
   - Purpose: Prevent catastrophic forgetting

3. **Critic Rollout**
   - Input: Mixed batch formatted as triplets
   - Prompt: Critic system prompt (comparison task)
   - Output: Critic CoT + label (Expert/Policy/Tie)

4. **Reward Assignment**
   - Parse Critic output to extract label
   - Compute R_pol and R_crit using reward matrix
   - Filter over-length samples

5. **Joint Update**
   - Update both Policy and Critic with weighted loss
   - Apply KL penalty for stability

## Expected Behavior

During training, you should observe:

1. **Critic Accuracy**: Starts near 50% (random), improves over time
2. **Policy Reward**: Increases as Policy learns to generate better solutions
3. **Tie Rate**: Stabilizes around 70% (as reported in the paper)
4. **Mode Collapse**: Prevented by the Tie option and replay buffer

## File Structure

```
verl/
├── workers/
│   └── reward_manager/
│       ├── raro.py              # Main reward manager
│       ├── raro_prompts.py      # Prompt templates
│       └── raro_replay_buffer.py # Replay buffer
├── trainer/
│   ├── ppo/
│   │   └── raro_utils.py        # Joint loss and dual-pass rollout
│   └── config/
│       └── raro.yaml            # RARO configuration
examples/
└── raro/
    ├── README.md                # This file
    └── raro_train.py            # Example training script
```

## Dataset Format

The training data should be in JSON format with the following structure:

```json
[
    {
        "question": "What is 2+2?",
        "answer": "4",
        "solution": "To find 2+2, we add the two numbers..."
    },
    ...
]
```

## Potential Issues and Solutions

### 1. Mode Collapse (Critic always outputs "Tie")
**Solution**: Reduce `tau_crit` to make correct identification more rewarding

### 2. Format Errors (Critic output can't be parsed)
**Solution**: These samples are automatically masked and not trained on

### 3. Out of Memory (OOM)
**Solution**: Reduce `batch_size` or `max_response_length`. The Critic phase doubles context length

### 4. Catastrophic Forgetting
**Solution**: Increase `replay_buffer_ratio` or `replay_buffer_size`

## References

- Paper: "Escaping the Verifier: Learning to Reason via Demonstrations"
- Base Framework: [VERL](https://github.com/volcengine/verl)

## License

This implementation follows the same license as the VERL framework (Apache 2.0).
