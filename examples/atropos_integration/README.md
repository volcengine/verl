# Atropos-VERL Integration

This directory contains a complete integration solution for using Atropos with VERL, addressing the **critical missing piece**: policy weight synchronization during RL training.

## The Core Problem

The original question was about **how inference/policy weights get updated every step** during RL training. In proper RL, you need:

1. **Rollout**: Generate sequences with current policy
2. **Training**: Update policy weights using advantages  
3. **Weight Sync**: Update inference engine with new weights
4. **Repeat**: Next rollout uses updated policy

Without step 3, your inference engine uses stale weights, leading to poor RL training.

## Our Solution

This integration demonstrates how VERL's **Sharding Manager** system solves this challenge with automatic weight synchronization.

## Files

### `atropos_example.py`
Complete demonstration of the RL loop with weight synchronization:
- `MockInferenceEngine`: Represents vLLM/SGLang inference engine
- `MockShardingManager`: Shows VERL's weight sync mechanism  
- `MockAtroposRLTrainer`: Complete RL trainer with proper weight management

### `verl/trainer/atropos_sft_trainer.py` 
Extended VERL SFT trainer with advantage weighting support:
- Advantage-weighted cross-entropy loss computation
- Token-level advantage normalization and clipping
- Compatible with VERL's FSDP infrastructure
- Direct interface for computing advantage-weighted SFT loss

### `verl/trainer/config/atropos_sft_trainer.yaml`
Configuration template for Atropos SFT training with advantage weighting:
- Standard VERL SFT configuration options
- Atropos-specific advantage weighting settings
- Data loading and API configuration options

## Quick Start

Run the complete integration demo:

```bash
python atropos_example.py
```

This demonstrates:
1. **Automatic weight synchronization** before each rollout
2. **Advantage-weighted training** with proper loss computation
3. **Memory optimization** with automatic inference engine offloading
4. **Complete RL loop** where each step uses updated policy weights

## Key Integration Points

### 1. Weight Synchronization (The Missing Piece!)

```python
# VERL's magic: automatic weight sync via context manager
with sharding_manager:
    # Automatically syncs training weights → inference engine
    # Does rollout with updated weights
    # Releases memory after inference
    rollout_data = inference_engine.generate(prompts)
```

### 2. Advantage-Weighted Loss

```python
# Core interface for Atropos integration
loss = trainer.compute_advantage_weighted_sft_loss(
    input_ids=tokens,      # Shape: (batch_size, seq_len)
    advantages=advantages, # Shape: (batch_size, seq_len) 
    loss_mask=loss_mask   # Shape: (batch_size, seq_len)
)
```

### 3. Complete RL Loop

```python
def rl_training_step(self, prompts):
    # 1. Rollout with automatic weight sync
    with self.sharding_manager:
        rollout_data = self.rollout_phase(prompts)
    
    # 2. Compute advantages from Atropos environment
    advantages = self.compute_advantages(rollout_data)
    
    # 3. Train with advantage-weighted loss
    loss = self.training_phase(rollout_data, advantages)
    
    # 4. Next rollout automatically uses updated weights!
    return loss
```

## Integration Options

### Option 1: Use VERL Sharding Managers (Recommended)

```python
from verl.workers.sharding_manager.fsdp_sglang import FSDPSGLangShardingManager

# Most compatible approach - leverages VERL's optimized infrastructure
sharding_manager = FSDPSGLangShardingManager(
    module=training_model,
    inference_engine=sglang_engine,
    model_config=config
)
```

### Option 2: Bridge with Atropos ChatScheduler

```python
# If Atropos has its own ChatScheduler, create a bridge
class AtroposChatSchedulerBridge:
    def sync_weights_to_atropos(self):
        state_dict = self.training_model.state_dict()
        self.atropos_scheduler.update_policy_weights(state_dict)
```

### Option 3: Custom Weight Sync

```python
# Full control implementation
def sync_weights(self):
    state_dict = self.training_model.state_dict()
    self.inference_engine.update_weights_from_tensor(state_dict.items())
```

## Example Output

```
============================================================
RL TRAINING STEP 0
============================================================
ROLLOUT PHASE (Step 0)

ENTERING SHARDING MANAGER
   Syncing training weights → inference engine...
   Updating inference engine weights...
   Updated 149 weight tensors
   Weight synchronization complete!
Generating with inference engine...
EXITING SHARDING MANAGER

COMPUTING ADVANTAGES
   Computed advantages shape: torch.Size([2, 17])
TRAINING PHASE
   Training loss: -1.5121
```

## AtroposSFTTrainer Usage

The `AtroposSFTTrainer` extends VERL's standard SFT trainer with advantage weighting:

```python
from verl.trainer.atropos_sft_trainer import AtroposSFTTrainer

# Initialize trainer with advantage weighting config
trainer = AtroposSFTTrainer(
    config=config,  # Include use_advantage_weighting=True
    device_mesh=device_mesh,
    ulysses_device_mesh=ulysses_device_mesh,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    val_dataset=val_dataset
)

# Direct advantage-weighted loss computation
loss = trainer.compute_advantage_weighted_sft_loss(
    input_ids=input_ids,
    advantages=advantages,
    loss_mask=loss_mask
)
```

## Configuration

Key configuration options in `atropos_sft_trainer.yaml`:

```yaml
# Enable advantage weighting
use_advantage_weighting: true
advantage_normalization: "batch"  # "none", "batch", "global"
advantage_clipping: [-5.0, 5.0]  # [min, max] or null

# Data configuration for Atropos integration
data:
  atropos:
    api_url: "http://localhost:8000"  # Optional API endpoint
    data_path: "path/to/data.jsonl"   # Optional static data
    batch_size: 32
    refresh_interval: 10
```

## Production Integration

For production use with Atropos:

1. **Replace mock classes** with real Atropos environment APIs
2. **Use appropriate sharding manager** (FSDP+SGLang, FSDP+vLLM, etc.)
3. **Configure memory optimization** for your GPU setup
4. **Add proper error handling** for weight sync failures
5. **Monitor weight sync overhead** vs training time
6. **Use AtroposSFTTrainer** for advantage-weighted training

## Why This Matters

This integration ensures that:
- Each rollout uses the **latest policy weights**
- Training updates are **immediately available** for next rollout  
- Memory is **efficiently managed** during weight transfers
- The RL loop maintains **proper training dynamics**

Without proper weight synchronization, you get:
- Stale policy weights during rollouts
- Poor RL training convergence
- Inconsistent policy behavior
- Wasted compute on outdated rollouts

## Next Steps

1. **Run** `atropos_example.py` to see the complete flow in action
2. **Review** `verl/trainer/atropos_sft_trainer.py` for the production-ready trainer
3. **Configure** using `verl/trainer/config/atropos_sft_trainer.yaml` as a template
4. **Adapt** the sharding manager approach to your Atropos setup
5. **Test** weight synchronization in your environment
6. **Deploy** with proper monitoring and error handling

The key insight is that **VERL's sharding manager system already solves the weight synchronization challenge** - you just need to integrate it properly with Atropos!