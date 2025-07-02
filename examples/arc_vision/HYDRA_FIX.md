# Fixing the Hydra Configuration Error

## The Problem

You're getting this error:
```
mismatched input 'arc_vision_grpo.yaml' expecting ID
```

This happens because VERL uses Hydra for configuration, and Hydra expects parameter overrides in dot notation, not direct YAML file paths.

## Root Cause

The command being run is:
```bash
python3 -m verl.trainer.main_ppo examples/arc_vision/config/arc_vision_grpo.yaml data.train_files=...
```

But VERL/Hydra expects:
```bash
python3 -m verl.trainer.main_ppo --config-name arc_vision_grpo data.train_files=...
```

## Solutions

### Option 1: Use the Fixed Script (Recommended)

```bash
cd /root/verl/examples/arc_vision
bash run_arc_vision_3b_fixed.sh
```

This script:
- Copies the config to where Hydra expects it (`verl/trainer/config/`)
- Uses the correct `--config-name` syntax
- Includes custom reward function overrides

### Option 2: Use Inline Configuration

```bash
cd /root/verl/examples/arc_vision
bash run_arc_vision_grpo_inline.sh
```

This script:
- Doesn't need any config file
- Specifies all parameters directly
- Easier to debug and modify

### Option 3: Manual Fix

If you want to fix your current script, replace:
```bash
python3 -m verl.trainer.main_ppo examples/arc_vision/config/arc_vision_grpo.yaml \
```

With:
```bash
cp examples/arc_vision/config/arc_vision_grpo.yaml verl/trainer/config/
python3 -m verl.trainer.main_ppo --config-name arc_vision_grpo \
```

## Verification

To confirm the custom reward function is being used:
- Check that `reward_model.enable=false` is set
- Verify `custom_reward_function.path` points to your reward file
- Look for "using customized reward function" in the logs

## Important Notes

1. **Reward Model**: We're NOT using VERL's neural reward model (it's disabled)
2. **Custom Function**: We ARE using the Arc Vision custom reward function
3. **Flash Attention**: Currently enabled with `use_remove_padding=True`