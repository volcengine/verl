# SAPO Recipe

This recipe enables Soft Adaptive Policy Optimization (SAPO) by selecting the SAPO policy loss and GRPO advantages.

## Files
- `recipe/sapo/config/sapo_trainer.yaml`: base config with SAPO loss mode and GRPO advantage estimator.
- `recipe/sapo/run_sapo_qwen2.5_0.5b_gsm8k.sh`: end-to-end smoke run for Qwen2.5-0.5B-Instruct on GSM8K.

## Run
```bash
bash recipe/sapo/run_sapo_qwen2.5_0.5b_gsm8k.sh
```

## Tuning
- `actor_rollout_ref.actor.policy_loss.tau_pos`: SAPO gate temperature for positive advantages.
- `actor_rollout_ref.actor.policy_loss.tau_neg`: SAPO gate temperature for negative advantages.
- `actor_rollout_ref.rollout.n`: number of responses per prompt for GRPO.
