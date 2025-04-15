# Dr. GRPO Open-Source Implementation

## Configuration
```yaml
actor_rollout_ref:
  actor:
    loss_agg_mode: "seq-mean-token-sum-norm" # turn off seq-dim averaging 
algorithm:
  scale_grpo_adv: False # turn off standard deviation norm
```