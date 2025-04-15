# Dr. GRPO Open-Source Implementation

## Configuration
```yaml
actor_rollout_ref:
  actor:
    loss_agg_mode: "seq-mean-token-sum-norm" # turn off seq-dim averaging 
algorithm:
  norm_adv_by_std_in_grpo: False # turn off standard deviation norm
```