# Dr. GRPO Open-Source Implementation


https://github.com/sail-sg/understand-r1-zero


This paper suggests a way to calculate the unbiased policy gradient.


## Configuration
```yaml
actor_rollout_ref:
  actor:
    loss_agg_mode: "seq-mean-token-sum-norm" # turn off seq-dim averaging
    use_kl_loss: False
algorithm:
  norm_adv_by_std_in_grpo: False # turn off standard deviation norm
```

, with all other parameters set same as GRPO.
