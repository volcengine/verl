# Group Relative Policy Optimization (GRPO)

Last updated: 10/13/2026.

In reinforcement learning, classic algorithms like PPO rely on a "critic" model to estimate the value of actions, guiding the learning process. However, training this critic model can be resource-intensive. 

GRPO simplifies this process by eliminating the need for a separate critic model. Instead, it operates as follows:
- Group Sampling: For a given problem, the model generates multiple possible solutions, forming a "group" of outputs.
- Reward Assignment: Each solution is evaluated and assigned a reward based on its correctness or quality.
- Baseline Calculation: The average reward of the group serves as a baseline. 
- Policy Update: The model updates its parameters by comparing each solution's reward to the group baseline, reinforcing better-than-average solutions and discouraging worse-than-average ones.

This approach reduces computational overhead by avoiding the training of a separate value estimation model, making the learning process more efficient. For more details, refer to the original paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/pdf/2402.03300)

## Key Components

- No Value Function (Critic-less): unlike PPO, GRPO does not train a separate value network (critic)
- Group Sampling (Grouped Rollouts): instead of evaluating one rollout per input, GRPO generates multiple completions (responses) from the current policy for each prompt. This set of completions is referred to as a group.
- Relative Rewards: within each group, completions are scored (e.g., based on correctness), and rewards are normalized relative to the group.

## Configuration

Note that all configs containing `micro_batch_size` are used to configure the maximum sample or token count per forward or backward pass to avoid GPU OOMs, whose value should not change algorithmic/convergence behavior.

Despite that many configurations start with the `ppo_` prefix, they work across different RL algorithms in verl, as the GRPO training loop is similar to that of PPO (without critic).

![image](https://github.com/user-attachments/assets/16aebad1-0da6-4eb3-806d-54a74e712c2d)

- `actor_rollout_ref.rollout.n`: For each prompt, sample n times. Default to 1. For GRPO, please set it to a value larger than 1 for group sampling.

- `data.train_batch_size`: The global batch size of prompts used to generate a set of sampled trajectories/rollouts. The number of responses/trajectories is `data.train_batch_size * actor_rollout_ref.rollout.n`.

- `actor_rollout_ref.actor.ppo_mini_batch_size`: The global actor mini-batch size in prompt-equivalent units. The trainer multiplies this configured value by `actor_rollout_ref.rollout.n` to obtain the number of trajectories per optimizer update, then partitions those trajectories across actor data-parallel ranks. It is not a per-GPU size.

- `actor_rollout_ref.actor.ppo_epochs`: Number of epochs for GRPO updates on one set of sampled trajectories for actor

- `actor_rollout_ref.actor.clip_ratio`: The GRPO clip range. Default to 0.2

- `algorithm.adv_estimator`: Default is gae. Please set it to grpo instead

- `actor_rollout_ref.actor.loss_agg_mode`: Default is "token-mean". Options include "token-mean", "seq-mean-token-sum", "seq-mean-token-mean". The original GRPO paper takes the sample-level loss (seq-mean-token-mean), which may be unstable in long-CoT scenarios. All GRPO example scripts provided in verl uses the default configuration "token-mean" for loss aggregation instead.

Instead of adding KL penalty in the reward, GRPO regularizes by directly adding the KL divergence between the trained policy and the reference policy to the loss:

- `actor_rollout_ref.actor.use_kl_loss`: To use kl loss in the actor. When used, we are not applying KL in the reward function. Default is False. Please set it to True for GRPO.

- `actor_rollout_ref.actor.kl_loss_coef`: The coefficient of kl loss. Default is 0.001.

- `actor_rollout_ref.actor.kl_loss_type`: Support kl(k1), abs, mse(k2), low_var_kl(k3) and full. Appending "+" in the end (e.g., 'k1+' and 'k3+') would apply straight through to employ k2 for unbiased gradient estimation, regardless of the kl value estimation (see https://github.com/verl-project/verl/pull/2953#issuecomment-3162113848 for more details). How to calculate the kl divergence between actor and reference policy. See this blog post for detailed analysis: http://joschu.net/blog/kl-approx.html

(grpo-batch-size-example)=
### Batch sizes and optimizer updates

The following counts apply to a complete synchronous rollout batch in which each sampled response produces one training trajectory, with actor updates enabled. They do not describe extra trajectories emitted by custom multi-output agent loops, dynamic filtering, synthetic padding, or asynchronous replay-buffer sampling.

| Symbol | Setting or quantity | Unit |
| --- | --- | --- |
| `B` | `data.train_batch_size` | Prompts in one global rollout batch |
| `n` | `actor_rollout_ref.rollout.n` | Responses sampled per prompt |
| `M` | `actor_rollout_ref.actor.ppo_mini_batch_size` | Global prompt-equivalent mini-batch size, expanded to `M * n` trajectories |
| `d` | Actor data-parallel size | Ranks that partition the actor training batch |
| `m` | `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` | Trajectories per local forward/backward pass when `use_dynamic_bsz=False` |
| `E` | `actor_rollout_ref.actor.ppo_epochs` | Passes over the sampled training batch |

Both PPO trainers expand `M` by `n` before calling the actor update. The worker then divides that trajectory count by the actor engine's data-parallel size. "Prompt-equivalent" describes the configured size; it does not guarantee that all responses to a prompt stay together after shuffling or batch balancing.

For evenly divisible fixed-size batches:

```text
trajectories per rollout batch          = B * n
trajectories per global actor mini-batch = M * n
trajectories per local actor mini-batch  = M * n / d
micro-batches per optimizer update      = M * n / (d * m)
actor optimizer updates per rollout batch = (B / M) * E
```

Use the **actor** data-parallel size for `d`, not the rollout tensor-parallel size or the total number of GPUs unless all actor GPUs are data-parallel ranks. For example, actor tensor, pipeline, or sequence parallelism can make `d` smaller than the total actor GPU count.

#### Worked example

Consider these settings with 8 actor data-parallel ranks and fixed micro-batching:

```yaml
data.train_batch_size: 1024
actor_rollout_ref.rollout.n: 5
actor_rollout_ref.actor.ppo_mini_batch_size: 256
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu: 80
actor_rollout_ref.actor.use_dynamic_bsz: false
actor_rollout_ref.actor.ppo_epochs: 1
```

One rollout batch samples `1024 * 5 = 5120` trajectories. Each global actor mini-batch contains `256 * 5 = 1280` trajectories, so each actor DP rank receives `1280 / 8 = 160` trajectories per optimizer update. Each rank processes two micro-batches of 80 trajectories and accumulates their gradients before the optimizer step.

Thus, `80` does **not** need to divide the configured `256`; it needs to divide the local expanded mini-batch size `160`. The rollout batch produces `1024 / 256 = 4` actor optimizer updates with `ppo_epochs=1`, or 8 updates with `ppo_epochs=2`. A trainer step consumes the rollout batch; it is not necessarily a single optimizer update. Increasing `ppo_epochs` reuses the sampled trajectories rather than generating another `B * n` responses for each epoch.

For this fixed-size case, check that `B` is divisible by `M`, `M * n` is divisible by `d`, and `M * n / d` is divisible by `m`. These are batch-count constraints, not a guarantee that the configuration fits in GPU memory.

#### Dynamic and forward-only micro-batches

With `actor_rollout_ref.actor.use_dynamic_bsz=True`, micro-batches are formed using the token budget `actor_rollout_ref.actor.ppo_max_token_len_per_gpu` and sequence lengths instead of the fixed trajectory count `m`. The fixed-size accumulation formula above no longer predicts the number of micro-batches. The global rollout and mini-batch counts remain the same under the stated assumptions.

`actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu` and `actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu` control local **forward-only** batches for actor and reference log-probability evaluation. They do not set the actor optimizer mini-batch size or its gradient-accumulation count. Their corresponding `log_prob_use_dynamic_bsz` and `log_prob_max_token_len_per_gpu` settings select token-budget batching instead when enabled.

## Advanced Extensions

### DrGRPO

[Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/pdf/2503.20783) claims there's optimization bias in GRPO, which leads to artificially longer responses, especially for incorrect outputs. This inefficiency stems from the way GRPO calculates advantages using group-based reward normalization. Instead, DrGRPO aggregates token-level losses by normalizing with a global constant to eliminate length bias.

Configure the following to enable DrGRPO, with all other parameters the same as GRPO's:

- `actor_rollout_ref.actor.loss_agg_mode`: "seq-mean-token-sum-norm", which turns off seq-dim averaging
- `actor_rollout_ref.actor.loss_scale_factor`: (Optional) Set to a constant integer (e.g., max response length) to ensure consistent normalization throughout training. If not set, uses the current batch's response length.
- `actor_rollout_ref.actor.use_kl_loss`: Please set it to False for DrGRPO
- `algorithm.norm_adv_by_std_in_grpo`: False, which turns off standard deviation norm

## Reference Example

Qwen2.5 GRPO training log and commands: [link](https://github.com/eric-haibin-lin/verl-data/blob/experiments/gsm8k/qwen2-7b-fsdp2.log)

```bash
bash examples/grpo_trainer/run_qwen3_8b_fsdp.sh
```

For more reference performance, please see https://verl.readthedocs.io/en/latest/algo/baseline.html
