# Proximal Policy Optimization (PPO)

Proximal Policy Optimization (PPO) is a family of policy gradient methods for reinforcement learning, proposed by OpenAI in 2017. PPO strikes a balance between simplicity, stability, and performance, making it one of the most widely used algorithms in modern RL applications, including large-scale language model fine-tuning.

Traditional policy gradient methods like REINFORCE or Vanilla Policy Gradient suffer from:

- High variance and sample inefficiency.
- Instability due to large policy updates.

PPO addresses this problem using a clipped surrogate objective that avoids overly large updates without requiring second-order derivatives.

For more technical details regarding PPO, we suggest reading the introduction in the [OpenAI spinning up tutorial](https://spinningup.openai.com/en/latest/algorithms/ppo.html), and the paper [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347).

## Key Components

- Actor-Critic Architecture: PPO requires both an actor model (policy) and a critic model (value function). This differs from other algorithms like GRPO and RLOO that don't require a critic model.

- Generalized Advantage Estimation (GAE): PPO uses GAE for computing advantage values, which helps reduce variance in policy gradient estimates while maintaining low bias.

- Clipped Surrogate Objective: The core of PPO is implemented through the clipped surrogate objective function that limits policy updates.

## Configuration

### Actor and Critic Configs

Most critic configs are similar to those of actors. We list a few key configs under `actor_rollout_ref.actor` below. Note that all configs containing `micro_batch_size` are used to configure the maximum sample or token count per forward or backward pass to avoid GPU OOMs, whose value should not change algorithmic/convergence behavior.

- data.train_batch_size: The global batch size used to generate a set of sampled trajectories/rollouts.

- ppo_mini_batch_size: A set of sampled trajectories is split into multiple sub-batches with batch_size=ppo_mini_batch_size for PPO updates. The ppo_mini_batch_size is a global size across all workers/gpus

- actor_rollout_ref.actor.use_kl_loss: to use kl loss in actor. When used, we are not applying KL in the reward function.

- actor_rollout_ref.actor.clip_ratio: PPO clip ratio

- actor_rollout_ref.actor.ppo_epochs: Number of epochs for PPO updates on one set of sampled data

- actor_rollout_ref.actor.use_kl_loss: Whether to enable kl loss. Default is False.

- actor_rollout_ref.actor.kl_loss_coef: The coefficient of kl loss. Default is 0.001.

- actor_rollout_ref.actor.kl_loss_type: Support kl, abs, mse, low_var_kl and full. How to calculate the kl divergence between actor and reference policy. For
specific options, refer to kl_penalty()

gemma: discount factor

lam: Trade-off between bias and variance in the GAE estimator

adv_estimator: Support gae, grpo, reinforce_plus_plus, reinforce_plus_plus_baseline, rloo

use_kl_in_reward: Whether to enable in-reward kl penalty. Default is False.

kl_penalty: Support kl, abs, mse, low_var_kl and full. How to calculate the kl divergence between actor and reference policy. For specific options, refer to kl_penalty() in core_algos.py .

kl_ctrl: Config for in-reward kl_penalty controller - kl_coef: The (initial) coefficient of in-reward kl_penalty. Default is 0.001. - type: ‘fixed’ for FixedKLController and ‘adaptive’ for AdaptiveKLController. - horizon and target_kl: See source code of AdaptiveKLController for details.








## Reference Example

