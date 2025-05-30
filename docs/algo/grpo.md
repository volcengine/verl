# Group Relative Policy Optimization (GRPO)

In reinforcement learning, classic algorithms like PPO rely on a "critic" model to estimate the value of actions, guiding the learning process. However, training this critic model can be resource-intensive. ￼

GRPO simplifies this process by eliminating the need for a separate critic model. Instead, it operates as follows: ￼
- Group Sampling: For a given problem, the model generates multiple possible solutions, forming a "group" of outputs. ￼
- Reward Assignment: Each solution is evaluated and assigned a reward based on its correctness or quality.
- Baseline Calculation: The average reward of the group serves as a baseline. ￼
- Policy Update: The model updates its parameters by comparing each solution's reward to the group baseline, reinforcing better-than-average solutions and discouraging worse-than-average ones.

This approach reduces computational overhead by avoiding the training of a separate value estimation model, making the learning process more efficient. For more details, refer to the original paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/pdf/2402.03300)

## Key Components

- No Value Function (Critic-less): unlike PPO, GRPO does not train a separate value network (critic)
- Group Sampling (Grouped Rollouts): instead of evaluating one rollout per input, GRPO generates multiple completions (responses) from the current policy for each prompt. This set of completions is referred to as a group.
- Relative Rewards: within each group, completions are scored (e.g., based on correctness), and rewards are normalized relative to the group.

## Configuration
