# Task List

1. ✅ Create multi-turn trajectory rollout skeleton
Implemented skeleton functions for generating n rollouts per question with multi-turn trajectories [(c1,o1),(c2,o2)...] structure
2. ✅ Create multi-turn GRPO advantage estimator
Extended existing GRPO advantage computation to handle multi-turn trajectories with trajectory-level rewards
3. ✅ Create trajectory reward computation skeleton
Implemented reward computation logic: 0 for bad format, 0 for wrong answer, reward=1/(max_context_length) otherwise
4. ✅ Create multi-turn rollout worker integration
Integrated the multi-turn trajectory logic with existing rollout workers and GRPO framework
5. ✅ Add configuration support for multi-turn GRPO
Added configuration options for the new multi-turn GRPO workflow parameters

