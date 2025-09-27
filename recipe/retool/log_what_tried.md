
# Tried
| Tried                                               | Comments                           |
|-----------------------------------------------------|------------------------------------|
| RAY_memory_usage_threshold=0.95                     | seems to help                      |
| RAY_memory_monitor_refresh_ms=0                     | seems to help                      |
| actor_rollout_ref.rollout.agent.num_workers=1       | not in config page, bit in src code, seems to work |
| (data.)train_batch_size=1                           | works                              |
| (actor_rollout_ref.actor.)ppo_mini_batch_size=1     | works                              |
| infer_tp=1                                          | works                              |
| train_sp=1                                          | works                              |



# Resources
- [ChatGPT trace](https://chatgpt.com/g/g-p-6811e563679c81919cde9697cc820272-arc/c/68d6bcbf-6cbc-8333-882a-817d17abf15f)
- [verl config explanations](https://verl.readthedocs.io/en/latest/examples/config.html)
- [verl performance tuning](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html)