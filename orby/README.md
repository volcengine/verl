# Run VERL on interative instance
- Run `mcli interactive --max-duration 120 --gpus 8 --tmux --cluster r7z24p1 --image=whatcanyousee/verl:ngc-cu124-vllm0.8.3-sglang0.4.5-mcore0.12.0-te2.2` to create the interactive instance with VERL docker image, following [VERL installation instructions](https://verl.readthedocs.io/en/latest/start/install.html#install-from-docker-image).
- Use [init_interactive.sh](/orby/scripts/init_interactive.sh) to init the instance. Change commands to your own config. You can also convert the ActionDescription dataset to the format required by VERL.
- Run `bash orby/scripts/run_qwen2_5_vl-7b_grpo.sh` to start training with the ActionDescription dataset.

# Rule-based reward
- The rule-based reward for ActionDescription is defined in [action_description.py](/orby/reward/action_description.py). It consists of 3 rewards: 1) whether the response contains both thinking and answer, 2) whether the action type is correct, 3) whether the action args are correct, e.g., x/y coordinate is within groundtruth bounding box.
- You can create rule-based reward for other dataset following this example. **Use Cursor to save a lot of time!**

# Dataset format
- VERL can read parquet files and we just need to make sure each record has the required features. Check [convert_action_description.py](/orby/data/convert_action_description.py) to see how we convert the dataset with caveats like adjusting bbox coordinates for resized images.

# Supported tasks / datasets
- Grounding
  - Action description dataset: [GRPO training](/orby/scripts/run_qwen2_5_vl_7b_grpo.sh), [offline eval](/orby/scripts/eval_qwen2_5_vl.sh) 
  - Uground dataset: [GRPO training](/orby/scripts/run_uground_grpo.sh)
  - ScreenSpot (v1, v2 and pro): [offline eval](/orby/scripts/eval_screenspot.sh)
- Subtask

# TODO
- Tune batch size, max seq length, etc to obtain best GPU utilization and performance.
- Multinode training (if needed).
