conda activate verl

cd /dfs/project/kgrlm/github/verl/examples/data_preprocess

To run CollabLLM

1) Generate data for training and validation
```bash
python preprocess_collabllm_datasets.py   --dataset collabllm/collabllm-multiturn-math-hard   --local_dir ~/data/collabllm-math-hard 
```

2) Run training with
```bash
 sh recipe/collabllm/train_collabllm.sh  
```

Structure:
1) Collaborative simulation -> SGlang Interation rollout: `verl/interactions/collabllm_interation.py`  
    This interaction class simulates the conversation between the user and the LLM.

    Interaction config is in `verl/examples/sglang_multiturn/config/interaction_config/collabllm_interaction_config.yaml`

2) Reward computation -> `verl/workers/reward_manager/collabllm.py` and `verl/recipe/collabllm/reward_function.py`
    This computes the reward based on the interaction data in step 1.


TODO:
