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


Modifications:
1) In `verl/workers/rollout/sglang_rollout/sglang_rollout.py`:
```

+               if self.config.multi_turn.collabllm_rollouts:
+                   finish_reason_type = None
+               else:
                    finish_reason_type = FinishReasonTypeEnum.from_str(output["meta_info"]["finish_reason"]["type"])
                    
```

```
else:
                        _req.add_assistant_message(
                            self.processing_class,
                            content,
                        )
                        if (
                            _req.interaction_kwargs
                            and self.interaction_map
                            and user_turns < self.config.multi_turn.max_user_turns
                            and current_turns < self.config.multi_turn.max_assistant_turns
                        ):
                            _req.state = AsyncRolloutRequestStateEnum.INTERACTING
                        else:
+                           # Add ending condition
+                           finish_reason_type = FinishReasonTypeEnum.STOP
                            _req.state = AsyncRolloutRequestStateEnum.COMPLETED
                            break
```

```

+           if self.config.multi_turn.collabllm_rollouts:
+               max_assistant_turns = self.config.multi_turn.max_assistant_turns
+
+               # for collabllm, firstly generate model reponses
+               self.config.multi_turn.max_assistant_turns = 1
+               output_req_list = loop.run_until_complete(
+                   asyncio.gather(
+                       *[
+                           self._async_rollout_a_request(req, do_sample, is_validate, **kwargs) 
+                           for req in req_list
+                       ],
+                       return_exceptions=False
+                   )
+               )
+
+               # then, collect interaction rollouts
+               self.config.multi_turn.max_assistant_turns = max_assistant_turns
+               for req in output_req_list:
+                   req.state = AsyncRolloutRequestStateEnum.INTERACTING
+
+               interaction_requests = [
+                   deepcopy(req) 
+                   for req in output_req_list 
+                   for _ in range(self.config.multi_turn.num_repeat_rollouts)
+               ]
+               interaction_req_list = loop.run_until_complete(
+                   asyncio.gather(
+                       *[
+                           self._async_rollout_a_request(req, do_sample, is_validate, **kwargs)
+                           for req in interaction_requests
+                       ],
+                       return_exceptions=False
+                   )
+               )
+               # merge interaction rollouts back to original responses
+               num_repeats = self.config.multi_turn.num_repeat_rollouts
+               for i, req in enumerate(output_req_list):
+                   start_idx = i * num_repeats
+                   end_idx = start_idx + num_repeats
+                   interaction_batch = interaction_req_list[start_idx:end_idx]
+                   
+                   # Extract messages from interaction rollouts
+                   req.messages = [
+                       interaction.messages for interaction in interaction_batch
+                   ]
+                   req.state = AsyncRolloutRequestStateEnum.COMPLETED
+
+           else:
```