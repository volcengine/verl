# SQL Agent with Agent Lightning

This example demonstrates how to use Agent Lightning to train a SQL agent using Agent Lightning + verl framework. The SQL agent is designed to interact with a database, execute SQL queries, and retrieve results to answer user queries.

**The example is tested with verl v0.5.0 and Agent Lightning v0.1.1.**

## Running the Example

1. Prepare the dataset: download from [here](https://drive.google.com/file/d/1oi9J1jZP9TyM35L85CL3qeGWl2jqlnL6/view) and unzip it to the `data` folder.
   ```bash
   pip install gdown
   gdown --fuzzy https://drive.google.com/file/d/1oi9J1jZP9TyM35L85CL3qeGWl2jqlnL6/view
   unzip -q spider-data.zip -d data
   rm spider-data.zip
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the training server:
   ```bash
   python -m agentlightning.verl \
       agentlightning.port=9997 \
       algorithm.adv_estimator=grpo \
       data.train_files=data/train_spider.parquet \
       data.val_files=data/test_dev_500.parquet \
       actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
       trainer.n_gpus_per_node=1 \
       data.train_batch_size=32 \
       actor_rollout_ref.rollout.n=4 \
       actor_rollout_ref.actor.ppo_mini_batch_size=32 \
       actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
       actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
       actor_rollout_ref.rollout.multi_turn.format=hermes \
       actor_rollout_ref.model.path=Qwen/Qwen2.5-Coder-3B-Instruct \
       data.max_prompt_length=4096 \
       data.max_response_length=2048 \
       data.truncation='error' \
       trainer.val_before_train=True \
       actor_rollout_ref.actor.optim.lr=1e-6 \
       actor_rollout_ref.model.use_remove_padding=True \
       actor_rollout_ref.actor.use_kl_loss=False \
       actor_rollout_ref.actor.kl_loss_coef=0.000 \
       actor_rollout_ref.actor.entropy_coeff=0 \
       actor_rollout_ref.actor.clip_ratio_low=0.2 \
       actor_rollout_ref.actor.clip_ratio_high=0.3 \
       actor_rollout_ref.model.enable_gradient_checkpointing=True \
       actor_rollout_ref.actor.fsdp_config.param_offload=True \
       actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
       actor_rollout_ref.rollout.name=vllm \
       actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
       actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
       actor_rollout_ref.ref.fsdp_config.param_offload=True \
       algorithm.use_kl_in_reward=False \
       trainer.critic_warmup=0 \
       trainer.logger=['console','wandb'] \
       trainer.project_name=AgentLightning \
       trainer.experiment_name=train_sql_agent \
       trainer.nnodes=1 \
       trainer.save_freq=256 \
       trainer.test_freq=32 \
       trainer.total_epochs=2 $@
    ```

4. Launch agent clients that connect with the server:
   ```bash
   export VERL_API_BASE=http://localhost:9997/  # Same as the server port. This is used for receiving tasks and sending results.
   python sql_agent.py \
       --litsqlagent.trained-agents write \  # Will only train the write and rewrite agent.
       --trainer.n-workers 16 \
       --litsqlagent.val-temperature 0
   ```

There is no hard requirement in the launching order of the server and clients. But remember to kill the long-running agent clients after the training is done.

## Debug the Agent without verl

You can run the agent client alone without the verl server. This is useful for debugging the agent logic and SQL execution.

1. Copy `.env.example` to `.env` and fill in your OpenAI API key. `VERL_API_BASE` does not really matter here because you are not connecting to the server end.

2. Run the agent client:
   ```bash
   dotenv run python sql_agent.py \
       --litsqlagent.trained-agents write \  # Will only select the trajectories related to write and rewrite.
       --trainer.n-workers 1 \  # For debug, use single process.
       --trainer.dev true  # Enable the dev debug mode.
   ```
