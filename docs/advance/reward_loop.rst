Reward Loop
===========

.. _yyding: https://yyding1.github.io

Author: `Yuyang Ding <https://yyding1.github.io>`_

Last updated: 12/3/2025.

.. warning::
   Reward Loop is ready for use, but the API may change in future releases.

Reward Loop is designed to support flexible and user-friendly reward computation, with implementation mostly in ``verl/experimental/reward``.

**Supported Types of Rewards:** Reward Loop covers all typical reward-computation scenarios.

- **Rule-based Reward**: The reward is determined by predefined rules, e.g., checking whether the predicted answer matches the ground truth via simple string matching.
- **Discriminative Reward Model (DisRM)**: The reward is produced by a specified discriminative reward model, such as ``Skywork/Skywork-Reward-Llama-3.1-8B-v0.2``.
- **Generative Reward Model (GenRM)**: The reward is obtained using a generative reward model, for example ``dyyyyyyyy/FAPO-GenRM-4B``.
- **Hybrid Reward Scenarios**: Reward Loop provides interfaces for plugging in reward models, allowing users to define custom reward logic based on their needs (e.g., combining rule-based methods with GenRM).

**Support Training Modes:** Reward Loop supports multiple execution modes for reward training

- **Colocate Mode**: The reward model shares the same resource pool as the actor/rollout/reference models. In this setup, all rollouts must complete first, after which the reward model is awakened to perform inference.
- **Standalone Mode**: The reward model runs on a separate resource pool, independent from the actor/rollout/reference models. In this setup, each sample is evaluated by the reward model immediately after its rollout finishes.

.. image:: https://github.com/yyDing1/verl-materials/blob/main/reward_loop.svg?raw=true

Architecture Design
-------------------

RewardLoopWorker
~~~~~~~~~~~~~~~~~

The ``RewardLoopWorker`` is responsible for handling batch-level reward computation across all supported execution modes, operating in an asynchronous manner.

.. image:: https://github.com/yyDing1/verl-materials/blob/main/reward_loop_worker.svg?raw=true

For each sample, the reward is computed according to the following logic:

- if ``custom_reward_function`` is provided, we directly use user-customized reward function

- if ``custom_reward_function`` is not provided:
   - **reward model is not enabled**: use default rule-based reward function
   - **reward model is discriminative**: compute reward score using disrm
   - **reward model is generative**: this is not permitted (user-customized reward func **must be** provided)

In most cases, we encourage users to define and use their own customized reward functions.

.. code:: python

   @ray.remote
   class RewardLoopWorker:
      async def compute_score_batch(self, data: DataProto) -> list[dict]:
         tasks = []
         for i in range(len(data)):
            tasks.append(asyncio.create_task(self.compute_score(data[i : i + 1])))
         outputs = await asyncio.gather(*tasks)
         return outputs

   async def compute_score(self, data: DataProto) -> dict:
      assert len(data) == 1, "RewardLoopWorker only support single data item"
      if self.config.custom_reward_function.path is not None:
         # directly use user-customized reward function
         return await self.reward_loop.run_single(data)
      else:
         if self.config.reward_model.enable:
            # we assume the rm is disrm
            # genrm must set custom_reward_function
            return await self.compute_score_disrm(data)
         else:
            return await self.reward_loop.run_single(data)


RewardLoopManager
~~~~~~~~~~~~~~~~~

In **standalone mode**, we directly launch one ``RewardLoopWorker`` for each ``AgentLoopWorker`` to handle reward computation independently.

In **colocate mode**, we launch a ``RewardLoopManager`` to

1. launch reward model if enabled
2. manage multiple ``RewardLoopWorker`` instances to handle CPU-intensive tasks such as code.

.. code:: python

   class RewardLoopManager:
      """
      RewardLoopManager run in single controller.
      This class will create reward loop workers and manage them.
      RewardLoopManager will deprecate fsdp/megatron RewardModelWorker in the future.
      """
   def __init__(self, config: DictConfig, rm_resource_pool: RayResourcePool = None):
      self.config = config
      if self.config.reward_model.enable:
         self.reward_model_manager = RewardModelManager(config.reward_model, rm_resource_pool)
         self.reward_router_address = self.reward_model_manager.get_router_address()
      else:
         self.reward_model_manager = None
         self.reward_router_address = None

      self._init_reward_loop_workers()

   def _init_reward_loop_workers(self):
      self.reward_loop_workers = []
      num_workers = self.config.reward_model.get("num_workers", 1)
      node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]

      for i in range(num_workers):
         # Round-robin scheduling over the all nodes
         node_id = node_ids[i % len(node_ids)]
         self.reward_loop_workers.append(
            RewardLoopWorker.options(
               name=f"reward_loop_worker_{i}",
               scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                  node_id=node_id,
                  soft=True,
               ),
            ).remote(self.config, self.reward_router_address)
         )

   def compute_rm_score(self, data: DataProto) -> DataProto:
      """
      Compute reward score for the given data.
      """
      ...


RewardModelManager
~~~~~~~~~~~~~~~~~~

To support flexible and scalable reward model computation, Reward Loop implement a reward router that coordinates requests among multiple reward model servers.

Each reward model runs as an independent server and is registered with the router.
This router will forward the requests to the registered reward servers with load balancing and return the results.
This design allows us to expose a single unified router address to user-defined reward functions, enabling them to access various reward models seamlessly through the same interface.

.. image:: https://github.com/yyDing1/verl-materials/blob/main/reward_loop_full.svg?raw=true

.. code:: python

   class RewardModelManager:
      """Reward model manager."""

      def __init__(
         self,
         config: RewardModelConfig,
         resource_pool: RayResourcePool = None,
      ):
         """
         Initialize the reward model manager.

         Args:
            config (RewardModelConfig): Reward model configuration.
            resource_pool (RayResourcePool, optional): Resource pool. Defaults to None.
         """
         self.config = config
         self.resource_pool = resource_pool
         self._initialize_llm_servers()
         self._initialize_router()
         assert self.config.rollout.skip_tokenizer_init is False, "Reward model should not skip tokenizer init."
         if self.config.rollout.free_cache_engine:
               self.sleep()

User-defined reward functions can be implemented as either synchronous or asynchronous.
``RewardLoopManager`` automatically detects the type of the user-defined function and executes it accordingly, ensuring that the reward computation process remains non-blocking.


User-Customized Reward Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Users can define custom reward functions, for instance, by integrating external generative rewards or rule-based rewards to accommodate diverse scenario requirements.

To facilitate this, the Reward Loop directly exposes the reward model interface, enabling complex reward computation pipelines that involve model-based scoring.
A user-defined reward function may look like the following:

.. code:: python

   async def compute_score_gsm8k(
      data_source: str,
      solution_str: str,
      ground_truth: str,
      extra_info: dict,
      reward_router_address: str,
      reward_model_tokenizer: PreTrainedTokenizer,
   ):
      """Compute the reward score."""

      # Step 1: Prepare prompt and request payload
      grm_prompt = GRM_PROMPT_TEMPLATE.format(problem=extra_info["question"], solution=solution_str)
      messages = [{"role": "user", "content": grm_prompt}]
      sampling_params = {"temperature": 0.7, "top_p": 0.8, "max_tokens": 4096}
      chat_complete_request = {"messages": messages, **sampling_params}

      # Step 2: Send async request to the reward model
      # here, chat_complete sends async http request to the router address
      result = await chat_complete(
         router_address=reward_router_address,
         chat_complete_request=chat_complete_request,
      )

      # Step 3: Parse model response and extract score
      grm_response = result.choices[0].message.content.strip()
      try:
         score_str = grm_response.split("\n\n")[-1].strip()
         score = int(score_str)
      except Exception:
         score = 0

      return {"score": score}

Runable examples are provided in the ``recipe/fapo`` directory for reference.
