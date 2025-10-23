Reward Loop
===========

Last updated: 10/23/2025.

.. warning::
   Reward Loop is ready for use, but the API may change in future releaes.

Reward Loop is designed for more flexible and easy-to-use reward computation.

**Design goal**:

- Make reward computation more efficient
- Support broader reward model interface (including discriminative and generative models)
- Make user customized reward function more flexible

.. image:: https://github.com/yyDing1/verl-materials/blob/main/reward_loop_overview.svg?raw=true

Async Reward Computation
------------------------

RewardLoopManager
~~~~~~~~~~~~~~~~~

The Reward Loop refactors the design of the reward manager so that each sample is processed asynchronously in the ``run_single`` function.
This asynchronous design enables the Reward Loop to handle multiple reward computations concurrently, significantly improving computation efficiency.

.. code:: python

   class RewardLoopManagerBase(ABC):
      async def run_single(self, data: DataProto) -> dict:
         # ... (data preprocessing)
         if self.is_async_reward_score:
            result = await self.compute_score(
                  data_source=data_source,
                  solution_str=response_str,
                  ground_truth=ground_truth,
                  extra_info=extra_info,
                  reward_router_address=self.reward_router_address,
                  reward_model_tokenizer=self.reward_model_tokenizer,
            )
         else:
            result = await self.loop.run_in_executor(
                  None,
                  lambda: self.compute_score(
                     data_source=data_source,
                     solution_str=response_str,
                     ground_truth=ground_truth,
                     extra_info=extra_info,
                     reward_router_address=self.reward_router_address,
                     reward_model_tokenizer=self.reward_model_tokenizer,
                  ),
            )
         # ... (reward postprocessing)
         return final_result

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

Reward Models and Router
------------------------

To support flexible and scalable reward model computation, RewardLoop implement a reward router that coordinates requests among multiple reward model servers.

Each reward model runs as an independent server and is registered with the router.
This router will forward the requests to the registered reward servers with load balancing and return the results.

This design allows us to expose a single unified router address to user-defined reward functions, enabling them to access various reward models seamlessly through the same interface.

RewardModelManager
~~~~~~~~~~~~~~~~~~

.. image:: https://github.com/yyDing1/verl-materials/blob/main/reward_loop_full.svg?raw=true

`RewardModelManager` will launch multiple reward servers and register them in the reward router.

.. code:: python

   class RewardModelManager:
      """Reward model manager."""

      def __init__(self, config: RewardModelConfig, worker_group: RayWorkerGroup = None):
         """
         Initialize the reward model manager.

         Args:
            config (RewardModelConfig): Reward model configuration.
            worker_group (RayWorkerGroup, optional): Worker group. Defaults to None.
         """
         self.config = config
         self.worker_group = worker_group
         self._initialize_llm_servers()
         self._initialize_router()
         if self.config.rollout.free_cache_engine:
            self.sleep()

Reward Router
~~~~~~~~~~~~~

The router is to forward the requests to the registered reward servers with load balancing.
- For sglang reward servers, we directly use the sglang router to forward the requests.
- For vllm reward servers, we implement a simple round-robin router to forward the requests.

.. code:: python

   class NaiveRouter:
      def __init__(
         self,
         worker_urls: list[str],
         max_connections: int = 1024,
         timeout: int = 60,
         max_attempts: int = 3,
         retry_delay: float = 2.0,
         verbose: bool = False,
      ):
         """A minimal async load-balancing router."""
         self.verbose = verbose
         self.app = FastAPI()
         self.worker_urls = worker_urls
         self.request_counts = {url: 0 for url in worker_urls}

         self.max_connections = max_connections
         self.timeout = timeout
         self.max_attempts = max_attempts
         self.retry_delay = retry_delay

         self.app = FastAPI()

         # Register startup / shutdown hooks
         self.app.on_event("startup")(self._on_startup)
         self.app.on_event("shutdown")(self._on_shutdown)

         # Catch-all proxy route
         self.app.api_route("/{endpoint:path}", methods=["GET", "POST"])(self._make_async_request)

         # Placeholder for aiohttp client
         self.client = None

Agent Reward Loop
-----------------

RewardLoop can be integrated with AgentLoop to enable sample-wise rollout and reward computation.

.. image:: https://github.com/yyDing1/verl-materials/blob/main/agent_reward_loop.svg?raw=true

