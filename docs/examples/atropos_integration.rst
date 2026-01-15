.. _atropos-integration-page:

Atropos-VERL Integration
========================

Overview
--------

The Atropos-VERL integration provides a production-ready implementation for using Atropos RL environments with VERL infrastructure, including **GRPO with token-level advantage overrides** and **GPRO (Group Relative Policy Optimization)** for automatic policy weight synchronization during RL training.

Key Features
~~~~~~~~~~~

- **VERL Inference Engines**: vLLM/SGLang with weight synchronization
- **GRPO Training**: Token-level advantage overrides from Atropos environments
- **GPRO Advantage-weighted SFT**: Production loss computation using VERL's GPRO implementation
- **Complete RL Training Loop**: Rollout → GPRO advantage computation → Training → Weight sync
- **Single-GPU Training**: Production training with GPRO advantage-weighted SFT
- **Production Data Loading**: Real datasets (GSM8K, MATH) with VERL's RL dataset format
- **API Integration**: Complete Atropos API integration with GPRO fallback when API unavailable

Installation
-----------

The Atropos integration is included with VERL. No additional installation is required.

.. code:: bash

   # Verify VERL installation
   python -c "import verl; print(verl.__version__)"

Quick Start
-----------

Basic Usage
~~~~~~~~~~

.. code:: python

   from recipe.atropos.main_atropos import AtroposRLTrainer

   # Configuration
   config = {
       "atropos": {"api_url": "http://localhost:9001", "timeout": 30},
       "use_advantage_weighting": True,
       "use_gpro": True,
       "gpro_epsilon": 1e-6,
       "gpro_norm_by_std": True,
       "advantage_normalization": "batch",
       "advantage_clipping": [-3.0, 3.0],
       "model_path": "microsoft/DialoGPT-medium",
       "device": "cuda",
       "batch_size": 4,
       "max_response_length": 32
   }

   # Initialize trainer
   trainer = AtroposRLTrainer(config)

   # Run training step
   prompts = ["What is 2+2?", "Explain quantum computing"]
   result = trainer.rl_training_step(prompts)
   print(f"Loss: {result['loss']:.4f}")

Command Line Usage
~~~~~~~~~~~~~~~~~

.. code:: bash

   # Single GPU demo
   python recipe/atropos/main_atropos.py

   # Single-GPU training
   python recipe/atropos/launch_atropos_verl.py --mode training

   # GRPO training with Atropos token-level advantages
   python recipe/atropos/example_gsm8k_grpo.py --config-path recipe/atropos/config --config-name gsm8k_grpo_example

   # Launch Atropos + vLLM + GRPO services
   python recipe/atropos/launch_atropos_verl_services.py --config recipe/atropos/config/gsm8k_grpo_example.yaml

   # Run tests
   python recipe/atropos/test_atropos_integration.py
   python recipe/atropos/tests/test_integration.py

API Reference
------------

AtroposRLTrainer
~~~~~~~~~~~~~~~

The main RL trainer class that implements the complete Atropos-VERL integration.

.. code:: python

   class AtroposRLTrainer:
       def __init__(self, config: Dict[str, Any], device_mesh=None):
           """
           Initialize Atropos RL trainer with VERL infrastructure.
           
           Args:
               config: Configuration dictionary
               device_mesh: Optional device mesh for distributed training
           """

       def rl_training_step(self, prompts: List[str]) -> Dict[str, Any]:
           """
           Complete RL training step with GPRO advantage computation.
           
           Args:
               prompts: List of input prompts
               
           Returns:
               Dictionary containing loss, advantages, step, and rollout data
           """

Configuration
-------------

Basic Configuration
~~~~~~~~~~~~~~~~~~

.. code:: yaml

   # Atropos API configuration
   atropos:
     api_url: "http://localhost:9001"
     timeout: 30

   # GPRO configuration
   use_gpro: true
   gpro_epsilon: 1e-6
   gpro_norm_by_std: true

   # Training configuration
   use_advantage_weighting: true
   advantage_normalization: "batch"  # "none", "batch", "global"
   advantage_clipping: [-3.0, 3.0]
   batch_size: 4
   max_response_length: 32

   # Model configuration
   model_path: "microsoft/DialoGPT-medium"
   device: "cuda"

GPRO Configuration
~~~~~~~~~~~~~~~~~

The integration uses VERL's production GPRO implementation with the following parameters:

- **`use_gpro`**: Enable GPRO advantage computation (default: True)
- **`gpro_epsilon`**: Numerical stability for GPRO (default: 1e-6)
- **`gpro_norm_by_std`**: Normalize advantages by standard deviation (default: True)

Data Loading Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: yaml

   data:
     data_source: "atropos_integration"
     max_prompts: 10
     prompt_format: "chat"
     parquet_paths:
       - "~/data/rlhf/gsm8k/train.parquet"
       - "~/data/rlhf/math/train.parquet"
     hf_datasets: ["gsm8k", "math", "hellaswag"]
     max_prompt_length: 512
     max_response_length: 32
     ability: "general"

GPRO Algorithm
--------------

The integration uses VERL's production GPRO implementation:

.. code:: python

   from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage

   # Compute advantages using GPRO
   advantages, returns = compute_grpo_outcome_advantage(
       token_level_rewards=token_level_rewards,
       response_mask=response_mask,
       index=group_indices,  # Groups responses by prompt
       epsilon=1e-6,
       norm_adv_by_std_in_grpo=True
   )

GPRO Key Features:

- **Group-based advantage computation**: Responses to the same prompt are grouped together
- **Relative advantage normalization**: Advantages are computed relative to the group mean
- **Standard deviation scaling**: Optional scaling by group standard deviation
- **Automatic fallback**: Uses GPRO when Atropos API is unavailable

Training Loop
------------

The integration implements a complete RL training loop:

1. **Rollout Phase**: Generate responses using current policy weights
2. **Advantage Computation**: Compute GPRO advantages using VERL's implementation
3. **Training Phase**: Update policy weights using GPRO advantage-weighted SFT
4. **Weight Synchronization**: Update inference engine with new weights automatically

.. code:: python

   # Complete training loop
   for step in range(num_steps):
       # Phase 1: Rollout
       rollout_data = trainer.rollout_phase(prompts)
       
       # Phase 2: Compute GPRO advantages
       advantages = trainer.compute_advantages_from_atropos(rollout_data)
       
       # Phase 3: Training with GPRO
       training_loss = trainer.training_phase(rollout_data, advantages)
       
       # Phase 4: Weight sync (automatic via sharding manager)

Inference Engine Integration
---------------------------

The integration supports multiple inference engines:

vLLM Engine
~~~~~~~~~~

.. code:: python

   # vLLM inference engine
   from vllm import LLM, SamplingParams
   
   llm = LLM(model=model_path, trust_remote_code=True)
   sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=32)

SGLang Engine
~~~~~~~~~~~~

.. code:: python

   # SGLang inference engine
   import sglang as sgl
   
   llm = sgl.Runtime(model_path=model_path)

Weight Synchronization
---------------------

The integration uses VERL's sharding managers for automatic weight synchronization:

.. code:: python

   class AtroposShardingManager:
       def __enter__(self):
           """Sync training weights → inference engine using VERL infrastructure."""
           with self.sharding_manager:
               state_dict = self.training_model.state_dict()
               self.inference_engine.update_weights_from_tensor(state_dict)
               self.inference_engine.resume_memory_occupation()

Testing
-------

Run the comprehensive test suite:

.. code:: bash

   python recipe/atropos/test_atropos_integration.py

Test Coverage:

- ✅ GPRO integration and advantage computation
- ✅ Model loading and inference
- ✅ VERL infrastructure integration
- ✅ GPRO advantage-weighted loss computation
- ✅ Weight synchronization mechanisms
- ✅ API connectivity and error handling

Troubleshooting
--------------

Common Issues
~~~~~~~~~~~~

1. **GPRO computation errors**
   .. code:: bash
      # Check that groups have sufficient samples
      assert len(group_samples) >= 2, "GPRO requires at least 2 samples per group"

2. **vLLM/SGLang not installed**
   .. code:: bash
      pip install vllm>=0.3.0  # or sglang>=0.1.0

3. **Atropos API not accessible**
   .. code:: bash
      # Check if Atropos server is running
      curl http://localhost:9001/status

4. **CUDA out of memory**
   .. code:: bash
      # Reduce batch size
      --batch_size 2

Error Handling
~~~~~~~~~~~~~

The integration provides robust error handling:

.. code:: python

   try:
       trainer = AtroposRLTrainer(config)
   except AtroposAPIError as e:
       print(f"Atropos API error: {e}")
       # Falls back to GPRO computation
   except Exception as e:
       print(f"Unexpected error: {e}")

Performance
-----------

Memory Management
~~~~~~~~~~~~~~~~

- **Automatic memory synchronization** between training and inference
- **Efficient weight updates** via VERL's sharding managers
- **Memory-efficient inference** with proper cleanup

Scalability
~~~~~~~~~~~

- **Single-GPU training** with GPRO advantage computation
- **Batch processing** with configurable sizes
- **Production data loading** with VERL's RL dataset format

Examples
--------

Complete Training Example
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   from recipe.atropos.main_atropos import AtroposRLTrainer
   from recipe.atropos.data_loader import AtroposDataLoader

   # Configuration
   config = {
       "atropos": {"api_url": "http://localhost:9001", "timeout": 30},
       "use_advantage_weighting": True,
       "use_gpro": True,
       "gpro_epsilon": 1e-6,
       "gpro_norm_by_std": True,
       "advantage_normalization": "batch",
       "advantage_clipping": [-3.0, 3.0],
       "max_response_length": 32,
       "batch_size": 4,
       "model_path": "microsoft/DialoGPT-medium",
       "device": "cuda",
   }

   # Initialize trainer
   trainer = AtroposRLTrainer(config)

   # Load production data
   data_config = {
       "data_source": "atropos_integration",
       "max_prompts": 10,
       "prompt_format": "chat",
       "parquet_paths": ["~/data/rlhf/gsm8k/train.parquet"],
       "hf_datasets": ["gsm8k", "math"],
       "max_prompt_length": 512,
       "max_response_length": 32,
       "ability": "general",
   }

   loader = AtroposDataLoader(data_config)
   prompts = loader.load_production_prompts()

   # Training loop
   for step in range(3):
       result = trainer.rl_training_step(prompts)
       print(f"Step {result['step']}: Loss = {result['loss']:.4f}")

Custom GPRO Configuration
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # Custom GPRO parameters
   config = {
       "use_gpro": True,
       "gpro_epsilon": 1e-8,  # More conservative epsilon
       "gpro_norm_by_std": False,  # Disable std normalization
       "advantage_normalization": "global",  # Global normalization
       "advantage_clipping": [-5.0, 5.0],  # Wider clipping range
   }

   trainer = AtroposRLTrainer(config)

Advanced Usage
-------------

Custom Advantage Computation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # Custom advantage computation with GPRO
   def custom_advantage_computation(token_data, scores):
       # Convert to GPRO format
       token_level_rewards = []
       response_mask = []
       index = []
       
       for i, (tokens, token_scores) in enumerate(zip(token_data, scores)):
           response_reward = sum(token_scores) if token_scores else 0.0
           token_rewards = [response_reward / len(tokens)] * len(tokens)
           token_level_rewards.append(token_rewards)
           response_mask.append([1.0] * len(tokens))
           index.append(hash(str(tokens[:10])))
       
       # Use VERL's GPRO implementation
       advantages, _ = compute_grpo_outcome_advantage(
           token_level_rewards=torch.tensor(token_level_rewards),
           response_mask=torch.tensor(response_mask),
           index=np.array(index),
           epsilon=1e-6,
           norm_adv_by_std_in_grpo=True
       )
       
       return advantages

Integration with VERL Infrastructure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # Use VERL's model loading utilities
   from verl.utils.fs import copy_to_local
   
   local_model_path = copy_to_local(model_path, verbose=True)
   
   # Use VERL's device utilities
   from verl.utils.device import get_device_name, is_cuda_available
   
   device_name = get_device_name()
   cuda_available = is_cuda_available()

Contributing
-----------

The Atropos integration follows VERL's recipe pattern and can be extended with:

1. **Additional RL algorithms** (PPO, DPO, etc.)
2. **Custom advantage computation** methods
3. **Specialized environment integrations**
4. **Advanced weight synchronization** strategies

License
-------

This integration is part of VERL and follows the same license terms. 
