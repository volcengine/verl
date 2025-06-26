.. _atropos-quick-reference:

Atropos Integration Quick Reference
==================================

Quick Start Commands
-------------------

.. code:: bash

   # Demo mode (single GPU)
   python recipe/atropos/main_atropos.py

   # Training mode
   python recipe/atropos/launch_atropos_verl.py --mode training

   # Test integration
   python recipe/atropos/test_atropos_integration.py

Common Configurations
--------------------

Basic Training Config
~~~~~~~~~~~~~~~~~~~~

.. code:: yaml

   # Minimal config for single-GPU training
   atropos:
     api_url: "http://localhost:9001"
     timeout: 30
   
   use_gpro: true
   use_advantage_weighting: true
   batch_size: 4
   max_response_length: 32
   model_path: "microsoft/DialoGPT-medium"
   device: "cuda"

Production Config
~~~~~~~~~~~~~~~~

.. code:: yaml

   # Production config with GPRO
   atropos:
     api_url: "http://localhost:9001"
     timeout: 30
   
   use_gpro: true
   gpro_epsilon: 1e-6
   gpro_norm_by_std: true
   use_advantage_weighting: true
   advantage_normalization: "batch"
   advantage_clipping: [-3.0, 3.0]
   
   batch_size: 8
   max_response_length: 64
   model_path: "microsoft/DialoGPT-medium"
   device: "cuda"
   
   data:
     data_source: "atropos_integration"
     max_prompts: 100
     prompt_format: "chat"
     parquet_paths: ["~/data/rlhf/gsm8k/train.parquet"]
     hf_datasets: ["gsm8k", "math"]
     max_prompt_length: 512
     max_response_length: 64
     ability: "general"

API Usage Patterns
-----------------

Basic Training Loop
~~~~~~~~~~~~~~~~~~

.. code:: python

   from recipe.atropos.main_atropos import AtroposRLTrainer
   
   config = {
       "atropos": {"api_url": "http://localhost:9001", "timeout": 30},
       "use_gpro": True,
       "use_advantage_weighting": True,
       "batch_size": 4,
       "max_response_length": 32,
       "model_path": "microsoft/DialoGPT-medium",
       "device": "cuda",
   }
   
   trainer = AtroposRLTrainer(config)
   prompts = ["What is 2+2?", "Explain quantum computing"]
   
   for step in range(10):
       result = trainer.rl_training_step(prompts)
       print(f"Step {step}: Loss = {result['loss']:.4f}")

Custom Advantage Computation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # Custom GPRO parameters
   config = {
       "use_gpro": True,
       "gpro_epsilon": 1e-8,
       "gpro_norm_by_std": False,
       "advantage_normalization": "global",
       "advantage_clipping": [-5.0, 5.0],
   }
   
   trainer = AtroposRLTrainer(config)

Data Loading
~~~~~~~~~~~

.. code:: python

   from recipe.atropos.data_loader import AtroposDataLoader
   
   data_config = {
       "data_source": "atropos_integration",
       "max_prompts": 50,
       "prompt_format": "chat",
       "parquet_paths": ["~/data/rlhf/gsm8k/train.parquet"],
       "hf_datasets": ["gsm8k", "math"],
       "max_prompt_length": 512,
       "max_response_length": 32,
       "ability": "general",
   }
   
   loader = AtroposDataLoader(data_config)
   prompts = loader.load_production_prompts()

Error Handling
--------------

.. code:: python

   try:
       trainer = AtroposRLTrainer(config)
       result = trainer.rl_training_step(prompts)
   except AtroposAPIError as e:
       print(f"Atropos API error: {e}")
       # Falls back to GPRO computation automatically
   except Exception as e:
       print(f"Unexpected error: {e}")

Troubleshooting
---------------

Common Issues and Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Atropos API not accessible**
   .. code:: bash
      # Check if server is running
      curl http://localhost:9001/status
      # Falls back to GPRO automatically

2. **CUDA out of memory**
   .. code:: bash
      # Reduce batch size
      --batch_size 2
      # Or reduce max_response_length
      --max_response_length 16

3. **GPRO computation errors**
   .. code:: bash
      # Ensure sufficient samples per group
      --batch_size 4  # At least 2 samples per prompt

4. **Model loading issues**
   .. code:: bash
      # Check model path
      ls ~/models/microsoft/DialoGPT-medium
      # Use smaller model for testing
      --model_path microsoft/DialoGPT-small

Performance Tuning
------------------

Memory Optimization
~~~~~~~~~~~~~~~~~~

.. code:: yaml

   # Memory-efficient config
   batch_size: 2
   max_response_length: 16
   max_prompt_length: 256
   gpu_memory_utilization: 0.5

Speed Optimization
~~~~~~~~~~~~~~~~~

.. code:: yaml

   # Speed-optimized config
   batch_size: 8
   max_response_length: 32
   use_torch_compile: true
   enforce_eager: false

Integration Examples
-------------------

With VERL Infrastructure
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # Use VERL utilities
   from verl.utils.fs import copy_to_local
   from verl.utils.device import get_device_name
   
   local_model_path = copy_to_local(model_path, verbose=True)
   device_name = get_device_name()

With Custom Models
~~~~~~~~~~~~~~~~~

.. code:: python

   # Custom model configuration
   config = {
       "model_path": "your/custom/model",
       "trust_remote_code": True,
       "use_remove_padding": True,
   }

With Custom Datasets
~~~~~~~~~~~~~~~~~~~

.. code:: python

   # Custom dataset configuration
   data_config = {
       "data_source": "atropos_integration",
       "custom_dataset_path": "path/to/your/dataset.parquet",
       "prompt_key": "question",
       "response_key": "answer",
   } 