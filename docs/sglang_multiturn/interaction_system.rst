Interaction System for Multi-turn RL Training
=============================================

Last updated: 06/25/2025.

Overview
--------

The verl interaction system enables dynamic, multi-turn conversational feedback during reinforcement learning training. This system allows models to engage in iterative problem-solving scenarios where interaction agents can provide corrective feedback, guidance, or evaluation based on the model's responses.

**New in Multi-Interaction Support**: The system now supports multiple named interactions within a single training session, enabling sophisticated training scenarios where different samples can use different interaction strategies. This allows for curriculum learning, domain-specific feedback, and flexible agent switching at the sample level.

Key features:

- **Async-based Architecture**: Non-blocking interaction processing for distributed training
- **Instance Management**: Stateful session handling with unique instance IDs for concurrent interactions
- **SGLang Integration**: Seamless integration with SGLang rollout system for multi-turn conversations
- **Configuration-driven**: Dynamic agent loading via YAML configuration files
- **Multi-Interaction Support**: Registry system enabling multiple named interactions per rollout
- **Sample-Level Selection**: Each sample can specify which interaction to use via configuration
- **Reward Integration**: Turn-level scoring mechanism integrated with verl's reward system

Architecture
------------

The interaction system follows a plugin-based architecture with clear separation of concerns:

.. code-block::

    Interaction Registry System
         ↓
    BaseInteraction (Abstract Interface)
         ↓
    Multiple Named Interactions (e.g., Gsm8kInteraction, CustomInteraction)
         ↓
    SGLang Rollout Integration (interaction_map)
         ↓
    Sample-Level Interaction Selection
         ↓
    Async Request Lifecycle Management

Core Components
~~~~~~~~~~~~~~~

**Interaction Registry System**

The interaction registry system allows loading and managing multiple named interactions:

.. code-block:: python

    from verl.interactions.utils.interaction_registry import initialize_interactions_from_config
    
    # Load multiple interactions from config
    interaction_map = initialize_interactions_from_config("config.yaml")
    
    # Access specific interaction by name
    gsm8k_interaction = interaction_map["gsm8k"]
    custom_interaction = interaction_map["custom_solver"]

**BaseInteraction Interface**

All interaction agents must implement the ``BaseInteraction`` abstract class:

.. code-block:: python

    from verl.interactions.base import BaseInteraction
    from typing import Dict, Any, List, Tuple, Optional

    class BaseInteraction:
        def __init__(self, config: Dict[str, Any]):
            self.config = config
            self.name: str = config.get("name", "interaction_agent")
        
        async def start_interaction(self, instance_id: Optional[str] = None, **kwargs) -> str:
            """Initialize interaction session, return instance_id"""
            
        async def generate_response(self, instance_id: str, messages: List[Dict[str, Any]], **kwargs) -> Tuple[bool, str, float, Dict[str, Any]]:
            """Generate response, return (should_terminate, response, score, metadata)"""
            
        async def calculate_score(self, instance_id: str, **kwargs) -> float:
            """Calculate turn-level score for RL training"""
            
        async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
            """Clean up resources"""

**Request Lifecycle**

The interaction system integrates with SGLang's async rollout via state management:

1. ``PENDING`` → Initialize interaction via ``start_interaction()``
2. ``GENERATING`` → Model generates response
3. ``INTERACTING`` → Process response via ``generate_response()``
4. ``GENERATING`` → Continue if not terminated, otherwise ``COMPLETED``

Configuration
-------------

**Basic Setup**

Enable interaction in your rollout configuration:

.. code-block:: yaml

    actor_rollout_ref:
        rollout:
            multi_turn:
                enable: true
                interaction_config_path: "path/to/interaction_config.yaml"
                max_user_turns: 10
                max_assistant_turns: 10

**Interaction Configuration File**

Create an interaction configuration file (e.g., ``interaction_config.yaml``):

**Single Interaction (Legacy Format)**

.. code-block:: yaml

    interaction:
      - name: "gsm8k"
        class_name: "verl.interactions.gsm8k_interaction.Gsm8kInteraction"
        config: {}

**Multiple Interactions (New Format)**

.. code-block:: yaml

    interaction:
      - name: "gsm8k"
        class_name: "verl.interactions.gsm8k_interaction.Gsm8kInteraction"
        config: {}
      - name: "custom_solver"
        class_name: "custom.interactions.CustomInteraction"
        config: 
          solver_type: "advanced"
          timeout: 30
      - name: "code_verifier"
        class_name: "verl.interactions.base.BaseInteraction"
        config: 
          verification_mode: "strict"

**Automatic Name Generation**

If no ``name`` field is provided, the system will automatically generate one from the class name:

.. code-block:: yaml

    interaction:
      - class_name: "verl.interactions.gsm8k_interaction.Gsm8kInteraction"
        config: {}
        # Automatically generates name: "gsm8k"

The system will dynamically load all specified interaction classes and make them available by name.

Implementation Example: GSM8K
-----------------------------

The GSM8K interaction demonstrates a complete implementation for math problem-solving scenarios:

.. code-block:: python

    from verl.interactions.base import BaseInteraction
    from verl.utils.reward_score import gsm8k
    from uuid import uuid4

    class Gsm8kInteraction(BaseInteraction):
        def __init__(self, config: dict):
            super().__init__(config)
            self._instance_dict = {}

        async def start_interaction(self, instance_id=None, ground_truth=None, **kwargs):
            if instance_id is None:
                instance_id = str(uuid4())
            self._instance_dict[instance_id] = {
                "response": "",
                "ground_truth": ground_truth,
                "reward": 0.0,
            }
            return instance_id

        async def generate_response(self, instance_id, messages, **kwargs):
            # Extract last user message content
            content = ""
            for item in reversed(messages):
                if item.get("role") == "user":
                    content = item.get("content", "")
                    break

            # Ensure GSM8K format (#### prefix)
            if content.startswith("#### "):
                self._instance_dict[instance_id]["response"] = content
            else:
                self._instance_dict[instance_id]["response"] = "#### " + content

            reward = await self.calculate_score(instance_id)
            if reward == 1.0:
                return True, "Your response is correct!", 1.0, {}
            else:
                return False, "Your response is incorrect! You need to reflect on your answer and try again.", 0.0, {}

        async def calculate_score(self, instance_id, **kwargs):
            return gsm8k.compute_score(
                self._instance_dict[instance_id]["response"],
                self._instance_dict[instance_id]["ground_truth"],
                method="flexible", format_score=0.0, score=1.0,
            )

        async def finalize_interaction(self, instance_id, **kwargs):
            del self._instance_dict[instance_id]

Training Integration
--------------------

**Training Script Configuration**

Include interaction configuration in your training command:

.. code-block:: bash

    python3 -m verl.trainer.main_ppo \\
        --config-path="$CONFIG_PATH" \\
        --config-name='gsm8k_multiturn_grpo_w_interaction' \\
        algorithm.adv_estimator=grpo \\
        data.train_batch_size=512 \\
        data.return_raw_chat=True \\
        actor_rollout_ref.rollout.name=sglang \\
        actor_rollout_ref.rollout.multi_turn.interaction_config_path="$PROJECT_DIR/examples/sglang_multiturn/config/interaction_config/gsm8k_interaction_config.yaml" \\
        trainer.total_epochs=15

**Data Requirements**

Ensure your dataset includes interaction parameters with the ``name`` field for interaction selection:

.. code-block:: python

    # Dataset should include interaction_kwargs in non_tensor_batch
    interaction_kwargs = [
        {"name": "gsm8k", "query": "What is 2+2?", "ground_truth": "4"},
        {"name": "custom_solver", "query": "Solve: x^2 + 5x + 6 = 0", "ground_truth": "x = -2, -3"},
        {"name": "gsm8k", "query": "What is 3+3?", "ground_truth": "6"},
    ]

**Sample-Level Interaction Selection**

Each sample can specify which interaction to use via the ``name`` field. This enables flexible training scenarios where different samples use different interaction strategies:

.. code-block:: python

    # Example: Math problems use GSM8K interaction, code problems use code verifier
    data_samples = [
        {
            "prompt": "What is 15% of 200?",
            "interaction_kwargs": {
                "name": "gsm8k",
                "query": "What is 15% of 200?", 
                "ground_truth": "30"
            }
        },
        {
            "prompt": "Write a function to check if a number is prime",
            "interaction_kwargs": {
                "name": "code_verifier",
                "code_type": "python",
                "expected_behavior": "return True for prime numbers"
            }
        }
    ]

**Backward Compatibility**

If no ``name`` field is provided in ``interaction_kwargs``, the system defaults to ``"gsm8k"`` for backward compatibility.

Best Practices
--------------

**Resource Management**

- Always implement proper cleanup in ``finalize_interaction()``
- Use unique instance IDs to avoid conflicts in concurrent training
- Handle edge cases like empty messages or malformed content

**Performance Optimization**

- Keep interaction logic lightweight to avoid blocking training
- Use async/await properly to maintain non-blocking behavior
- Consider caching expensive computations within interaction instances

**Testing**

Comprehensive testing is essential for interaction systems:

.. code-block:: python

    import pytest
    from unittest.mock import patch

    @pytest.mark.asyncio
    async def test_interaction_workflow():
        interaction = YourInteraction({})
        
        # Test complete workflow
        instance_id = await interaction.start_interaction(ground_truth="expected_answer")
        
        messages = [{"role": "user", "content": "user_response"}]
        should_terminate, response, reward, metadata = await interaction.generate_response(instance_id, messages)
        
        assert should_terminate in [True, False]
        assert isinstance(reward, float)
        
        await interaction.finalize_interaction(instance_id)

Advanced Usage
--------------

**Multi-Interaction Training Strategies**

You can design sophisticated training scenarios using multiple interactions:

.. code-block:: python

    # Example: Progressive difficulty with different interaction agents
    class MathTrainingPipeline:
        def create_interaction_config(self):
            return {
                "interaction": [
                    {
                        "name": "basic_math",
                        "class_name": "verl.interactions.gsm8k_interaction.Gsm8kInteraction",
                        "config": {"difficulty": "easy"}
                    },
                    {
                        "name": "advanced_math", 
                        "class_name": "custom.interactions.AdvancedMathInteraction",
                        "config": {"difficulty": "hard", "allow_hints": True}
                    },
                    {
                        "name": "competition_math",
                        "class_name": "custom.interactions.CompetitionMathInteraction", 
                        "config": {"time_limit": 300, "show_steps": False}
                    }
                ]
            }
    
        def create_curriculum_data(self, epoch):
            if epoch < 5:
                return [{"name": "basic_math", ...} for _ in samples]
            elif epoch < 10:
                return [{"name": "advanced_math", ...} for _ in samples]
            else:
                return [{"name": "competition_math", ...} for _ in samples]

**Custom Scoring Functions**

You can integrate custom reward functions:

.. code-block:: python

    async def calculate_score(self, instance_id, **kwargs):
        response = self._instance_dict[instance_id]["response"]
        ground_truth = self._instance_dict[instance_id]["ground_truth"]
        
        # Custom evaluation logic
        if custom_evaluation_function(response, ground_truth):
            return 1.0
        else:
            return 0.0

**Multi-step Interactions**

For complex scenarios requiring multiple feedback rounds:

.. code-block:: python

    async def generate_response(self, instance_id, messages, **kwargs):
        instance = self._instance_dict[instance_id]
        instance["attempts"] += 1
        
        # Evaluate current response
        reward = await self.calculate_score(instance_id)
        
        if reward > 0.8:
            return True, "Excellent work!", reward, {}
        elif instance["attempts"] < 3:
            return False, "Good attempt, but try to improve...", reward, {}
        else:
            return True, "Maximum attempts reached.", reward, {}

Troubleshooting
---------------

**Common Issues**

1. **Instance ID Conflicts**: Ensure unique instance IDs across concurrent sessions
2. **Memory Leaks**: Always call ``finalize_interaction()`` to clean up resources
3. **Blocking Operations**: Keep interaction logic async and non-blocking
4. **Configuration Errors**: Verify interaction config path and class name are correct
5. **Interaction Name Conflicts**: Ensure all interactions have unique names in the configuration
6. **Missing Interaction**: Verify the ``name`` field in ``interaction_kwargs`` matches available interactions
7. **Backward Compatibility**: When migrating from single to multi-interaction, add ``name`` fields to existing data

**Debugging**

Enable debug logging to trace interaction flow:

.. code-block:: bash

    export VERL_LOGGING_LEVEL=DEBUG

**Performance Monitoring**

Monitor interaction performance impact on training throughput and adjust accordingly.

Related Documentation
--------------------

- :doc:`multiturn`: Basic multi-turn rollout configuration
- :doc:`sandbox_fusion`: Tool integration with SGLang
- :doc:`search_tool_example`: Search tool implementation example