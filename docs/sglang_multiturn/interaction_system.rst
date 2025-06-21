Interaction System for Multi-turn RL Training
=============================================

Overview
--------

The verl interaction system enables dynamic, multi-turn conversational feedback during reinforcement learning training. This system allows models to engage in iterative problem-solving scenarios where an interaction agent can provide corrective feedback, guidance, or evaluation based on the model's responses.

Key features:

- **Async-based Architecture**: Non-blocking interaction processing for distributed training
- **Instance Management**: Stateful session handling with unique instance IDs for concurrent interactions
- **SGLang Integration**: Seamless integration with SGLang rollout system for multi-turn conversations
- **Configuration-driven**: Dynamic agent loading via YAML configuration files
- **Reward Integration**: Turn-level scoring mechanism integrated with verl's reward system

Architecture
------------

The interaction system follows a plugin-based architecture with clear separation of concerns:

.. code-block::

    BaseInteraction (Abstract Interface)
         ↓
    Gsm8kInteraction (Concrete Implementation)
         ↓
    SGLang Rollout Integration
         ↓
    Async Request Lifecycle Management

Core Components
~~~~~~~~~~~~~~~

**BaseInteraction Interface**

All interaction agents must implement the ``BaseInteraction`` abstract class:

.. code-block:: python

    from verl.interactions.base import BaseInteraction
    from typing import Dict, Any, List, Tuple, Optional

    class BaseInteraction:
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

Create an interaction configuration file (e.g., ``gsm8k_interaction_config.yaml``):

.. code-block:: yaml

    interaction:
      - class_name: "verl.interactions.gsm8k_interaction.Gsm8kInteraction"
        config: {}

The system will dynamically load the specified interaction class using importlib.

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

Ensure your dataset includes interaction parameters:

.. code-block:: python

    # Dataset should include interaction_kwargs in non_tensor_batch
    interaction_kwargs = [
        {"query": "What is 2+2?", "ground_truth": "4"},
        {"query": "What is 3+3?", "ground_truth": "6"},
    ]

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