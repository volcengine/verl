.. _attention-implementation-override:

Attention Implementation Override
==================================

Last updated: 11/05/2025.

By default, VERL auto-detects the attention backend (prefers ``flash_attention_3`` > ``flash_attention_2`` > ``flex_attention`` > ``sdpa``) unless overridden.
You can override this setting when needed.

Supported Attention Implementations
-----------------------------------

The following attention implementations are supported (subject to model and hardware compatibility):

- ``flash_attention_3``
- ``flash_attention_2``
- ``flex_attention``
- ``sdpa``
- ``eager``
- ``flashinfer`` (SGLang/vLLM only)

Where to set:
- Per-role: ``actor.attn_implementation``, ``rollout.attn_implementation``, ``ref.attn_implementation``, ``critic.attn_implementation``, ``reward_model.attn_implementation``
- Via environment: ``VERL_ATTN_IMPLEMENTATION``

When to Override
----------------

You might want to override the attention implementation in the following scenarios:

- **Debugging**: Use ``eager`` for easier debugging and better error messages
- **Compatibility**: Some models or hardware configurations may not support ``flash_attention_2``
- **Memory constraints**: Different implementations have different memory characteristics
- **Performance tuning**: Testing different implementations for optimal performance

Configuration Examples
-----------------------

PPO Training with Eager Attention (per-role)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

    python3 ppo_trainer.py \
        +actor_rollout_ref.actor.attn_implementation=eager \
        +actor_rollout_ref.rollout.attn_implementation=eager \
        +actor_rollout_ref.ref.attn_implementation=eager \
        [other parameters...]

PPO Training with SDPA Attention (actor only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

    python3 ppo_trainer.py \
        +actor_rollout_ref.actor.attn_implementation=sdpa \
        [other parameters...]

Critic Model Override
~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

    python3 ppo_trainer.py \
        +critic.attn_implementation=eager \
        [other parameters...]

YAML Configuration (defaults to auto per-role)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: yaml

    actor_rollout_ref:
      actor:
        attn_implementation: auto
      rollout:
        # defaults to actor's attn_implementation unless specified
        attn_implementation: ${oc.select:actor_rollout_ref.actor.attn_implementation,auto}
      ref:
        # defaults to actor's attn_implementation unless specified
        attn_implementation: ${oc.select:actor_rollout_ref.actor.attn_implementation,auto}

    critic:
      attn_implementation: ${oc.select:actor_rollout_ref.actor.attn_implementation,auto}

Role-specific overrides
~~~~~~~~~~~~~~~~~~~~~~~

When your training and inference engines need different backends, set per-role keys. Precedence:
``<role>.attn_implementation`` > ``VERL_ATTN_IMPLEMENTATION`` > auto.

.. code:: yaml

    actor_rollout_ref:
      # Prefer FA3 for the HF training model
      actor:
        attn_implementation: flash_attention_3
      # Prefer FlashInfer for SGLang/vLLM rollout
      rollout:
        attn_implementation: flashinfer
      # Use SDPA for the reference policy (HF)
      ref:
        attn_implementation: sdpa

    critic:
      # Critic can use its own backend
      attn_implementation: eager

    reward_model:
      # Reward model can use its own backend
      attn_implementation: eager

Important Notes
---------------

**Defaults**: All roles default to ``auto``. Use per-role keys to override explicitly.

**Backward Compatibility**: If you don't specify ``attn_implementation`` (or ``VERL_ATTN_IMPLEMENTATION``),
VERL will auto-detect and select a supported backend (preferring ``flash_attention_3`` > ``flash_attention_2`` > ``flex_attention`` > ``sdpa``).

**Model Support**: Not all models support all attention implementations. Ensure your model is compatible 
with the chosen attention implementation before training.

**Performance Impact**: Different attention implementations have varying performance characteristics. 
``flash_attention_2`` typically offers the best performance, while ``eager`` provides better debugging capabilities.

**Hardware Dependencies**: Some attention implementations (like ``flash_attention_2``) may require 
specific hardware or CUDA versions. If you encounter compatibility issues, try using ``eager`` or ``sdpa``.

Troubleshooting
---------------

If you encounter errors when using a specific attention implementation:

1. **Check model compatibility**: Verify that your model supports the chosen attention implementation
2. **Try eager attention**: Use ``attn_implementation=eager`` as a fallback for debugging
3. **Check hardware requirements**: Ensure your hardware supports the attention implementation
4. **Review error messages**: Attention implementation errors often provide clear guidance on supported options

Example Error Resolution
~~~~~~~~~~~~~~~~~~~~~~~~

If you see an error like "flash_attention_2 is not supported", you can resolve it by switching to eager attention:

.. code:: bash

    python3 ppo_trainer.py +actor_rollout_ref.actor.attn_implementation=eager

This override ensures your training can proceed while you investigate the flash attention compatibility issue.
