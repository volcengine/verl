Multi-turn Rollout Support
=========================

Basic Configuration
~~~~~~~~~~~~~~~~~

To enable multi-turn rollout, make sure to configure the following fields in your rollout configuration:

.. code-block:: yaml

    actor_rollout_ref: 
        rollout: 
            multi_turn: True
            name: "sglang_async"

These configuration activates the sglang_async engine for multi-turn interaction during rollout.

Custom Tool Configuration
~~~~~~~~~~~~~~~~~~~~~~~

For custom environment interaction tools, you can specify your tool configurations in a YAML file.  
To do so, use the following format in your rollout config:

.. code-block:: yaml

    actor_rollout_ref:
        rollout:
            tool_kwargs:
                tools_config_file: <path_to_tool_yaml_file>

This allows integration of customized tool behaviors during actor rollout steps. You may refer to the GSM8KTool_example_configuration_ for guidance.

GSM8K Multi-turn Training Performance  
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See the training performance of multi-turn rollout on the GSM8K task HERE_.

.. _HERE: https://wandb.ai/zhaochenyang20/gsm8k_async_rl/runs/1ro1r7om?nw=nwuserzhaochenyang20

.. _GSM8KTool_example_configuration: ../../examples/sglang_multiturn/config/tool_config/gsm8k_tool_config.yaml