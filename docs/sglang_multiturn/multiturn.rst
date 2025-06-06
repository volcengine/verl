Multi-turn Rollout Support
==========================

Basic Configuration
~~~~~~~~~~~~~~~~~~~

To enable multi-turn rollout, make sure to configure the following fields in your rollout configuration:

.. code-block:: yaml

    actor_rollout_ref: 
        rollout: 
            multi_turn: True
            name: "sglang"

These configuration activates the sglang engine for multi-turn interaction during rollout.

Custom Tool Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

For custom environment interaction tools, you can implement your own tools based on ``verl.tools.base_tool.BaseTool``. Then, specify your tool configurations in a YAML file:

.. code-block:: yaml

    tools:
      - class_name: ""
        config: {}
        tool_schema:

You may refer to GSM8KTool_example_configuration_, which is one example of the tool configurations. Its implementation can be found in gsm8k_tool.py_.

Finally, set the ``tools_config_file`` in your rollout config:

.. code-block:: yaml

    actor_rollout_ref:
        rollout:
            tool_kwargs:
                tools_config_file: <path_to_tool_yaml_file>

This allows integration of customized tool behaviors during actor rollout steps. 

GSM8K Multi-turn Training Performance  
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See the training performance of multi-turn rollout on the GSM8K task HERE_.

.. _HERE: https://wandb.ai/zhaochenyang20/gsm8k_async_rl/runs/1ro1r7om?nw=nwuserzhaochenyang20

.. _GSM8KTool_example_configuration: https://github.com/volcengine/verl/blob/main/examples/sglang_multiturn/config/tool_config/gsm8k_tool_config.yaml

.. _gsm8k_tool.py: https://github.com/volcengine/verl/blob/main/verl/tools/gsm8k_tool.py

Search Tool Integration
~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1

   search_tool_example