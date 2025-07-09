veRL Trace Function Usage Instructions
========================================

Applicable Scenarios
--------------------

Agentic RL involves multiple turns of conversations, tool invocations, and user interactions during the rollout process. During the Model Training process, it is necessary to track function calls, inputs, and outputs to understand the flow path of data within the application. The Trace feature helps, in complex multi-round conversations, to view the transformation of data during each interaction and the entire process leading to the final output by recording the inputs, outputs, and corresponding timestamps of functions, which is conducive to understanding the details of how the model processes data and optimizing the training results.

The veRL Trace feature integrates commonly used Agent trace tools, including wandb weave and mlflow, which are already supported. Users can choose the appropriate trace tool according to their own needs and preferences. Here, we introduce the usage of each tool.


Trace Parameter Configuration
-----------------------------

- ``trainer.rollout_trace.backend=mlflow|weave`` # the trace backend type
- ``trainer.rollout_trace.token2text=True`` # To show decoded text in trace view


Glossary
--------

+----------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Object         | Explaination                                                                                                                                                         |
+================+======================================================================================================================================================================+
| trajectory     | A complete multi-turn conversation includes:                                                                                                                         |
|                | 1. LLM output at least once                                                                                                                                          |
|                | 2. Tool Call                                                                                                                                                         |
+----------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| step           | The training step corresponds to the global_steps variable in the trainer                                                                                             |
+----------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| sample_index   | The identifier of the sample, defined in the extra_info.index of the dataset, is usually a number, but there are also cases where a uuid is used.                      |
+----------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| rollout_n      | In the GROP algorithm, each sample is rolled out n times, and rollout_n represents the number of the rollout.                                                          |
+----------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| validate       | Is it using the test dataset for evaluation?                                                                                                                          |
+----------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------+


Usage of wandb weave
--------------------

1.1 Basic Configuration
~~~~~~~~~~~~~~~~~~~~~~~

1. Set the ``WANDB_API_KEY`` environment variable
2. veRL Configuration Parameters

   1. ``trainer.rollout_trace.backend=weave``
   2. ``trainer.logger=['console', 'wandb']``. This item is optional. Trace and logger are independent functions. When using Weave, it is recommended to also enable the wandb logger to implement both functions in one system.
   3. ``trainer.project_name=$project_name``
   4. ``trainer.experiment_name=$experiment_name``

Note:
The Weave Free Plan comes with a default monthly network traffic allowance of 1GB. During the training process, the amount of trace data generated is substantial, reaching dozens of gigabytes per day, so it is necessary to select an appropriate wandb plan.


1.2 View Trace Logs
~~~~~~~~~~~~~~~~~~~

After executing the training, on the project page, you can see the WEAVE sidebar. Click Traces to view it.

Each Trace project corresponds to a trajectory. You can filter and select the trajectories you need to view by step, sample_index, rollout_n, and experiment_name.

After enabling token2text, prompt_text and response_text will be automatically added to the output of ToolAgentLoop.run, making it convenient to view the input and output content.

.. image:: https://private-user-images.githubusercontent.com/4373761/461954072-ff30bbca-f9c8-434f-a3c2-0e333d16fa68.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTIwNTc3MjksIm5iZiI6MTc1MjA1NzQyOSwicGF0aCI6Ii80MzczNzYxLzQ2MTk1NDA3Mi1mZjMwYmJjYS1mOWM4LTQzNGYtYTNjMi0wZTMzM2QxNmZhNjgucG5nP1gtQW16LUFsZ29yaXRobT1BV1M0LUhNQUMtU0hBMjU2JlgtQW16LUNyZWRlbnRpYWw9QUtJQVZDT0RZTFNBNTNQUUs0WkElMkYyMDI1MDcwOSUyRnVzLWVhc3QtMSUyRnMzJTJGYXdzNF9yZXF1ZXN0JlgtQW16LURhdGU9MjAyNTA3MDlUMTAzNzA5WiZYLUFtei1FeHBpcmVzPTMwMCZYLUFtei1TaWduYXR1cmU9ZTI1MGIzZjEwZTFmMDRjMTU0ODEyOTFkMWM5YjU3MzM2YzYzMjcyZmM4NGE1ZDY3NjMzYjZjNTAyN2JmZmMxOSZYLUFtei1TaWduZWRIZWFkZXJzPWhvc3QifQ.ydn_hzBMQZs_9lozSvwhGuy1dXRaxqjBDTxkvHFZAHw

1.3 Compare Trace Logs
~~~~~~~~~~~~~~~~~~~~~~

Weave can select multiple trace items and then compare the differences among them.

.. image:: https://private-user-images.githubusercontent.com/4373761/461954031-0b9ed8db-58a7-4769-88fb-bda204dc9fc8.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTIwNTc3MjksIm5iZiI6MTc1MjA1NzQyOSwicGF0aCI6Ii80MzczNzYxLzQ2MTk1NDAzMS0wYjllZDhkYi01OGE3LTQ3NjktODhmYi1iZGEyMDRkYzlmYzgucG5nP1gtQW16LUFsZ29yaXRobT1BV1M0LUhNQUMtU0hBMjU2JlgtQW16LUNyZWRlbnRpYWw9QUtJQVZDT0RZTFNBNTNQUUs0WkElMkYyMDI1MDcwOSUyRnVzLWVhc3QtMSUyRnMzJTJGYXdzNF9yZXF1ZXN0JlgtQW16LURhdGU9MjAyNTA3MDlUMTAzNzA5WiZYLUFtei1FeHBpcmVzPTMwMCZYLUFtei1TaWduYXR1cmU9MDg3NTAzYWZlOTM2NmY3ZmI2ZGFjOTY0ODFiNzRmMmUxNDg1ZWZjNjU0NWQwYjg5MTZjMjY5NzllZDRiNjEwNCZYLUFtei1TaWduZWRIZWFkZXJzPWhvc3QifQ.mlpKmPGUkKJOgFJXLlF8dwFzdcptpjIKwHcZMBgRB5k


Usage of mlflow
---------------

1. Basic Configuration
~~~~~~~~~~~~~~~~~~~~~~

1. Set the ``MLFLOW_TRACKING_URI`` environment variable, which can be:

   1. Http and https URLs corresponding to online services
   2. Local files or directories, such as ``sqlite:////tmp/mlruns.db``, indicate that data is stored in ``/tmp/mlruns.db``. When using local files, it is necessary to initialize the file first (e.g., start the UI: ``mlflow ui --backend-store-uri sqlite:////tmp/mlruns.db``) to avoid conflicts when multiple workers create files simultaneously.

2. veRL Configuration Parameters

   1. ``trainer.rollout_trace.backend=mlflow``
   2. ``trainer.logger=['console', 'mlflow']``. This item is optional. Trace and logger are independent functions. When using mlflow, it is recommended to also enable the mlflow logger to implement both functions in one system.
   3. ``trainer.project_name=$project_name``
   4. ``trainer.experiment_name=$experiment_name``


2. View Log
~~~~~~~~~~~

Since ``trainer.project_name`` corresponds to Experiments in mlflow, in the mlflow view, you need to select the corresponding project name, then click the "Traces" tab to view traces. Among them, ``trainer.experiment_name`` corresponds to the experiment_name of tags, and tags corresponding to step, sample_index, rollout_n, etc., are used for filtering and viewing.

For example, searching for ``"tags.step = '1'"`` can display all trajectories of step 1.

.. image:: https://private-user-images.githubusercontent.com/4373761/464135842-38d11bf2-5c43-480c-88db-19e0c8443e74.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTIwNTc5NDEsIm5iZiI6MTc1MjA1NzY0MSwicGF0aCI6Ii80MzczNzYxLzQ2NDEzNTg0Mi0zOGQxMWJmMi01YzQzLTQ4MGMtODhkYi0xOWUwYzg0NDNlNzQucG5nP1gtQW16LUFsZ29yaXRobT1BV1M0LUhNQUMtU0hBMjU2JlgtQW16LUNyZWRlbnRpYWw9QUtJQVZDT0RZTFNBNTNQUUs0WkElMkYyMDI1MDcwOSUyRnVzLWVhc3QtMSUyRnMzJTJGYXdzNF9yZXF1ZXN0JlgtQW16LURhdGU9MjAyNTA3MDlUMTA0MDQxWiZYLUFtei1FeHBpcmVzPTMwMCZYLUFtei1TaWduYXR1cmU9MzI1Nzc1ZWM0MzUzZGViZGI2ZmNiNzM2N2E0MmZjMmZkMzViZjhlMjZlZGM0MmNlZDBkZmRlMTM3ZDRhNWJiNSZYLUFtei1TaWduZWRIZWFkZXJzPWhvc3QifQ.Z3dkL00_7kf6bSutlGar24nKfYZuwrp01ivKr9iRwAc

Opening one of the trajectories allows you to view each function call process within it.

After enabling token2text, prompt_text and response_text will be automatically added to the output of ToolAgentLoop.run, making it convenient to view the content.

.. image:: https://private-user-images.githubusercontent.com/4373761/464135865-7f486e4a-59bb-46b0-a32d-5e8d0d08759b.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTIwNTc5NDEsIm5iZiI6MTc1MjA1NzY0MSwicGF0aCI6Ii80MzczNzYxLzQ2NDEzNTg2NS03ZjQ4NmU0YS01OWJiLTQ2YjAtYTMyZC01ZThkMGQwODc1OWIucG5nP1gtQW16LUFsZ29yaXRobT1BV1M0LUhNQUMtU0hBMjU2JlgtQW16LUNyZWRlbnRpYWw9QUtJQVZDT0RZTFNBNTNQUUs0WkElMkYyMDI1MDcwOSUyRnVzLWVhc3QtMSUyRnMzJTJGYXdzNF9yZXF1ZXN0JlgtQW16LURhdGU9MjAyNTA3MDlUMTA0MDQxWiZYLUFtei1FeHBpcmVzPTMwMCZYLUFtei1TaWduYXR1cmU9ZmJkMmZiOGZhMTEzNjY1ZjM4MWE5YzM3ODQ0MjIzOTdlYWQzYjMwNjI4YWRhZDA1NjRjN2M4ZTNkYmJkZjBkYiZYLUFtei1TaWduZWRIZWFkZXJzPWhvc3QifQ.80MjKBgx9BxY62R9WpSCGduyjrgbvhfYV_zSJ_sqA7M

Note:
1. mlflow does not support comparing multiple traces
2. veRL failed to associate the trace with the run, so the trace content cannot be seen in the mlflow run logs.