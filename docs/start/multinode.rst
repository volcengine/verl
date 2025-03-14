Multinode Training
==================

.. _wuxibin89: https://github.com/wuxibin89

Author: `Xibin Wu <https://github.com/wuxibin89>`_

Manual
------

Set up multinode ray cluster
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. Start head node with ``ray start --head --dashboard-host=0.0.0.0``, there're 2 address you should care about:

- GCS address: ``ray start --address=<address>``, where worker node should connect to.
- Dashboard address: ``<address>:8265``, where you should submit job to the cluster.

.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/ray/head.png?raw=true

2. Start worker node with ``ray start --address=<address>`` you get above.

.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/ray/worker.png?raw=true

3. Now you should see the cluster have 2 nodes with ``ray status``.

.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/ray/status.png?raw=true

4. Additionally, you can access dashboard in the browser with the address you get above. 

*Firewall rules maybe need configure to access the dashboard, if there's any trouble, please contact your network administrator.*

.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/ray/overview.png?raw=true

Submit job to ray cluster
~~~~~~~~~~~~~~~~~~~~~~~~~
1. Submit ray job to cluster with the dashboard address you get above.

.. code-block:: bash

    ray job submit --address="http://127.0.0.1:8265" \
        --runtime-env=verl/trainer/runtime_env.yaml \
        --no-wait \
        -- \
        python3 -m verl.trainer.main_ppo \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=2 \
        ...

.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/ray/submit.png?raw=true

2. Then you can check the job status with the following commands:

- ray job list: list all jobs submitted to the cluster.
- ray job logs <Submission ID>: query the logs of the job.
- ray job status <Submission ID>: query the status of the job.
- ray job stop <Submission ID>: request the job to be stopped.

3. You can also access driver/task/actor logs in ``/tmp/ray/session_latest/logs/``, driver log is ``job-driver-raysubmit_<Submission ID>.log``.

4. We strongly recommend you to view job detail from dashboard in multinode training, because it provide more structure way to view the job information.

.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/ray/job.png?raw=true
.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/ray/job_detail.png?raw=true


Slurm
-----
TBD

How to debug?
---------------------

Legacy Ray Debugger
~~~~~~~~~~~~~~~~~~~
1. Ray has a builtin legacy `debugger <https://docs.ray.io/en/latest/ray-observability/user-guides/debug-apps/ray-debugging.html>`_ that allows you to debug your distributed applications. To enable debugger, start ray cluster with ``RAY_DEBUG=legacy`` and ``--ray-debugger-external``.

.. code-block:: bash

    # start head node
    RAY_DEBUG=legacy ray start --head --dashboard-host=0.0.0.0 --ray-debugger-external
    # start worker node
    RAY_DEBUG=legacy ray start --address='10.124.46.192:6379' --ray-debugger-external

2. Set up breakpoint in your code, and submit job to cluster. Then run ``ray debug`` to wait breakpoint:

.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/ray/legacy.png?raw=true

Ray Distributed Debugger VSCode Extension
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Starting with Ray 2.39, Anyscale introduce a new `Ray Distributed Debugger <https://docs.ray.io/en/latest/ray-observability/ray-distributed-debugger.html>`_ VSCode extension. Please follow the instruction to install the extension, and then add cluster with the dashboard address you get above.

*NOTE: Don't forget remove RAY_DEBUG=legacy and --ray-debugger-external in ray start*

.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/ray/debugger.png?raw=true

2. Set up breakpoint in your code, and submit job to cluster. Then the extension will show the breakpoint information.

.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/ray/breakpoint.png?raw=true
