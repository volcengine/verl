Frequently Asked Questions
====================================

Ray related
------------

How to add breakpoint for debugging with distributed Ray?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Please checkout the official debugging guide from Ray: https://docs.ray.io/en/latest/ray-observability/ray-distributed-debugger.html


Distributed training
------------------------

How to run multi-node post-training with Ray?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can start a ray cluster and submit a ray job, following the official guide from Ray: https://docs.ray.io/en/latest/ray-core/starting-ray.html

Then in the configuration, set the ``trainer.nnode`` config to the number of machines for your job.

How to use verl on a Slurm-managed cluster?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ray provides users with `this <https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html>`_ official
tutorial to start a Ray cluster on top of Slurm. We have verified the :doc:`GSM8K example<../examples/gsm8k_example>`
on a Slurm cluster under a multi-node setting with the following steps.

1. [Optional] If your cluster support `Apptainer or Singularity <https://apptainer.org/docs/user/main/>`_ and you wish
to use it, convert verl's Docker image to an Apptainer image. Alternatively, set up the environment with the package
manager available on your cluster or use other container runtimes (e.g. through `Slurm's OCI support <https://slurm.schedmd.com/containers.html>`_) available to you.

.. code:: bash

    apptainer pull /your/dest/dir/vemlp-th2.4.0-cu124-vllm0.6.3-ray2.10-te1.7-v0.0.3.sif docker://verlai/verl:vemlp-th2.4.0-cu124-vllm0.6.3-ray2.10-te1.7-v0.0.3

2. Follow :doc:`GSM8K example<../examples/gsm8k_example>` to prepare the dataset and model checkpoints.

3. Modify `examples/slurm/ray_on_slurm.slurm <https://github.com/volcengine/verl/blob/main/verl/examples/slurm/ray_on_slurm.slurm>`_ with your cluster's own information.

4. Submit the job script to the Slurm cluster with `sbatch`.

Please note that Slurm cluster setup may vary. If you encounter any issues, please refer to Ray's
`Slurm user guide <https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html>`_ for common caveats.

Illegal memory access
---------------------------------

If you encounter the error message like ``CUDA error: an illegal memory access was encountered`` during rollout, most likely it is due to a known issue from vllm.
Please set the following environment variable. The env var must be set before the ``ray start`` command if any.

.. code:: bash

    export VLLM_ATTENTION_BACKEND=XFORMERS

If in doubt, print this env var in each rank to make sure it is properly set.

Checkpoints
------------------------

If you want to convert the model checkpoint into huggingface safetensor format, please refer to ``scripts/model_merger.py``.
