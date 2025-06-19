Frequently Asked Questions
====================================

Ray related
------------

How to add breakpoint for debugging with distributed Ray?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Please checkout the official debugging guide from Ray: https://docs.ray.io/en/latest/ray-observability/ray-distributed-debugger.html


"Unable to register worker with raylet"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The cause of this issue is due to some system setting, e.g., SLURM added some constraints on how the CPUs are shared on a node. 
While `ray.init()` tries to launch as many worker processes as the number of CPU cores of the machine,
some constraints of SLURM restricts the `core-workers` seeing the `raylet` process, leading to the problem.

To fix this issue, you can set the config term ``ray_init.num_cpus`` to a number allowed by your system.

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

3. Modify `examples/slurm/ray_on_slurm.slurm <https://github.com/volcengine/verl/blob/main/examples/slurm/ray_on_slurm.slurm>`_ with your cluster's own information.

4. Submit the job script to the Slurm cluster with `sbatch`.

Please note that Slurm cluster setup may vary. If you encounter any issues, please refer to Ray's
`Slurm user guide <https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html>`_ for common caveats.

If you changed Slurm resource specifications, please make sure to update the environment variables in the job script if necessary.


Install related
------------------------

NotImplementedError: TensorDict does not support membership checks with the `in` keyword. 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Detail error information: 

.. code:: bash

    NotImplementedError: TensorDict does not support membership checks with the `in` keyword. If you want to check if a particular key is in your TensorDict, please use `key in tensordict.keys()` instead.

Cause of the problem: There is no suitable version of tensordict package for the linux-arm64 platform. The confirmation method is as follows:

.. code:: bash

    pip install tensordict==0.6.2

Output example:

.. code:: bash

    ERROR: Could not find a version that satisfies the requirement tensordict==0.6.2 (from versions: 0.0.1a0, 0.0.1b0, 0.0.1rc0, 0.0.2a0, 0.0.2b0, 0.0.3, 0.1.0, 0.1.1, 0.1.2, 0.8.0, 0.8.1, 0.8.2, 0.8.3)
    ERROR: No matching distribution found for tensordict==0.6.2

Solution 1st:
  Install tensordict from source code:

.. code:: bash

    pip uninstall tensordict
    git clone https://github.com/pytorch/tensordict.git
    cd tensordict/
    git checkout v0.6.2
    python setup.py develop
    pip install -v -e .

Solution 2nd:
  Temperally modify the error takeplace codes: tensordict_var -> tensordict_var.keys()


Illegal memory access
---------------------------------

If you encounter the error message like ``CUDA error: an illegal memory access was encountered`` during rollout, most likely it is due to a known issue from vllm(<=0.6.3).
Please set the following environment variable. The env var must be set before the ``ray start`` command if any.

.. code:: bash

    export VLLM_ATTENTION_BACKEND=XFORMERS

If in doubt, print this env var in each rank to make sure it is properly set.

Checkpoints
------------------------

If you want to convert the model checkpoint into huggingface safetensor format, please refer to ``scripts/model_merger.py``.


Triton ``compile_module_from_src`` error
------------------------------------------------

If you encounter triton compilation error similar to the stacktrace below, please set the ``use_torch_compile`` flag according to
https://verl.readthedocs.io/en/latest/examples/config.html to disable just-in-time compilation for fused kernels.

.. code:: bash

  File "/data/lbh/conda_envs/verl/lib/python3.10/site-packages/triton/runtime/jit.py", line 345, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
  File "/data/lbh/conda_envs/verl/lib/python3.10/site-packages/triton/runtime/autotuner.py", line 338, in run
    return self.fn.run(*args, **kwargs)
  File "/data/lbh/conda_envs/verl/lib/python3.10/site-packages/triton/runtime/jit.py", line 607, in run
    device = driver.active.get_current_device()
  File "/data/lbh/conda_envs/verl/lib/python3.10/site-packages/triton/runtime/driver.py", line 23, in __getattr__
    self._initialize_obj()
  File "/data/lbh/conda_envs/verl/lib/python3.10/site-packages/triton/runtime/driver.py", line 20, in _initialize_obj
    self._obj = self._init_fn()
  File "/data/lbh/conda_envs/verl/lib/python3.10/site-packages/triton/runtime/driver.py", line 9, in _create_driver
    return actives[0]()
  File "/data/lbh/conda_envs/verl/lib/python3.10/site-packages/triton/backends/nvidia/driver.py", line 371, in __init__
    self.utils = CudaUtils()  # TODO: make static
  File "/data/lbh/conda_envs/verl/lib/python3.10/site-packages/triton/backends/nvidia/driver.py", line 80, in __init__
    mod = compile_module_from_src(Path(os.path.join(dirname, "driver.c")).read_text(), "cuda_utils")
  File "/data/lbh/conda_envs/verl/lib/python3.10/site-packages/triton/backends/nvidia/driver.py", line 57, in compile_module_from_src
    so = _build(name, src_path, tmpdir, library_dirs(), include_dir, libraries)
  File "/data/lbh/conda_envs/verl/lib/python3.10/site-packages/triton/runtime/build.py", line 48, in _build
    ret = subprocess.check_call(cc_cmd)
  File "/data/lbh/conda_envs/verl/lib/python3.10/subprocess.py", line 369, in check_call
    raise CalledProcessError(retcode, cmd)

What is the meaning of train batch size, mini batch size, and micro batch size?
------------------------------------------------------------------------------------------

This figure illustrates the relationship between different batch size configurations.

https://excalidraw.com/#json=pfhkRmiLm1jnnRli9VFhb,Ut4E8peALlgAUpr7E5pPCA

.. image:: https://github.com/user-attachments/assets/16aebad1-0da6-4eb3-806d-54a74e712c2d

How to generate ray timeline to analyse performance of a training job?
------------------------------------------------------------------------------------------

To generate the ray timeline file, you can set the config term ``ray_init.timeline_file`` to a json file path.
For example:

.. code:: bash

    ray_init.timeline_file=/tmp/ray_timeline.json
  
The file will be generated in the specified path at the end of a training job.
You can use tools like chrome://tracing or the Perfetto UI and view the ray timeline file.

This figure shows the ray timeline file generated by from a training job on 1 node with 4 GPUs

.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/ray_timeline.png?raw=true

How to set proxy only for wandb?
------------------------------------------------------------------------------------------

If you need a proxy to access wandb, you can add below config in your training job script.
Comparing to using global https_proxy env variable, this approach won't mess up other http requests, such as ChatCompletionScheduler.

.. code:: bash

  +trainer.wandb_proxy=http://<your proxy and port>
