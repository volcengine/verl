Multinode Training
==================

.. _wuxibin89: https://github.com/wuxibin89

Author: `Xibin Wu <https://github.com/wuxibin89>`_, `Yusheng Su <https://yushengsu-thu.github.io/>`_.

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









Multi-node training on AMD clusters
---------------------------------------------------------------------------------------

If you want to run multi-node training with slurm with Docker/Podman container on AMD Cluster, you can use the following script. 

If you encounter any issues in using AMD GPUs running verl, please contact `Yusheng Su <https://yushengsu-thu.github.io/>`_.

.. note::
    1. You need to use ``podman`` or ``docker`` in the following script. We will release the apptainer script later.
    2. If you want to use ``podman``, you just replace ``docker`` with ``podman`` in the following script.

The script includes the following steps:

1. SLURM Configuration
2. Environment Setup
3. Docker/Podman Container Setup
4. Ray Cluster Initialization
5. Data Preprocessing
6. Model Setup
7. Training Launch


slurm_script.sh
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    #!/bin/bash

    #SBATCH --job-name=verl-ray-on-slurm
    #SBATCH --nodes=2
    #SBATCH --ntasks-per-node=2
    #SBATCH --mem=200G
    #SBATCH --time=30-00:00:00
    #SBATCH --gpus-per-node=8
    #SBATCH --cpus-per-task=28
    #SBATCH --output=../verl_log/slurm-%j.out
    #SBATCH --error=../verl_log/slurm-%j.err
    #SBATCH --nodelist=gpu-[0,1]


    # load necessary modules
    ### Run this setup
    # [Cluster]: Use docker
    # docker pull docker.io/rocm/vllm:rocm6.2_mi300_ubuntu20.04_py3.9_vllm_0.6.4


    ##########################################################################
    ###The following setting should be set in different project and cluster###
    ##########################################################################

    ### Project
    CONTAINER_NAME="multinode_verl_training"
    IMG="verl.rocm"
    DOCKERFILE="docker/Dockerfile.rocm"
    # echo $PWD
    verl_workdir="${HOME}/projects/verl_upstream"
    export TRANSFORMERS_CACHE="${HOME}/.cache/huggingface"
    export HF_HOME=$TRANSFORMERS_CACHE

    ### Cluster Network Setting
    export NCCL_DEBUG=TRACE
    export GPU_MAX_HW_QUEUES=2
    export TORCH_NCCL_HIGH_PRIORITY=1
    export NCCL_CHECKS_DISABLE=1
    # export NCCL_IB_HCA=rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7 
    export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_8,mlx5_9
    export NCCL_IB_GID_INDEX=3
    export NCCL_CROSS_NIC=0
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export NCCL_PROTO=Simple
    export RCCL_MSCCL_ENABLE=0
    export TOKENIZERS_PARALLELISM=false
    export HSA_NO_SCRATCH_RECLAIM=1
    ##########################################################################

    ### For rocm and training script
    export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export ROCR_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES
    export CUDA_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES


    # Build and launch the Docker container
    srun bash -c "
        # Exit on any error
        set -e 

        # Clean up dangling images (images with <none> tag)
        docker image prune -f

        # Need to pull the docker first
        docker pull docker.io/rocm/vllm:rocm6.2_mi300_ubuntu20.04_py3.9_vllm_0.6.4
        
        if ! docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "${IMG}"; then
            echo \"Building ${IMG} image...\"
            docker build -f \"${DOCKERFILE}\" -t \"${IMG}\" .
        else
            echo \"${IMG} image already exists, skipping build\"
        fi

        # Removing old container if exists
        docker rm \"${CONTAINER_NAME}\" 2>/dev/null || true

        # Checking network devices
        ibdev2netdev

        # Launch the docker
        docker run --rm -d \
        -e HYDRA_FULL_ERROR=1 \
        -e HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES} \
        -e ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES} \
        -e CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
        -e NCCL_DEBUG=${NCCL_DEBUG} \
        -e GPU_MAX_HW_QUEUES=${GPU_MAX_HW_QUEUES} \
        -e TORCH_NCCL_HIGH_PRIORITY=${TORCH_NCCL_HIGH_PRIORITY} \
        -e NCCL_CHECKS_DISABLE=${NCCL_CHECKS_DISABLE} \
        -e NCCL_IB_HCA=${NCCL_IB_HCA} \
        -e NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX} \
        -e NCCL_CROSS_NIC=${NCCL_CROSS_NIC} \
        -e CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS} \
        -e NCCL_PROTO=${NCCL_PROTO} \
        -e RCCL_MSCCL_ENABLE=${RCCL_MSCCL_ENABLE} \
        -e TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM} \
        -e HSA_NO_SCRATCH_RECLAIM=${HSA_NO_SCRATCH_RECLAIM} \
        -e TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE} \
        -e HF_HOME=${HF_HOME} \
        --network host \
        --device /dev/dri \
        --device /dev/kfd \
        --device /dev/infiniband \
        --group-add video \
        --cap-add SYS_PTRACE \
        --security-opt seccomp=unconfined \
        --privileged \
        -v \${HOME}:\${HOME} \
        -v \${HOME}/.ssh:/root/.ssh \
        -w "${verl_workdir}" \
        --shm-size 128G \
        --name \"${CONTAINER_NAME}\" \
        \"${IMG}\" \
        tail -f /dev/null

        echo \"Container setup completed\"
    "
        # (Optional): If you do not want to root mode and require assign yuorself as the user
        # Please add `-e HOST_UID=$(id -u)` and `-e HOST_GID=$(id -g)` into the above docker launch script. 





    ### Ray launch the nodes before training

    # Getting the node names
    nodes_array=($(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' '))

    head_node=${nodes_array[0]}
    head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

    # if we detect a space character in the head node IP, we'll
    # convert it to an ipv4 address. This step is optional.
    if [[ "$head_node_ip" == *" "* ]]; then
        IFS=' ' read -ra ADDR <<<"$head_node_ip"
    if [[ ${#ADDR[0]} -gt 16 ]]; then
        head_node_ip=${ADDR[1]}
    else
        head_node_ip=${ADDR[0]}
    fi
        echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
    fi

    port=6379
    ip_head=$head_node_ip:$port
    export ip_head
    echo "IP Head: $ip_head"

    # make sure we set environment variables before Ray initialization
    export VLLM_ATTENTION_BACKEND=XFORMERS

    # Print out all env variables
    printenv

    echo "Starting HEAD at $head_node"
    srun --nodes=1 --ntasks=1 -w "$head_node" \
        docker exec "${CONTAINER_NAME}" \
            ray start --head --node-ip-address="$head_node_ip" --port=$port \
            --dashboard-port=8266 \
            --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_NODE}" --block &
    # optional, though may be useful in certain versions of Ray < 1.0.
    sleep 10

    # number of nodes other than the head node
    worker_num=$((SLURM_JOB_NUM_NODES - 1))

    for ((i = 1; i <= worker_num; i++)); do
        node_i=${nodes_array[$i]}
        echo "Debug: Starting worker on node_i = ${node_i}"
        if [ -z "$node_i" ]; then
            echo "Error: Empty node name for worker $i"
            continue
        fi
        echo "Starting WORKER $i at $node_i"
        srun --nodes=1 --ntasks=1 -w "$node_i" \
            docker exec "${CONTAINER_NAME}" \
                ray start --address "$ip_head" --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_NODE}" --block &
        sleep 5
    done




    # Ray initlization test (See whether any error in the above excution)
    echo "Testing Ray initialization in the slurm nodes..."
    docker exec "${CONTAINER_NAME}" python3 -c '
    import ray
    try:
        ray.init(address="auto")
        print("\n=== Ray Cluster Status ===")
        print(f"Number of nodes: {len(ray.nodes())}")
        for node in ray.nodes():
            print("Node: {}, Status: {}".format(node["NodeManagerHostname"], node["Alive"]))
            # print(f"Node: {node}")
        ray.shutdown()
        print("Ray initialization successful!")
    except Exception as e:
        print(f"Ray initialization failed: {str(e)}")
    '
    echo "=== Ray test completed ==="
    ######



    # Run data preprocessing

    echo "Starting data preprocessing..."
    docker exec "${CONTAINER_NAME}" \
        python3 "examples/data_preprocess/gsm8k.py" "--local_dir" "../data/gsm8k"

    echo "Starting data preprocessing..."
    docker exec "${CONTAINER_NAME}" \
        python3 "examples/data_preprocess/math_dataset.py" "--local_dir" "../data/math"

    train_files="../data/gsm8k/train.parquet"
    val_files="../data/gsm8k/test.parquet"

    # Download and test model
    echo "Loading model..."
    docker exec "${CONTAINER_NAME}" \
        python3 -c "import transformers; transformers.pipeline('text-generation', model='Qwen/Qwen2-7B-Instruct')"
    MODEL_PATH="Qwen/Qwen2-7B-Instruct"

    # Set model path after pipeline test
    MODEL_PATH="Qwen/Qwen2.5-0.5B-Instruct"

    echo "== Data and model loading Done =="

    echo "Start to train..."

    docker exec "${CONTAINER_NAME}" \
        python3 -c "import transformers; transformers.pipeline('text-generation', model='Qwen/Qwen2-7B-Instruct')"
    MODEL_PATH="Qwen/Qwen2-7B-Instruct"


    PYTHONUNBUFFERED=1 srun --overlap --nodes=${SLURM_NNODES} --ntasks=1 -w "$head_node" \
        docker exec "${CONTAINER_NAME}" \
        python3 -m verl.trainer.main_ppo \
        data.train_files=$train_files \
        data.val_files=$val_files \
        data.train_batch_size=1024 \
        data.max_prompt_length=1024 \
        data.max_response_length=1024 \
        actor_rollout_ref.model.path=$MODEL_PATH \
        actor_rollout_ref.model.enable_gradient_checkpointing=False \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=256 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        critic.optim.lr=1e-5 \
        critic.model.use_remove_padding=True \
        critic.model.path=$MODEL_PATH \
        critic.model.enable_gradient_checkpointing=False \
        critic.ppo_micro_batch_size_per_gpu=8 \
        critic.model.fsdp_config.param_offload=False \
        critic.model.fsdp_config.optimizer_offload=False \
        algorithm.kl_ctrl.kl_coef=0.0001 \
        trainer.critic_warmup=0 \
        trainer.logger=['console','wandb'] \
        trainer.project_name='verl_example' \
        trainer.experiment_name='Qwen2.5-32B-Instruct_function_rm' \
        trainer.n_gpus_per_node=${SLURM_GPUS_PER_NODE} \
        trainer.val_before_train=False \
        trainer.nnodes=${SLURM_NNODES} \
        trainer.save_freq=-1 \
        trainer.test_freq=10 \
        trainer.total_epochs=15


Run multi-node training with above slurm_script.sh
~~~~~~~~~~~~~~~~~~~~
Just sbatch your slurm_script.sh

.. code-block:: bash

    sbatch slurm_script.sh

