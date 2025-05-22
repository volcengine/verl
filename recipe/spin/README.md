# SPIN: Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models (verl Recipe)

This repository hosts a `verl` recipe inspired by the paper **"Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models"** (SPIN). The implementation uses an **Online Direct Preference Optimization (Online DPO)** approach for language model alignment. This method allows a model to iteratively improve its capabilities by learning from preferences generated using its own outputs, potentially reducing reliance on external preference datasets or stronger teacher models.

Paper Authors: [Zixiang Chen](https://github.com/uclaml/SPIN)\*, [Yihe Deng](https://github.com/uclaml/SPIN)\*, [Huizhuo Yuan](https://scholar.google.com/citations?user=8foZzX4AAAAJ)\*, [Kaixuan Ji](https://scholar.google.com/citations?user=FOoKDukAAAAJ), [Quanquan Gu](https://web.cs.ucla.edu/~qgu/)

verl Implementation Authors: [Chendong Wang](https://cdwang96.github.io/), [Chenyang Zhao](https://github.com/zhaochenyang20)

[[Webpage](https://uclaml.github.io/SPIN/)] [[Huggingface](https://huggingface.co/papers/2401.01335)] [[Paper](https://arxiv.org/abs/2401.01335)] [[Original Implementation](https://github.com/uclaml/SPIN)]

## Algorithm: Online DPO Inspired by SPIN

This recipe implements an Online DPO algorithm adapted to the `verl` Reinforcement Learning framework, drawing inspiration from concepts presented in SPIN. It provides an alternative to PPO for fine-tuning language models.

**Core Idea:** Instead of maximizing a scalar reward signal, this approach directly optimizes the policy model to align with preference data generated *online* during training:

1.  **Generation:** The current policy model (actor) generates two (or more) responses for each prompt in a batch.
2.  **Preference Labeling:** A reward model or reward function evaluates these generated responses to determine which one is preferred (chosen) and which is dispreferred (rejected).
3.  **Update:** This preference tuple (`prompt`, `chosen_response`, `rejected_response`) is used to update the actor model using the DPO loss function, comparing against a reference model.

**Connection to SPIN:**
While this recipe uses the DPO loss, the online generation loop where the current model generates data used for its own update shares conceptual similarities with the self-play idea in SPIN. The periodic update of the reference model (potentially using weights from the actor) further aligns with SPIN's iterative self-improvement concepts.

**Reference Papers:**
* **SPIN:** [Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models](https://arxiv.org/abs/2401.01335) (Chen et al., 2024)
* **DPO:** [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) (Rafailov et al., 2023)

## Implementation within verl 
The recipe is expected to be working on verl v0.3.0.post1

This implementation adapts the existing PPO infrastructure provided by `verl`:

* **No Critic:** The value function critic model used in PPO is not required and is omitted.
* **Reference Model:** An explicit reference policy model (`ref_policy_wg`) is maintained and used in the DPO loss calculation. This implementation allows for periodically updating the reference model's weights from the actor model (controlled by `ref_update_freq`).
* **Preference Calculation:** Logic (`compute_onlineDPO_pref` in `core_algos.py`) determines chosen/rejected pairs based on scores from a reward source.
* **DPO Loss:** The PPO policy loss and advantage calculations are replaced with the DPO loss computation (`compute_online_dpo_loss` in `core_algos.py`) within the actor update step (`dp_actor.py`).
* **Training Orchestration:** The `SpinTrainer` (in `spin_trainer.py`) manages the training loop: generation, preference labeling, optional reference model updates, and policy updates via the DPO loss.

## Reproduce the Experiment (Example Setup)

The following steps outline how to set up the environment and run the SPIN recipe, based on the provided test log using GSM8K and Qwen2.5-3B-Instruct.

1.  **Setup Environment (Example using Docker):**
    ```bash
    # Start a container with GPU access and shared memory
    docker run -it --name spin_test --gpus all \
        --shm-size=32g \
        --ipc=host \
        -v /path/to/host/.cache:/root/.cache \
        -e HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN> \
        lmsysorg/sglang:latest \
        /bin/bash

    # Inside the container or on your host machine:
    # Ensure /tmp is writable
    mkdir -p /tmp
    chmod 1777 /tmp

    # Install Python 3.10 (if not present) and venv
    sudo apt update
    sudo apt install -y python3.10 python3.10-venv tmux
    python3 -m ensurepip --upgrade

    # Create and activate a virtual environment
    python3 -m venv ~/.python/spin_env
    source ~/.python/spin_env/bin/activate

    # Install uv (fast package installer)
    python3 -m pip install uv
    ```

2.  **Install verl and Dependencies:**
    ```bash
    # Clone the verl repository and checkout the spin branch
    cd ~
    git clone git@github.com:volcengine/verl.git](git@github.com:volcengine/verl.git) && cd verl

    # Install flash-attn (handle potential build issues)
    python3 -m uv pip install wheel packaging
    python3 -m uv pip install flash-attn --no-build-isolation --no-deps

    # Install verl with sglang extras
    python3 -m uv pip install -e ".[sglang]"
    ```
    *Note: If `flash-attn` installation fails, try the manual steps again or consult its documentation.*

3.  **Login & Download Data/Model:**
    ```bash
    # Login to Weights & Biases (optional, for logging)
    export WANDB_API_KEY=<YOUR_WANDB_API_KEY>
    # wandb login

    # Download the GSM8K dataset
    python3 examples/data_preprocess/gsm8k.py --local_dir ~/data/gsm8k # Adjusted path

    # Download the base model (Example: Qwen2.5-3B-Instruct)
    huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir $HOME/models/Qwen2.5-3B-Instruct
    ```

4.  **Configure:**
    * Modify the configuration file (e.g., `config/spin_trainer.yaml` or the one specified in the run script) with correct paths to your downloaded model, data, desired hyperparameters (`dpo_beta`, learning rate, etc.), and distributed training settings (nodes, GPUs per node).
    * Pay attention to `actor_rollout_ref.model_path`, `data` paths, `reward_model` config (if using one), and `trainer.ref_update_freq`.

5.  **Run Training:**
    ```bash
    # Set CUDA visible devices (adjust based on your hardware and config)
    export CUDA_VISIBLE_DEVICES=0,1,2,3

    # Launch the training script (e.g., test.sh or a custom script)
    # Ensure test.sh points to the correct config and main script
    bash recipe/spin/run_spin.sh
    ```

## Configuration

* The primary configuration is typically managed through a YAML file specified in the launch script (e.g., `config/spin_trainer.yaml`).
* Key configuration sections:
    * `data`: Paths to training/validation prompt files, batch sizes, sequence lengths.
    * `actor_rollout_ref`: Paths to the base model (used for actor and initial reference), FSDP settings, optimization parameters (learning rate, scheduler).
    * `reward_model`: Configuration for the reward model used for online preference labeling (path, batch size, etc.). Can be omitted if using a simpler reward function.
    * `algorithm`: DPO-specific hyperparameters like `dpo_beta`, `dpo_loss_type`.
    * `trainer`: Distributed training settings (nodes, GPUs per node), logging (WandB), checkpointing frequency, and `ref_update_freq` (set > 0 to enable periodic reference model updates from the actor).

## Key Files

* `main_spin.py`: Main entry point using Hydra to load config and launch the `SpinTrainer`.
* `spin_trainer.py`: Defines the `SpinTrainer` class orchestrating the Online DPO training loop.
* `fsdp_workers.py`: Implements Ray workers (Actor, Reference) potentially using FSDP.
* `dp_actor.py`: Contains the actor class, including the DPO policy update logic.
* `core_algos.py`: Includes helper functions for `compute_online_dpo_loss` and `compute_onlineDPO_pref`.
* `config/spin_trainer.yaml` (or similar): Main Hydra configuration file for the recipe.
* `run_spin.sh` (or similar): Example bash script for launching a training run.
* `README.md`: This file.

## Acknowledgement

We sincerely thank the contribution and guidance from the `verl` community and advisors, including (adapted from SPPO):

-   [Yue Wu](https://yuewu.us/)
-   [Yuhao Yang](https://github.com/yhyang201)
-   [Yifan Zhang](https://github.com/yifanzhang-pro)
-   [Yongan Xiang](https://github.com/BearBiscuit05)
-   [Junrong Lin](https://github.com/ocss884)
-   [Yuxuan Tong](https://github.com/tongyx361)
-   [Guangming Shen](https://github.com/PeterSH6)
-   [Biao He](https://www.linkedin.com/in/biao-he/)
-   [Qingquan Song](https://qingquansong.github.io/)
-   [Quanquan Gu](https://web.cs.ucla.edu/~qgu/)
