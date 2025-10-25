<p align="center">
<h1 align="center">FAPO: Flawed-Aware Policy Optimization for Efficient and Reliable Reasoning</h1>

<p align="center">
    <a href="https://fapo-rl.github.io/"><img alt="Project Page" src="https://img.shields.io/badge/📒-Project Page-blue"></a>
    <a href="https://verl.readthedocs.io/en/latest/advance/reward_loop.html"><img alt="Infra Design" src="https://img.shields.io/badge/🏗️-Infra Design-teal">
    <a href="https://huggingface.co/collections/dyyyyyyyy/fapo"><img alt="Resources" src="https://img.shields.io/badge/🤗 HuggingFace-Data & Models-green"></a>
    <a href=""><img alt="Paper" src="https://img.shields.io/badge/📄-Arxiv Paper-orange"></a>
    <a href="https://github.com/yyDing1/FAPO"><img alt="Code" src="https://img.shields.io/badge/💻-Code-blueviolet"></a>
</p>

- **Algorithm Insights:** Visit our [Project Page](https://fapo-rl.github.io/) for an overview; comprehensive details are available in the [Paper]().
- **Infrastructure Design:** Refer to the [Reward Loop](https://verl.readthedocs.io/en/latest/advance/reward_loop.html) document for architectural insights.
- **Open-Source Software:** Explore the [Huggingface Collections](https://huggingface.co/collections/dyyyyyyyy/fapo) for datasets and models.


![fapo-result](https://fapo-rl.github.io/_astro/intro_main.DKe72RHX_1Us2HB.webp)

## Step 1: Train FAPO-GenRM-4B (Generative Reward Model)

We provide our training and evaluation datasets [here](https://huggingface.co/datasets/dyyyyyyyy/FAPO-Critic).
Directly download them to `${RAY_DATA_HOME}/data/`.

Then, submit the training job to the ray cluster:

```bash
cd verl # Repo root
export RAY_ADDRESS="..." # The Ray cluster address to connect to
export RAY_DATA_HOME="..." # The directory to store the data
export WORKING_DIR="${PWD}" # The local directory to package to the Ray cluster
# Set the runtime environment like env vars and pip packages for the Ray cluster in yaml
export RUNTIME_ENV="./recipe/fapo/runtime_env.yaml" # This sets environment variables for the Ray cluster
bash recipe/fapo/run_fapo_genrm_train.sh
```

You can skip this step if you want to use the pre-trained FAPO-GenRM-4B model available [here](https://huggingface.co/dyyyyyyyy/FAPO-GenRM-4B).

## Step 2: Integrate the GRM into the Final Training

Our training data is identical to that of DAPO-Math-17K, except that we replace the instruction with "Put the final answer in \boxed{}", which is a common practice for current instruct models.

You can construct the training and evaluation datasets by:
```bash
python recipe/fapo/prepare_fapo_data.py --local_dir ${RAY_DATA_HOME}/data/
```

Or you can directly use the data available [here](https://huggingface.co/datasets/dyyyyyyyy/FAPO-Reasoning-Dataset).

To integrate the GRM into the final training, we provide two options:

1. **Launch GRM as an external service:** Launch multiple model servers and a router in advance to handle and dispatch incoming requests. Refer to `verl/recipe/genrm_remote` for more details. The scripts is `verl/recipe/fapo/run_fapo_{7b/32b}_remote.sh`.
2. **Launch GRM in verl single controller:** Start the GRM model directly inside the verl single controller with an integrated router. (Note: this feature is still unstable for large-scale training scenarios.)

```bash
cd verl # Repo root
export RAY_ADDRESS="..." # The Ray cluster address to connect to
export WORKING_DIR="${PWD}" # The local directory to package to the Ray cluster
# Set the runtime environment like env vars and pip packages for the Ray cluster in yaml
export RUNTIME_ENV="./recipe/fapo/runtime_env.yaml" # This sets environment variables for the Ray cluster

# run Baseline Models
bash recipe/fapo/run_baseline_7b.sh  # 7b baseline model
bash recipe/fapo/run_baseline_32b.sh  # 32b baseline model

# run FAPO Models (with external GRM service)
# Note that you should launch the external GRM service first,
# and specify the router address in the compute_score function
bash recipe/fapo/run_fapo_7b_remote.sh  # 7b fapo model
bash recipe/fapo/run_fapo_32b_remote.sh  # 32b fapo model

# run FAPO Models (single controller mode)
bash recipe/fapo/run_fapo_7b.sh  # 7b fapo model
bash recipe/fapo/run_fapo_32b.sh  # 32b fapo model
```

## Infrastructure Design

We implement RewardLoop to enable efficient and flexible reward computation.
The core implementation can be found in `verl/experimental/reward/`.
Refer to [this official document](https://verl.readthedocs.io/en/latest/advance/reward_loop.html) for more implementation details.
