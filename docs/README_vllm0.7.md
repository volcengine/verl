# Readme for verl(vllm>=0.7) version

## Installation

Note: This version of veRL supports **FSDP** for training and **vLLM** for rollout. (Megatron-LM is not supported yet.)

```
# Create the conda environment
conda create -n verl python==3.10
conda activate verl

# Install verl
git clone https://github.com/volcengine/verl.git
cd verl
pip3 install -e .

# Install vLLM>=0.7
# (Option1) pip3 install vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
# (Option2) pip3 install "vllm>=0.7.0" 

# Install flash-attn
pip3 install flash-attn --no-build-isolation

```

Note that if you are installing stable versions of vLLM (Option2), you need to make some tiny patches manually on vllm (/path/to/site-packages/vllm after installation) after the above steps:

- vllm/distributed/parallel_state.py: Remove the assertion below:

```
if (world_size
        != tensor_model_parallel_size * pipeline_model_parallel_size):
    raise RuntimeError(
        f"world_size ({world_size}) is not equal to "
        f"tensor_model_parallel_size ({tensor_model_parallel_size}) x "
        f"pipeline_model_parallel_size ({pipeline_model_parallel_size})")

```

- vllm/executor/uniproc_executor.py: change `local_rank = rank` to `local_rank = int(os.environ["LOCAL_RANK"])`
- vllm/model_executor/model_loader/weight_utils.py: remove the `torch.cuda.empty_cache()` in `pt_weights_iterator`

These modifications have already been merged into the main branch of vLLM. Thus nightly vLLM or building vLLM from source do not need these patches.

## Features

### Use cuda graph

After installation, examples using FSDP as training backends can be used. By default, the `enforce_eager` is set to True, which disables the cuda graph. To enjoy cuda graphs and the sleep mode of vLLM>=0.7, add the following lines to the bash script:

```
actor_rollout_ref.rollout.enforce_eager=False \
actor_rollout_ref.rollout.free_cache_engine=False \

```

For a typical job like examples/ppo_trainer/run_qwen2-7b_seq_balance.sh, the rollout generation time is 115 seconds with vLLM0.6.3, while it is 85 seconds with vLLM0.7.0. By enabling the cudagraph, the generation duration is further reduced to 62 seconds.

**Note:** Currently, if the `n` is greater than 1 in `SamplingParams` in vLLM>=0.7, there is a potential performance issue on the stability of rollout generation time (Some iterations would see generation time bursts). We are working with the vLLM team to check this issue.

### Other features in vLLM

1. **num_scheduler_step>1:** not supported yet (weight loading has not been aligned with `MultiStepModelRunner`)
2. **Prefix caching:** not supported yet (vLLM sleep mode does not support prefix caching)
3. **Chunked prefill:** supported