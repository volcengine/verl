verl performance tuning for AMD (ROCm Kernel)
=====================================================

Last updated: 04/25/2025.

Author: `Yang Wang <https://github.com/YangWang92/>`_

Patch vLLM to Enable Sleep Mode for AMD GPUs
--------------------------------------------------------------

By default, verl requires vLLM to enable sleep mode, which allows vLLM to offload GPU memory to CPU memory after rollout. However, this feature is still under review by the vLLM community.

To enable vLLM's sleep mode, you can first use community patched code (from `this pull request <https://github.com/vllm-project/vllm/pull/12695>`_) to build vLLM from the source code in the corresponding pull request. After the patch merged in vLLM main branch, you can directly install vLLM from the latest version.

1. Clone the vLLM repository and build it with the following commands:

.. code-block:: bash

    git clone -b sleep_amd https://github.com/HollowMan6/vllm.git
    cd vllm
    sudo ln -sf /opt/rocm/lib/libamdhip64.so /usr/lib/libamdhip64.so
    VLLM_TARGET_DEVICE=rocm ROCM_PATH=/opt/rocm/ VLLM_GPU_LANG=HIP SETUPTOOLS_SCM_PRETEND_VERSION=0.8.4.dev python3 setup.py develop

2. Additionally, make sure to use the ROCm version in your Docker image lager than or equal to ROCm 6.3.4, and we recommend to use ROCm 6.4.0 for better performance (see `this comment <https://github.com/vllm-project/vllm/pull/12695#issuecomment-2637839574>`_).

After the upgrade, you can verify whether sleep mode is enabled by running the following test code (from `this comment <https://github.com/vllm-project/vllm/pull/12695#issuecomment-2637839574>`_).

.. code-block:: python

	import torch
	from vllm import LLM

	llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", enable_sleep_mode=True)

	def run_inference(prompt):
		outputs = llm.generate(prompt)
		for output in outputs:
			prompt = output.prompt
			generated_text = output.outputs[0].text
			print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


	print("CUDA Memory Usage (after inference):")
	torch.cuda.empty_cache()
	print(f"{torch.cuda.memory_allocated()=}")

	run_inference("San Francisco is")
	llm.sleep()

	print("CUDA Memory Usage (after sleep):")
	torch.cuda.empty_cache()
	print(f"{torch.cuda.memory_allocated()=}")

	llm.wake_up()

	print("CUDA Memory Usage (after wakeup):")
	torch.cuda.empty_cache()
	print(f"{torch.cuda.memory_allocated()=}")

	run_inference("Paris is")

If sleep mode is enabled, you should see the memory usage reduce after sleep.

After applying the vLLM patch and completing the installation, you can enable sleep mode in verl to reduce memory overhead. This allows verl to offload unused GPU memory during rollout, significantly lowering the memory footprint during long-context training or multi-node reinforcement learning.


Enable CUDA Graph and Bypass ROCm-related issues
--------------------------------------------------------------

Due to potential issues with CUDA graph capture in ROCm, we’ve found that vLLM’s CUDA graph feature cannot be enabled on multiple nodes in verl on AMD platforms with vLLM V1 mode. This leads to significantly slower rollout performance.

Our investigation shows that ROCm may trigger an unexpected crash when attempting to capture large batches with CUDA graph. One workaround is to patch the LLM configuration (from `this commit <https://github.com/volcengine/verl/blob/v0.3.0.rc0/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py#L100-L115>`_).

.. code-block:: python
	
    self.inference_engine = LLM(
        model=model_path,
        enable_sleep_mode=True,
        tensor_parallel_size=tensor_parallel_size,
        distributed_executor_backend="external_launcher",
        dtype=config.dtype,
        enforce_eager=config.enforce_eager,
        gpu_memory_utilization=config.gpu_memory_utilization,
        disable_custom_all_reduce=True,
        disable_mm_preprocessor_cache=True,
        limit_mm_per_prompt=limit_mm_per_prompt,
        skip_tokenizer_init=False,
        max_model_len=max_model_len,
        load_format=load_format,
        disable_log_stats=config.disable_log_stats,
        max_num_batched_tokens=max_num_batched_tokens,
        enable_chunked_prefill=config.enable_chunked_prefill,
        enable_prefix_caching=True,
        trust_remote_code=trust_remote_code,
        # enable compilation config to bypass oom on rocm
	# change depends on your GPU memory size
        compilation_config={"cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64]},
        seed=config.get('seed', 0),
    )

Then, you can enable CUDA graph by setting the following environment variables (see `this page <https://github.com/volcengine/verl/blob/v0.3.0.rc0/docs/README_vllm0.8.md>`_):

.. code-block:: bash

	actor_rollout_ref.rollout.enforce_eager=False \
	actor_rollout_ref.rollout.free_cache_engine=False \
