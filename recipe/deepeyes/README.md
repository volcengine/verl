# DeepEyes: Incentivizing "Thinking with Images" via Reinforcement Learning

This directory contains the implementation for reproducing the DeepEyes paper within the verl framework, supporting multi-turn visual tool calls. This implementation is based on the original [DeepEyes paper](https://arxiv.org/abs/2505.14362) and its [official implementation](https://github.com/Visual-Agent/DeepEyes), integrated with the multi-modal and multi-turn capabilities of the verl framework.

## Reproducing the Experiment

First, preprocess the original DeepEyes-Dataset-47k. This step is necessary to add parameters required by the VERL framework's tools.

```bash
python recipe/deepeyes/deepeyes47k_preprocess.py --dataset_dir <your_local_data_directory> --save_dir <directory_to_save_processed_data>
```

> **Note on the 'Chart' Dataset:**
> 
> The provided preprocessing script intentionally excludes `data_v0.8_visual_toolbox_v2.parquet`, which contains the 'Chart' data. This subset consists of very high-resolution images, often resembling large figures composed of multiple sub-plots, much like those found in academic papers.
>
> Consequently, even after using the zoom-in tool, the resulting cropped images remain large. This poses a significant risk of causing Out-of-Memory (OOM) errors, which can abruptly terminate the training process. 
> 
> **We strongly recommend against training on the 'Chart' dataset on a single node.**

Next, launch an inference service to act as a judge for reward calculation. You can use the following script as a reference:

```bash
vllm serve /path/to/Qwen2.5-72B-Instruct \
    --port 18901 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 32768 \
    --tensor-parallel-size 1 \
    --served-model-name "judge" \
    --trust-remote-code \
    --disable-log-requests \
    --tensor-parallel-size 8 \
```

Finally, you can start the training:

```bash
bash recipe/deepeyes/run_deepeyes_grpo.sh
```

## References and Acknowledgements

- [DeepEyes Paper](https://arxiv.org/abs/2505.14362)
- [DeepEyes Official Implementation](https://github.com/Visual-Agent/DeepEyes)

---
If you need further details for reproduction or encounter any issues, feel free to open an issue or contact the maintainers. 