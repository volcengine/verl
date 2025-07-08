# DeepEyes: Incentivizing "Thinking with Images" via Reinforcement Learning

This directory contains the implementation for reproducing the DeepEyes paper within the verl framework, supporting multi-turn visual tool calls. This implementation is based on the original [DeepEyes paper](https://arxiv.org/abs/2505.14362) and its [official implementation](https://github.com/Visual-Agent/DeepEyes), integrated with the multi-modal and multi-turn capabilities of the verl framework.

## Reproduce the Experiment

TODO: add results details here.

```bash
export WANDB_API_KEY=<YOUR_WANDB_API_KEY>

python3 recipe/deepeyes/deepeyes47k_preprocess.py --dataset_dir <your_local_data_directory> --save_dir <directory_to_save_processed_data>

bash recipe/deepeyes/run_deepeyes_grpo.sh
```

## References and Acknowledgements

- [DeepEyes Paper](https://arxiv.org/abs/2505.14362)
- [DeepEyes Official Implementation](https://github.com/Visual-Agent/DeepEyes)

---
If you need further details for reproduction or encounter any issues, feel free to open an issue or contact the maintainers. 