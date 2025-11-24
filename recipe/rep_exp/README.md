<div align="center">

# Representation-Based Exploration for Language Models: <br> From Test-Time to Post-Training

[📄 arXiv](https://arxiv.org/abs/2510.11686) &nbsp; &nbsp; [🌐 Website](https://rep-exp.github.io) &nbsp; &nbsp; [🐦 Twitter / X ](https://x.com/JensTuyls/status/1978244454617128993)

</div>

## Installation 🔌

Besides the base verl installation, which you can find [here](https://verl.readthedocs.io/en/latest/start/install.html), the only package to install is scikit-learn.
```bash
pip install scikit-learn
```

## Running the Experiments 🚀

You can reproduce or extend our experiments by running the following commands:

```bash
# General format
sh recipe/rep_exp/train_elliptical.sh $TASK $SPARSE_DIM $BETA $SEED

# MATH
sh recipe/rep_exp/train_elliptical.sh math 32 0.01 42

# GSM8K
sh recipe/rep_exp/train_elliptical.sh gsm8k 32 0.01 42

# DAPO-WITH-AIME
sh recipe/rep_exp/train_elliptical.sh dapo-with-aime24 128 0.01 42
```
where `$TASK` is the task name, `$SPARSE_DIM` is the sparse dimension, `$BETA` is the beta parameter, and `$SEED` is the seed.

## Evaluation 📊
Once done training, you can evaluate the model on the test set by following two steps.
1. Merge the model checkpoint. 

This is necessary because the model checkpoint is saved in multiple shards (depending on the nubmer of GPUs), and we need to merge them into a single checkpoint.

```bash
sh recipe/rep_exp/model_merge.sh /path/to/global_step_X/actor # where X is the global step of the checkpoint with the best pass@1 on dev
```

2. Evaluate the merged model.

```bash
sh recipe/rep_exp/eval.sh $TASK /path/to/global_step_X/actor/hf #where X is the global step of the checkpoint with the best pass@1 on dev
```

The results should be in a folder named `eval` and saved as a JSON file.

## Citation 📝

```bibtex
@article{tuyls2025representation,
  title={Representation-Based Exploration for Language Models: From Test-Time to Post-Training},
  author={Tuyls, Jens and Foster, Dylan J and Krishnamurthy, Akshay and Ash, Jordan T},
  journal={arXiv preprint arXiv:2510.11686},
  year={2025}
}
```

## Contact 📬

If you have any questions or suggestions, feel free to reach out at [jtuyls@princeton.edu](mailto:jtuyls@princeton.edu).