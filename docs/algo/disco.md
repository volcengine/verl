# Recipe: Discriminative Constrained Optimization(DisCO)

Last updated: 09/04/2025.


 📝 [Paper@arXiv](https://arxiv.org/abs/2505.12366) | 🏠 [Repo@GitHub](https://github.com/Optimization-AI/DisCO) | 🤗 [Models@HF](https://huggingface.co/ganglii)
### 💡 Introducing **DisCO** — *Discriminative Constrained Optimization*

**DisCO** is a new RL framework grounded in **discriminative learning**. It trains models by **increasing scores for positive answers while decreasing those for negatives**, enabling:

* ❌ **No Early Entropy Collapse**
* ⚡ **Faster convergence**
* 📉 **More stable training**
* ⚖️ **Handles sparse rewards** – robust to imbalanced data with advanced discriminative approaches

---

### 📈 Quick Results

On six math reasoning benchmarks with a 1.5B model, **DisCO outperforms GRPO and its variants**:

* **+7% vs GRPO**
* **+6% vs DAPO**

**DisCO with 8k response length is on par with or even better than GRPO with 32k response length**

---


## Quickstart

1. Prepare the datasets:

```bash
bash prepare_data.sh # This downloads the datasets to current folder
```

2. Run script:

```bash
cd verl # Repo root
bash recipe/disco/run_disco_1.5b.sh # or other scripts
```

## Configuration

To configure DisCO within the framework, use the following YAML settings. 

```yaml
algorithm:
  adv_estimator: disco  # Use disco dummy advantage function 
actor_rollout_ref:
  actor:
    policy_loss:
      loss_mode: 'disco'
      score_func: 'logL' # score function used in disco. Options: 'logL', 'Lratio'
      delta: 1e-4    
      beta: 1e3     
      tau: 10   # tau=10 is recommended for 'logL',  tau=1 is recommended for 'Lratio'
trainer:
  # We input all responses to a given question in one forward for better performance. 
  # Specifically, it better to have:
  #     (ppo_micro_batch_size_per_gpu * nnodes * n_gpus_per_node) % rollout.n = 0
  balance_batch: False 
```


## More Results

Comparison with baseline models and baseline methods for fine-tuning 1.5B models. OpenAI-o1-preview is included as a reference.  MRL denotes Max Response Length utilized in training/testing. The shaded models are trained by other works and the shaded numbers are reported in their original works or in DeepScalaR. All other results are either evaluated on existing models or on the models trained by us using  different approaches. Methods in the bottom area are all for fine-tuning  DeepSeek-R1-Distill-Qwen-1.5B model on the same DeepScaleR dataset. DS is short for DeepSeek-R1, DSR is short for DeepScalaR.

<p align="center"><img alt="Comparison with baselines on 1.5B model" src="https://github.com/Optimization-AI/DisCO/blob/main/assets/1p5model.png" width="800"/></p>


Comparison with baseline models and baseline methods for fine-tuning 7B models. Methods in the bottom area are all for fine-tuning  DeepSeek-R1-Distill-Qwen-7B model on the the same DeepScalaR dataset.

<p align="center"><img alt="Comparison with baselines on 7B model" src="https://github.com/Optimization-AI/DisCO/blob/main/assets/7Bmodel.png" width="800"/></p>

Training dynamics of different methods: left two are for fine-tuning 1.5B model and right two are for fine-tuning 7B model. (a), (c) plot the training reward (averaged over generated outputs for questions used in each step) vs the number of training steps; (b), (d) plot the generation entropy vs training steps.

<p align="center"><img alt="Training Dynamics" src="https://github.com/Optimization-AI/DisCO/blob/main/assets/training-dyanmics.png" width="800"/></p>


## Citing DisCO

If you find DisCO useful in your research, please consider citing the following paper:
```bibtex
@article{li2025disco,
  title={DisCO: Reinforcing Large Reasoning Models with Discriminative Constrained Optimization},
  author={Li, Gang and Lin, Ming and Galanti, Tomer and Tu, Zhengzhong and Yang, Tianbao},
  journal={arXiv preprint arXiv:2505.12366},
  year={2025}
}
```


