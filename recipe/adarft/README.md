<h1 style="text-align: center;">AdaRFT: Adaptive Curriculum Reinforcement Finetuning</h1>

📢 **New extension to `verl`!** We propose an adaptive curriculum learning method for efficient and scalable reinforcement finetuning (RFT) of LLMs — now implemented as an extension to this repo.

> **Efficient Reinforcement Finetuning via Adaptive Curriculum Learning**  
> Taiwei Shi†, Yiyang Wu†, Linxin Song†, Tianyi Zhou▽, Jieyu Zhao†  
> †University of Southern California, ▽University of Maryland  
> [[Paper]](https://arxiv.org/abs/2504.05520)

### 🧠 Highlights
- Dynamically adapts training difficulty using a lightweight curriculum scheduler
- Compatible with standard RFT algorithms like PPO, GRPO, REINFORCE++
- Improves both **sample efficiency** and **final accuracy** on math reasoning benchmarks
- Up to **2× faster convergence** vs PPO baseline
- Seamlessly integrated into `verl` without modifying reward functions or model architectures

### 📦 Preprocessed Data
- **Difficulty annotations**: [DeepScaleR](https://huggingface.co/datasets/lime-nlp/DeepScaleR_Difficulty)
- **Training data**: [verl/data](https://github.com/uscnlp-lime/verl/tree/main/verl/data)

### 🚀 Usage
To use AdaRFT, you can simply use our example [script](https://github.com/uscnlp-lime/verl/blob/main/examples/adarft/run_qwen2.5-1.5b_seq_balance.sh). 

You can also enable it in [ppo_trainer.yaml](https://github.com/uscnlp-lime/verl/blob/main/verl/trainer/config/ppo_trainer.yaml#L18-L24) or via command-line by setting the following flags:

```bash
python3 -m verl.trainer.main_ppo \
    ... \
    data.adarft.enable=True \
    data.adarft.beta=0.5 \       # Target reward (success rate) the model aims to maintain
    data.adarft.alpha=2 \        # Sensitivity of difficulty updates based on reward difference
    data.adarft.eta=50 \         # Step size to scale reward signal to difficulty space
    data.adarft.d_min=0 \        # Minimum difficulty bound
    data.adarft.d_max=100 \      # Maximum difficulty bound
    ...
```

Make sure your dataset includes difficulty scores (e.g., from [here](https://github.com/uscnlp-lime/verl/tree/main/verl/data)) for AdaRFT to function properly.

### 📚 Citation
✉️ Feel free to reach out to **Taiwei Shi (taiweish@usc.edu)** or **Jieyu Zhao (jieyuz@usc.edu)** with questions or collaborations!

```bibtex
@misc{shi2025efficient,
    title={Efficient Reinforcement Finetuning via Adaptive Curriculum Learning},
    author={Taiwei Shi and Yiyang Wu and Linxin Song and Tianyi Zhou and Jieyu Zhao},
    year={2025},
    eprint={2504.05520},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```