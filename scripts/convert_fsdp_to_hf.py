#!/usr/bin/env python
# encoding: utf-8
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch
import fire
from collections import defaultdict


def main(
    fsdp_checkpoint_path, huggingface_model_path, output_path, world_size=32
):
    """
    Convert FSDP checkpoint to HuggingFace checkpoint

    Args:
        fsdp_checkpoint_path: path to the FSDP checkpoint
        huggingface_model_path: path to the HuggingFace model
        output_path: path to save the converted checkpoint
    Usage:
        python convert_fsdp_to_hf.py \
            checkpoints/global_step_10/actor/ \
            checkpoints/global_step_10/actor/huggingface/ \
            qwen2.5_7b_rl_step10
    """
    state_dict = defaultdict(list)

    for rank in range(world_size):
        filepath = f"{fsdp_checkpoint_path}/model_world_size_{world_size}_rank_{rank}.pt"
        print("loading", filepath)
        this_state_dict = torch.load(filepath)
        for key, value in this_state_dict.items():
            state_dict[key].append(value.to_local())

    for key in state_dict:
        state_dict[key] = torch.cat(state_dict[key], dim=0)

    config = AutoConfig.from_pretrained(huggingface_model_path)
    model = AutoModelForCausalLM.from_config(config)
    model.load_state_dict(state_dict)

    model.save_pretrained(output_path, max_shard_size="10GB")

    tokenizer = AutoTokenizer.from_pretrained(huggingface_model_path)
    tokenizer.save_pretrained(output_path)


if __name__ == "__main__":
    fire.Fire(main)

