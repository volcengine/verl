import os
import socket
import multiprocessing as mp

import torch

from omegaconf import OmegaConf
import argparse
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer



from verl.experimental.dataset.sampler import AbstractSampler
from verl.utils.import_utils import load_extern_type
from verl.trainer.ppo.ray_trainer import RolloutDataCollector
from verl.utils.model import get_generation_config, print_model_size, update_model_config

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="config for vllm rollout")
    
    
    ### vllm engieen
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",)   
    parser.add_argument("--tp", type=int, default=1,help="tensor parallel size (num GPUs)")
    parser.add_argument("--max-model-len", type=int,  default=8192)
    parser.add_argument("--gpu-mem-util",  type=float, default=0.9)
    parser.add_argument("--dtype",  default="auto",)
    parser.add_argument("--trust-remote-code", default=True)
    
    ### data
    parser.add_argument("--data_config", type=str, default="/data/hxy/verl/verl/trainer/config/data/legacy_data.yaml", help="Path to the dataset config file")
    parser.add_argument("--train_files", type=str, default="./data/gsm8k/train.parquet", help="Path to the training data files, separated by commas")
    parser.add_argument("--val_files", type=str, default="./data/gsm8k/test.parquet", help="Path to the validation data files, separated by commas")
    
    return parser.parse_args()

def run_rollout(args,llm) -> None:
    # Download the checkpoint from HDFS to the local machine.
    # `use_shm` determines whether to use shared memory, which could lead to faster model loading if turned on
    
    data_config_path = args.data_config
    from omegaconf import OmegaConf
    data_config = OmegaConf.load(str(data_config_path))
    from verl.utils.fs import copy_to_local
    local_path = copy_to_local(
        args.model, use_shm=False
        )
    # Instantiate the tokenizer and processor.
    from verl.utils import hf_processor, hf_tokenizer

    trust_remote_code = args.trust_remote_code
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
    generation_config = get_generation_config(local_path, trust_remote_code=trust_remote_code)
    # Used for multimodal LLM, could be None
    processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
    train_dataset = create_rl_dataset(args.train_files, data_config, tokenizer, processor, is_train=True)
    val_dataset = create_rl_dataset(args.val_files, data_config, tokenizer, processor, is_train=False)
    train_sampler = create_rl_sampler(data_config, train_dataset)
    trainer = RolloutDataCollector(
        config=None,
        data_config=data_config,
        generation_config=generation_config,
        tokenizer=tokenizer,
        processor=processor,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_sampler=train_sampler,
    )
    trainer.rollout_data(llm)

def create_rl_dataset(data_paths, data_config, tokenizer, processor, is_train=True):
    """Create a dataset.

    Arguments:
        data_paths: List of paths to data files.
        data_config: The data config.
        tokenizer (Tokenizer): The tokenizer.
        processor (Processor): The processor.

    Returns:
        dataset (Dataset): The dataset.
    """
    from torch.utils.data import Dataset

    from verl.utils.dataset.rl_dataset import RLHFDataset

    dataset_cls = RLHFDataset
    print(f"Using dataset class: {dataset_cls.__name__}")

    # Instantiate the dataset using the determined dataset class
    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
    )

    return dataset


def create_rl_sampler(data_config, dataset):
    """Create a sampler for the dataset.

    Arguments:
        data_config: The data config.
        dataset (Dataset): The dataset.

    Returns:
        sampler (Sampler): The sampler.
    """
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    if data_config.sampler is not None and data_config.sampler.get("class_path", None) is not None:
        curriculum_class = load_extern_type(
            data_config.sampler.class_path,
            data_config.sampler.class_name,
        )
        sampler = curriculum_class(
            data_source=dataset,
            data_config=data_config,
        )
        assert isinstance(sampler, AbstractSampler)
        assert data_config.get("dataloader_num_workers", 8) == 0, (
            "If using curriculum, num_workers must be 0 to prevent data caching. "
            "If the dataloader caches data before the batch is done the "
            "curriculum sampler won't have the opportunity to reorder it. "
        )

    # Use a sampler to facilitate checkpoint resumption.
    # If shuffling is enabled in the data configuration, create a random sampler.
    elif data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(data_config.get("seed", 1))
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        # If shuffling is disabled, use a sequential sampler to iterate through the dataset in order.
        sampler = SequentialSampler(data_source=dataset)

    return sampler


def build_llm(args):

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if visible:
        gpus = [g for g in visible.split(",") if g.strip() != ""]
        if args.tp > len(gpus):
            raise ValueError(f"--tp={args.tp} exceeds visible GPUs={len(gpus)}")
    else:
        if torch.cuda.is_available() and args.tp > torch.cuda.device_count():
            raise ValueError(f"--tp={args.tp} exceeds total GPUs={torch.cuda.device_count()}")

    print(f"Loading {args.model} with TP={args.tp} ...")
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        trust_remote_code=args.trust_remote_code,
    )
    ##### test
    # tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)

    # prompts = [
    #     tok.apply_chat_template(
    #         [{"role":"user","content":"讲讲 RMSNorm 是什么，和 LayerNorm 的区别？"}],
    #         tokenize=False, add_generation_prompt=True
    #     ),
    #     tok.apply_chat_template(
    #         [{"role":"user","content":"写一个两行的 Python 函数，返回 x 的平方。"}],
    #         tokenize=False, add_generation_prompt=True
    #     ),
    # ]

    # params = SamplingParams(max_tokens=128, temperature=0.7, top_p=0.9)
    # outputs = llm.generate(prompts, params)

    # for i, out in enumerate(outputs):
    #     print(f"\n--- Prompt {i+1} ---")
    #     print(out.outputs[0].text)
    return llm

if __name__ == "__main__":
    args = parse_args()
    llm = build_llm(args)
    run_rollout(args,llm)
