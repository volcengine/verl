import os

import pytest
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl.protocol import DataProto
from verl.workers.deepspeed_workers import ActorRolloutRefWorker
from verl.workers.sharding_manager.deepspeed_vllm import DeepSpeedVLLMShardingManager


def _build_real_config():
    return OmegaConf.create(
        {
            "nccl_timeout": 600,
            "model": {
                "path": "Qwen/Qwen2.5-0.5B-Instruct",
                "use_remove_padding": True,
                "enable_gradient_checkpointing": False,
            },
            "rollout": {
                "name": "vllm",
                "mode": "sync",
                "n": 1,
                "seed": 42,
                "tensor_model_parallel_size": 1,
                "data_parallel_size": 1,
                "pipeline_model_parallel_size": 1,
                "gpu_memory_utilization": 0.3,
                "enforce_eager": False,
                "dtype": "bfloat16",
                "load_format": "safetensors",
                "free_cache_engine": False,
                "max_model_len": 1024,
                "max_num_batched_tokens": 2048,
                "max_num_seqs": 8,
                "enable_chunked_prefill": False,
                "log_prob_micro_batch_size_per_gpu": 1,
            },
        }
    )


def test_deepspeed_sharding_manager_single_gpu_real():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required for real DeepSpeed sharding test.")
    if not os.environ.get("RUN_DEEPSPEED_REAL"):
        pytest.skip("Set RUN_DEEPSPEED_REAL=1 to enable the heavy DeepSpeed sharding test.")

    pytest.importorskip("vllm")

    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method=os.environ.get("DIST_INIT_METHOD", "tcp://127.0.0.1:29500"),
            rank=int(os.environ["RANK"]),
            world_size=int(os.environ["WORLD_SIZE"]),
        )

    config = _build_real_config()
    worker = ActorRolloutRefWorker(config=config, role="rollout")

    try:
        worker.init_model()
        assert isinstance(worker.rollout_sharding_manager, DeepSpeedVLLMShardingManager)

        tokenizer = worker.tokenizer
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        prompt_text = "你好，世界！请用一句话介绍你自己。"
        encoded = tokenizer(prompt_text, return_tensors="pt", padding=True).to(device)

        seq_len = encoded["input_ids"].shape[1]
        position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)

        batch = TensorDict(
            {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "position_ids": position_ids,
            },
            batch_size=[encoded["input_ids"].shape[0]],
        )

        data_proto = DataProto(
            batch=batch,
            meta_info={
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
            },
        )

        output = worker.generate_sequences(data_proto)
        assert "responses" in output.batch
        assert output.batch["responses"].shape[0] == 1

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
