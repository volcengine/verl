import pytest
import torch
import ray
import os
from unittest.mock import patch

from verl import DataProto
from verl.workers.fsdp_workers import RewardModelWorker
from verl.utils.config import OmegaConf
from verl.single_controller.ray.base import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.workers.config.reward_model import (
    RewardModelConfig, RewardModelInnerConfig, RewardModelRolloutConfig, RewardModelFSDPConfig, ProfilerConfig
)
from verl.utils import hf_tokenizer

# Use a community-verified, minimal model
TINY_MODEL_ID = "Qwen/Qwen2.5-0.5B"

# Wrap Worker as Ray Actor and request 1 GPU
REMOTE_CLS = ray.remote(num_gpus=1)(RewardModelWorker)

@pytest.fixture(scope="module") # Make Ray cluster start once for all tests
def ray_cluster():
    """Provide a Ray cluster that starts at test beginning and shuts down at test end."""
    if not torch.cuda.is_available():
        pytest.skip("GPU not available, cannot run Ray-based GPU tests.")
    
    # RayWorkerGroup will set all required environment variables for Actors
    ray.init(num_gpus=1)
    yield
    ray.shutdown()

@pytest.fixture
def minimal_worker_config():
    """
    Create a minimal top-level configuration using RewardModelConfig that allows RewardModelWorker to run.
    """
    # Initialize config directly using RewardModelConfig constructor
    rm_config = RewardModelConfig(
        enable=True,
        rm_mode="generator",
        micro_batch_size_per_gpu=1,
        model=RewardModelInnerConfig(
            path=TINY_MODEL_ID,
            input_tokenizer=TINY_MODEL_ID,
            rollout=RewardModelRolloutConfig(
                name="vllm",
                tensor_model_parallel_size=1,
            ),
            fsdp_config=RewardModelFSDPConfig(fsdp_size=1),
            use_remove_padding=False,
        ),
        strategy='fsdp',
        use_dynamic_bsz=False,
        profiler=ProfilerConfig(),
    )

    return OmegaConf.create(rm_config)

@patch("verl.workers.fsdp_workers.FSDP", side_effect=lambda model, **kwargs: model)
def test_rm_worker_with_two_inputs(mock_fsdp, ray_cluster, minimal_worker_config):
    """
    Test case with two input data:
    1. Create RewardModelWorker in Ray Actor.
    2. Call init_model().
    3. Call compute_rm_score() and verify results for two inputs.
    """
    # --- 1. Setup and initialization ---
    resource_pool = RayResourcePool([1], use_gpu=True)
    class_with_args = RayClassWithInitArgs(cls=REMOTE_CLS, config=minimal_worker_config)
    worker_group = RayWorkerGroup(resource_pool, class_with_args)
    worker_group.execute_all_sync("init_model")
    # --- 2. Prepare two input data ---
    tokenizer = hf_tokenizer(TINY_MODEL_ID)
    max_seq_len = minimal_worker_config.model.rollout.prompt_length

    # First input
    prompt1 = "1 + 2 = ?"
    response1 = "#### 3"
    input_ids1 = tokenizer(prompt1, return_tensors="pt", padding="max_length",
                          max_length=max_seq_len, truncation=True)["input_ids"]
    response_ids1 = tokenizer(response1, return_tensors="pt", padding="max_length",
                            max_length=max_seq_len, truncation=True, add_special_tokens=False)["input_ids"]
    attention_mask1 = (input_ids1 != tokenizer.pad_token_id).long()
    position_ids1 = torch.zeros_like(input_ids1)
    non_padding_mask1 = (input_ids1 != tokenizer.pad_token_id)
    position_ids1[non_padding_mask1] = torch.arange(1, non_padding_mask1.sum() + 1, dtype=torch.long)

    # Second input
    prompt2 = "3 * 4 = ?"
    response2 = "#### 10"
    input_ids2 = tokenizer(prompt2, return_tensors="pt", padding="max_length",
                          max_length=max_seq_len, truncation=True)["input_ids"]
    response_ids2 = tokenizer(response2, return_tensors="pt", padding="max_length",
                            max_length=max_seq_len, truncation=True, add_special_tokens=False)["input_ids"]
    attention_mask2 = (input_ids2 != tokenizer.pad_token_id).long()
    position_ids2 = torch.zeros_like(input_ids2)
    non_padding_mask2 = (input_ids2 != tokenizer.pad_token_id)
    position_ids2[non_padding_mask2] = torch.arange(1, non_padding_mask2.sum() + 1, dtype=torch.long)

    # Create DataProto containing two samples
    input_data = DataProto.from_dict(
        tensors={
            "input_ids": torch.cat([input_ids1, input_ids2], dim=0),
            "attention_mask": torch.cat([attention_mask1, attention_mask2], dim=0),
            "responses": torch.cat([response_ids1, response_ids2], dim=0),
            "position_ids": torch.cat([position_ids1, position_ids2], dim=0),
        },
        non_tensors={
            "raw_prompt": [
                [{"role": "user", "content": prompt1}],
                [{"role": "user", "content": prompt2}]
            ],
            "reward_model": [
                {"ground_truth": "3"},
                {"ground_truth": "12"}
            ],
        }
    )

    # --- 3. Execute computation and verify ---
    output_protos = worker_group.compute_rm_score(input_data)
    assert isinstance(output_protos, DataProto) and len(output_protos) == 2

    # Verify returned results
    assert "rm_scores" in output_protos.batch
    scores_tensor = output_protos.batch["rm_scores"]
    assert isinstance(scores_tensor, torch.Tensor)
    assert scores_tensor.ndim == 2  # Should be (batch_size, seq_len)
    assert scores_tensor.shape[0] == 2  # batch_size=2
    # Verify first sample is close to 5, second sample is close to 0
    assert torch.isclose(scores_tensor[0].max(), torch.tensor(5.0), rtol=0.1)  # Allow 10% relative error
    assert torch.isclose(scores_tensor[1].max(), torch.tensor(0.0), atol=0.1)  # Allow 0.1 absolute error
