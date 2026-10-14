# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Rollout with vLLM, recompute old_log_probs with Megatron, and check the rollout-vs-actor log-prob gap.

Usage (8 GPUs):
    MODEL_PATH=~/models/Qwen3.5-0.8B EP=1 \
        .venv/bin/python -m pytest -s tests/experimental/agent_loop/test_megatron_vllm_logprob_diff.py
    MODEL_PATH=~/models/Qwen3.5-35B-A3B EP=8 \
        .venv/bin/python -m pytest -s tests/experimental/agent_loop/test_megatron_vllm_logprob_diff.py
    # fp8 blockwise rollout, weights quantized on the trainer side
    MODEL_PATH=~/models/Qwen3.5-0.8B EP=1 QUANT_LINEAR=FP8_BLOCK QUANT_MOE=FP8_BLOCK \
        .venv/bin/python -m pytest -s tests/experimental/agent_loop/test_megatron_vllm_logprob_diff.py
    # same, with Megatron ModelOpt QAT fake-quantizing to the rollout scheme
    MODEL_PATH=~/models/Qwen3.5-0.8B EP=1 QUANT_LINEAR=FP8_BLOCK QUANT_MOE=FP8_BLOCK QAT_MODE=fp8 \
        .venv/bin/python -m pytest -s tests/experimental/agent_loop/test_megatron_vllm_logprob_diff.py
"""

import json
import os

import numpy as np
import pytest
import ray
import torch
from omegaconf import DictConfig
from hydra import compose, initialize_config_dir

from tests.experimental.agent_loop.agent_utils import init_agent_loop_manager_and_workers
from verl.protocol import DataProto
from verl.trainer.ppo.rollout_corr_helper import compute_offpolicy_metrics
from verl.utils import tensordict_utils as tu
from verl.utils.debug.metrics import calculate_debug_metrics
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

TP = int(os.getenv("TP", 2))
PP = int(os.getenv("PP", 1))
CP = int(os.getenv("CP", 1))
EP = int(os.getenv("EP", int(os.environ.get("EP", 1))))
ETP = int(os.getenv("ETP", 1))
GEN_TP = int(os.getenv("GEN_TP", 2))
N_GPUS = int(os.getenv("N_GPUS", 8))
MAX_K3_KL = float(os.getenv("MAX_K3_KL", 1e-2))
# Rollout compressed-tensors presets, e.g. QUANT_LINEAR=FP8_BLOCK QUANT_MOE=FP8_BLOCK; unset keeps the rollout in bf16.
QUANT_LINEAR = os.getenv("QUANT_LINEAR")
QUANT_MOE = os.getenv("QUANT_MOE")
# Megatron ModelOpt QAT mode matching the rollout scheme, e.g. QAT_MODE=fp8; unset disables QAT.
QAT_MODE = os.getenv("QAT_MODE")


@pytest.fixture
def init_config() -> DictConfig:

    with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config")):
        config = compose(
            config_name="ppo_trainer",
            overrides=[
                "model_engine=megatron",
                f"trainer.n_gpus_per_node={N_GPUS}",
                "trainer.nnodes=1",
                "actor_rollout_ref.actor.optim.total_training_steps=1",
                "actor_rollout_ref.actor.use_dynamic_bsz=True",
                "actor_rollout_ref.actor.megatron.use_remove_padding=True",
                f"actor_rollout_ref.actor.megatron.tensor_model_parallel_size={TP}",
                f"actor_rollout_ref.actor.megatron.pipeline_model_parallel_size={PP}",
                f"actor_rollout_ref.actor.megatron.context_parallel_size={CP}",
                f"actor_rollout_ref.actor.megatron.expert_model_parallel_size={EP}",
                f"actor_rollout_ref.actor.megatron.expert_tensor_parallel_size={ETP}",
                "actor_rollout_ref.actor.megatron.param_offload=True",
                "actor_rollout_ref.actor.megatron.optimizer_offload=True",
                "++actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=auto",
                "+actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True",
                "+actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True",
                "actor_rollout_ref.rollout.name=vllm",
                "actor_rollout_ref.rollout.mode=async",
                f"actor_rollout_ref.rollout.tensor_model_parallel_size={GEN_TP}",
                "actor_rollout_ref.rollout.gpu_memory_utilization=0.6",
                "actor_rollout_ref.rollout.prompt_length=1024",
                "actor_rollout_ref.rollout.response_length=2048",
                "actor_rollout_ref.rollout.n=4",
                "actor_rollout_ref.rollout.agent.num_workers=2",
                "actor_rollout_ref.rollout.skip_tokenizer_init=True",
            ],
        )

    config.actor_rollout_ref.model.path = os.path.expanduser(os.getenv("MODEL_PATH", "~/models/Qwen3.5-35B-A3B"))
    if QUANT_LINEAR or QUANT_MOE:
        config.actor_rollout_ref.rollout.quantization = {"linear": QUANT_LINEAR, "moe": QUANT_MOE, "ignore": []}
    if QAT_MODE:
        config.actor_rollout_ref.actor.megatron.qat.enable = True
        config.actor_rollout_ref.actor.megatron.qat.mode = QAT_MODE
    return config


def compute_old_log_prob(actor_rollout_wg, batch: DataProto) -> DataProto:
    """Same as RayPPOTrainer._compute_old_log_prob."""
    batch_td = left_right_2_no_padding(batch.to_tensordict())
    tu.assign_non_tensor(batch_td, calculate_entropy=True, compute_loss=False)
    output = actor_rollout_wg.compute_log_prob(batch_td)
    log_probs = no_padding_2_padding(tu.get(output, "log_probs"), batch_td)
    entropy = no_padding_2_padding(tu.get(output, "entropy"), batch_td)
    return DataProto.from_tensordict(
        tu.get_tensordict({"old_log_probs": log_probs.float(), "entropys": entropy.float()})
    )


def test_megatron_vllm_logprob_diff(init_config):
    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
                "VLLM_USE_V1": "1",
                "VLLM_ALLREDUCE_USE_SYMM_MEM": "0",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            }
        }
    )

    agent_loop_manager, actor_rollout_wg, checkpoint_manager = init_agent_loop_manager_and_workers(init_config)

    questions = [
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
        "How many clips did Natalia sell altogether in April and May?",
        "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. "
        "How much did she earn?",
        "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. "
        "Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. "
        "How much more money does Betty need to buy the wallet?",
        "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
        "Write a short poem about the ocean.",
        "Explain the difference between TCP and UDP in a few sentences.",
        "Let's play a role playing game. Your name is Alice, your favorite color is blue. Introduce yourself.",
        "What is the derivative of x^3 * sin(x)?",
    ]
    raw_prompts = [[{"role": "user", "content": q}] for q in questions]
    batch = DataProto(
        non_tensor_batch={
            "raw_prompt": np.array(raw_prompts),
            "agent_name": np.array(["single_turn_agent"] * len(raw_prompts)),
            "data_source": np.array(["openai/gsm8k"] * len(raw_prompts)),
            "reward_model": np.array([{"style": "rule", "ground_truth": "1.0"}] * len(raw_prompts)),
        },
    )
    n = init_config.actor_rollout_ref.rollout.n
    batch = batch.repeat(n)

    # rollout
    result = agent_loop_manager.generate_sequences(prompts=batch)
    checkpoint_manager.sleep_replicas()
    assert len(result) == len(raw_prompts) * n
    assert "rollout_log_probs" in result.batch
    result.meta_info.pop("timing", None)

    # recompute old_log_probs with megatron
    result.meta_info["temperature"] = init_config.actor_rollout_ref.rollout.temperature
    result.meta_info["global_token_num"] = torch.sum(result.batch["attention_mask"], dim=-1).tolist()
    old_log_prob = compute_old_log_prob(actor_rollout_wg, result)
    old_log_prob.batch.pop("entropys")
    result = result.union(old_log_prob)

    old_log_probs = result.batch["old_log_probs"]
    rollout_log_probs = result.batch["rollout_log_probs"]
    response_mask = result.batch["response_mask"].bool()
    assert old_log_probs.shape == rollout_log_probs.shape == response_mask.shape

    metrics = compute_offpolicy_metrics(
        old_log_prob=old_log_probs, rollout_log_prob=rollout_log_probs, response_mask=response_mask
    )
    metrics.update(calculate_debug_metrics(result))
    log_prob_abs_diff = (old_log_probs - rollout_log_probs).abs()[response_mask]
    metrics["log_prob_abs_diff_mean"] = log_prob_abs_diff.mean().item()
    metrics["log_prob_abs_diff_max"] = log_prob_abs_diff.max().item()
    metrics["num_response_tokens"] = int(response_mask.sum().item())
    print(json.dumps(metrics, indent=2))

    assert np.isfinite(metrics["kl"]) and np.isfinite(metrics["k3_kl"])
    assert metrics["k3_kl"] < MAX_K3_KL, f"k3_kl {metrics['k3_kl']} exceeds {MAX_K3_KL}"

    print("Test passed!")
    ray.shutdown()
