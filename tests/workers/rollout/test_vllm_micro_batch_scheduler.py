import numpy as np
import ray
import torch
from datasets import load_dataset
from omegaconf import OmegaConf

from tests.workers.rollout.async_rollout_utils import init_async_rollout_manager
from verl.protocol import DataProto


class TestVllmMicroBatchScheduler:
    def test_micro_batch_scheduler(self):
        ray.init(
            runtime_env={
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true",
                    "NCCL_DEBUG": "WARN",
                    "VLLM_LOGGING_LEVEL": "INFO",
                    "VLLM_USE_V1": "1",
                }
            }
        )

        # Load config
        config = OmegaConf.load("verl/trainer/config/ppo_trainer.yaml")
        config.actor_rollout_ref.model.path = "Qwen/Qwen3-4B"
        config.actor_rollout_ref.rollout.mode = "async"
        config.actor_rollout_ref.rollout.chat_scheduler = "examples.ppo_trainer.naive_chat_scheduler.MicroBatchChatCompletionScheduler"
        config.actor_rollout_ref.rollout.prompt_length = 8192
        config.actor_rollout_ref.rollout.response_length = 8192

        # Init sandbox and async rollout manager
        async_rollout_manager = init_async_rollout_manager(config, scheduler_kwargs={"max_inflight_req": 4})

        # Build dataset
        dataset = load_dataset("Maxwell-Jia/AIME_2024", split="train")
        prompts = DataProto(non_tensor_batch={"raw_prompt": np.array([[{"role": "user", "content": problem}] for problem in dataset["Problem"]])})
        print(f"length of data proto : {len(prompts)}")
        result = async_rollout_manager.generate_sequences(prompts=prompts)
        torch.save(result, "result.pt")
        assert len(result) == len(dataset)
