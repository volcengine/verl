import numpy as np
import torch
from transformers import AutoTokenizer

from verl import DataProto
from verl.workers.reward_manager import PrimeRewardManager


def compute_score(data_sources: str, solution_strs: str, ground_truths, extra_infos: dict):
    return len(solution_strs)


def test_prime_reward_manager():
    local_model_path = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, padding_side="left")
    data = DataProto.from_single_dict(
        {
            "prompts": torch.tensor([[1, 2, 3], [7, 8, 9]]),
            "responses": torch.tensor([[4, 5, 6], [7, 8, 9]]),
            "attention_mask": torch.tensor([[0, 1, 1, 1, 1, 0], [0, 1, 1, 1, 0, 0]]),
            "reward_model": np.array([{}, {}], dtype=object),
            "data_source": np.array(["source1", "source2"], dtype=object),
        }
    )

    manager = PrimeRewardManager(tokenizer, 0, compute_score, return_dict=False)
    reward = manager(data)
    assert reward[0][1].item() == 2 and reward[1][0].item() == 1, f"Wrong reward_tensor: {reward}"


if __name__ == "__main__":
    test_prime_reward_manager()
