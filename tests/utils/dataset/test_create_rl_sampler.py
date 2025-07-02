"""
test create_rl_sampler
"""

from omegaconf import OmegaConf
from torch.utils.data import Dataset

from verl.trainer.main_ppo import create_rl_sampler


class FakeChatDataset(Dataset):
    def __init__(self):
        self.data = [
            {"prompt": "What's your name?", "response": "My name is Assistant."},
            {"prompt": "How are you?", "response": "I'm doing well, thank you."},
            {"prompt": "What is the capital of France?", "response": "Paris."},
            {
                "prompt": "Tell me a joke.",
                "response": "Why did the chicken cross the road? To get to the other side!",
            },
            {"prompt": "What is 2+2?", "response": "4"},
        ]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def test_multiturn_sft_dataset():
    print("Starting test...")
    data_config = OmegaConf.create(
        {
            "curriculum": {
                "curriculum_class_path": "verl.utils.dataset.curriculum_sampler",
                "curriculum_class": "RandomCurriculumSampler",
            }
        }
    )

    dataset = FakeChatDataset()
    create_rl_sampler(data_config, dataset)
