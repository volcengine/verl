import pytest
import torch
from unittest.mock import MagicMock

# 导入需要测试的类和它依赖的数据结构
from verl.workers.fsdp_workers import RewardModelWorker
from verl import DataProto
from verl.utils import hf_tokenizer

# 使用一个真实的、轻量级的 tokenizer
TINY_TOKENIZER_ID = "Qwen/Qwen3-1.7B"

# 这是一个独立的函数，我们将用它作为测试的目标
extract_fn = RewardModelWorker._extract_inputs_for_reward

@pytest.fixture(scope="session")
def tokenizer():
    """
    创建一个全局的 tokenizer 实例，在所有测试中共享，只加载一次。
    """
    return hf_tokenizer(TINY_TOKENIZER_ID)

@pytest.fixture
def mock_self(tokenizer):
    """
    创建一个模拟的 'self' 对象，只包含 _extract_inputs_for_reward 需要的属性。
    """
    mock_self_instance = MagicMock()
    mock_self_instance.tokenizer = tokenizer
    return mock_self_instance

def test_extract_inputs_for_reward_single_item(mock_self, tokenizer):
    """
    测试 _extract_inputs_for_reward 函数处理单个批次项的能力。
    这是一个纯 CPU 测试。
    """
    # 1. 准备输入数据 (最关键的一步)
    prompt_text = "What is the capital of France?"
    response_text = "The capital of France is Paris."
    ground_truth = "Paris"

    # 将文本编码为 token ID
    prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"]
    response_ids = tokenizer(response_text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    
    # 模拟一个完整的 attention_mask
    full_attention_mask = torch.ones(1, prompt_ids.shape[1] + response_ids.shape[1])

    # `chat_history` 是 non_tensor_batch 的一部分
    chat_history = [{"role": "user", "content": prompt_text}]

    # 构造 DataProto，使用 from_dict 方法
    input_data = DataProto.from_dict(
        tensors={
            "attention_mask": full_attention_mask,
            "responses": response_ids,
            "batch_size": torch.tensor([1]),
        },
        non_tensors={
            "raw_prompt": [chat_history],
            "reward_model": [{"ground_truth": ground_truth}],
        }
    )

    # 2. 调用被测函数
    # 直接调用函数，将 mock_self 作为第一个参数，模拟 'self'
    questions, answers, ground_truths = extract_fn(mock_self, input_data)

    # 3. 断言结果
    assert isinstance(questions, list) and len(questions) == 1
    assert questions[0] == prompt_text

    assert isinstance(answers, list) and len(answers) == 1
    # decode 可能会有额外的空格，使用 .strip() 使断言更健壮
    assert answers[0].strip() == response_text.strip()

    assert isinstance(ground_truths, list) and len(ground_truths) == 1
    assert ground_truths[0] == ground_truth


def test_extract_inputs_for_reward_with_padding(mock_self, tokenizer):
    """
    测试函数在存在 padding 的情况下是否能正确解码 response。
    这是一个纯 CPU 测试。
    """
    # 1. 准备数据
    tokenizer.pad_token_id = tokenizer.eos_token_id

    prompt_text = "Question"
    response_text = "Answer"
    ground_truth = "Answer"
    
    prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"]
    response_ids_unpadded = tokenizer(response_text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    
    # 手动给 response 添加 padding
    padding_length = 5
    padding = torch.tensor([[tokenizer.pad_token_id] * padding_length])
    response_ids_padded = torch.cat([response_ids_unpadded, padding], dim=1)

    # 构造 attention_mask
    full_input_ids_shape = (1, prompt_ids.shape[1] + response_ids_padded.shape[1])
    # prompt 和 response 的非 padding 部分是 1
    valid_len = prompt_ids.shape[1] + response_ids_unpadded.shape[1]
    attention_mask = torch.zeros(full_input_ids_shape, dtype=torch.long)
    attention_mask[0, :valid_len] = 1

    chat_history = [{"role": "user", "content": prompt_text}]

    input_data = DataProto.from_dict(
        tensors={
            "attention_mask": attention_mask,
            "responses": response_ids_padded,
            "batch_size": torch.tensor([1]),
        },
        non_tensors={
            "raw_prompt": [chat_history],
            "reward_model": [{"ground_truth": ground_truth}],
        }
    )
    # 2. 调用
    questions, answers, ground_truths = extract_fn(mock_self, input_data)

    # 3. 断言
    assert answers[0].strip() == response_text.strip()
