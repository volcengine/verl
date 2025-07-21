# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

"""
Replace DataProto with raw TensorDict
"""
import copy
import random

import numpy as np
import pytest
import torch

from verl.utils import tensordict_utils as tu


def test_union_tensor_dict():
    obs = torch.randn(100, 10)

    meta_info1 = {"top_p": 0.8}
    meta_info2 = {"top_p": 0.9}
    data1 = {"obs": obs, "act": torch.randn(100, 3), "data_sources": ["gsm8k"] * 100}
    data2 = {"obs": obs, "next_obs": torch.randn(100, 10), "rew": torch.randn(100), "data_sources": ["gsm8k"] * 100}

    data_with_copied_obs = {"obs": obs.clone(), "next_obs": torch.randn(100, 10), "rew": torch.randn(100)}

    data1 = tu.get_tensordict(tensor_dict=data1)
    data2 = tu.get_tensordict(tensor_dict=data2)
    data_with_copied_obs = tu.get_tensordict(data_with_copied_obs)

    data = tu.union_tensor_dict(data1, data2)
    with pytest.raises(AssertionError):
        # conflict in tensor values
        data = tu.union_tensor_dict(data1, data_with_copied_obs)

    data1 = tu.assign_non_tensor_dict(data1, meta_info1)
    data = tu.union_tensor_dict(data1, data2)  # works ok

    data2 = tu.assign_non_tensor_dict(data2, meta_info2)

    with pytest.raises(AssertionError):
        # conflict in NonTensorData
        data = tu.union_tensor_dict(data1, data2)

    data1.pop("top_p")
    data2.pop("top_p")

    data = tu.union_tensor_dict(data1, data2)

    data2["data_sources"][0] = "math"
    with pytest.raises(AssertionError):
        # conflict in NonTensorData
        data = tu.union_tensor_dict(data1, data2)


def test_tensor_dict_constructor():
    obs = torch.ones(100, 10)
    act = torch.zeros(100, 10, 3)
    data_source = ["gsm8k"] * 100
    non_tensor_dict = {'name': "abdce"}

    data = tu.get_tensordict(tensor_dict={"obs": obs, "act": act, "data_source": data_source},
                             non_tensor_dict=non_tensor_dict)


    assert data.batch_size == torch.Size([100])

    # test slicing
    assert torch.all(torch.eq(data[0]["obs"], torch.ones(10))).item()
    assert torch.all(torch.eq(data[0]["act"], torch.zeros(10, 3))).item()
    assert data[0]["data_source"] == "gsm8k"

    assert torch.all(torch.eq(data[0:2]["obs"], torch.ones(2, 10))).item()
    assert torch.all(torch.eq(data[0:2]["act"], torch.zeros(2, 10, 3))).item()
    assert data[0:2]["data_source"] == ["gsm8k"] * 2

    # test non tensor data
    data["name"] == "abdce"

def test_tensordict_with_images():
    # each sample contains a sequence with multiple images of different sizes
    vocab_size = 128
    a = torch.randint(low=0, high=vocab_size, size=(11,))
    b = torch.randint(low=0, high=vocab_size, size=(13,))
    input_ids = [a, b]
    input_ids = torch.nested.as_nested_tensor(input_ids, layout=torch.jagged)

    # must be numpy
    # TODO(vermouth1992). We may use nested tensor too. But this requires nested over nested
    a_images = [torch.randint(low=0, high=255, size=(3, 256, 256), dtype=torch.uint8).numpy(),
                torch.randint(low=0, high=255, size=(3, 128, 128), dtype=torch.uint8).numpy()]
    b_images = [torch.randint(low=0, high=255, size=(3, 256, 256), dtype=torch.uint8).numpy(),
                torch.randint(low=0, high=255, size=(3, 128, 128), dtype=torch.uint8).numpy(),
                torch.randint(low=0, high=255, size=(3, 64, 64), dtype=torch.uint8).numpy()]

    images = [a_images, b_images]

    data = tu.get_tensordict({"input_ids": input_ids, "images": images})

    assert np.all(np.equal(data[0]["images"][0], a_images[0]))
    assert torch.all(torch.eq(data[0]["input_ids"], a))


def test_tensordict_with_packing():
    vocab_size = 128
    a = torch.randint(low=0, high=vocab_size, size=(11,))
    b = torch.randint(low=0, high=vocab_size, size=(13,))
    input_ids = [a, b]
    input_ids = torch.nested.as_nested_tensor(input_ids, layout=torch.jagged)

    data = tu.get_tensordict({"input_ids": input_ids})

    # test cu_seqlens
    cu_seqlens = torch.tensor([0, 11, 24])
    assert torch.all(torch.eq(cu_seqlens, data["input_ids"].offsets()))

    # test index
    assert torch.all(torch.eq(data["input_ids"][0], a))
    assert torch.all(torch.eq(data["input_ids"][1], b))

    assert torch.all(torch.eq(data[0]["input_ids"], a))
    assert torch.all(torch.eq(data[1]["input_ids"], b))

    data_lst = data.chunk(2)

    assert torch.all(torch.eq(data_lst[0]["input_ids"][0], a))
    assert torch.all(torch.eq(data_lst[1]["input_ids"][0], b))


def test_tensordict_eq():
    obs = torch.tensor([1, 2, 3, 4, 5, 6])
    data_sources = ["abc", "def", "abc", "def", "pol", "klj"]
    non_tensor_dict = {"train_sample_kwargs": {"top_p": 1.0},
                       "val_sample_kwargs": {"top_p": 0.7}}
    data = tu.get_tensordict({"obs": obs, "data_sources": data_sources}, non_tensor_dict=non_tensor_dict)

    obs = torch.tensor([1, 2, 3, 4, 5, 6])
    data_sources = ["abc", "def", "abc", "def", "pol", "klj"]
    non_tensor_dict = {"train_sample_kwargs": {"top_p": 1.0},
                       "val_sample_kwargs": {"top_p": 0.7}}
    data1 = tu.get_tensordict({"obs": obs, "data_sources": data_sources}, non_tensor_dict=non_tensor_dict)

    tu.assert_tensordict_eq(data, data1)

    data2 = copy.deepcopy(data1)
    data2["obs"][0] += 1

    with pytest.raises(AssertionError):
        tu.assert_tensordict_eq(data, data2)

    data2 = copy.deepcopy(data1)
    data2["data_sources"][0] = 'math'

    with pytest.raises(AssertionError):
        tu.assert_tensordict_eq(data, data2)

    data2 = copy.deepcopy(data1)
    data2["train_sample_kwargs"]['top_p'] = 0.9

    with pytest.raises(AssertionError):
        tu.assert_tensordict_eq(data, data2)



def test_tensor_dict_make_iterator():
    obs = torch.tensor([1, 2, 3, 4, 5, 6])
    data_sources = ["abc", "def", "abc", "def", "pol", "klj"]
    non_tensor_dict = {"train_sample_kwargs": {"top_p": 1.0},
                       "val_sample_kwargs": {"top_p": 0.7}}
    dataset = tu.get_tensordict({"obs": obs, "data_sources": data_sources}, non_tensor_dict=non_tensor_dict)

    dataloader = tu.make_iterator(dataset, mini_batch_size=2, epochs=2, seed=0, dataloader_kwargs={"shuffle": False, "drop_last": False})

    expected_tensor_dict = [dataset[0:2], dataset[2:4], dataset[4:6], dataset[0:2], dataset[2:4], dataset[4:6]]

    i = 0

    for d in dataloader:
        tu.assert_tensordict_eq(d, expected_tensor_dict[i])
        i += 1

    data_iter_1 = tu.make_iterator(dataset, mini_batch_size=3, epochs=1, seed=1, dataloader_kwargs={"shuffle": True})
    data_list_1 = []
    for data in data_iter_1:
        data_list_1.append(data)

    data_iter_2 = tu.make_iterator(dataset, mini_batch_size=3, epochs=1, seed=1, dataloader_kwargs={"shuffle": True})
    data_list_2 = []
    for data in data_iter_2:
        data_list_2.append(data)

    for data1, data2 in zip(data_list_1, data_list_2, strict=True):
        tu.assert_tensordict_eq(data1, data2)


def test_reorder():
    obs = torch.tensor([1, 2, 3, 4, 5, 6])
    labels = ["a", "b", "c", "d", "e", "f"]
    non_tensor_dict = {'name': "abdce"}

    data = tu.get_tensordict(tensor_dict={"obs": obs, "labels": labels}, non_tensor_dict=non_tensor_dict)
    data = data[torch.tensor([3, 4, 2, 0, 1, 5])]

    assert torch.all(torch.eq(data["obs"], torch.tensor([4, 5, 3, 1, 2, 6])))
    assert np.all(data["labels"] == np.array(["d", "e", "c", "a", "b", "f"]))
    assert data["name"] == "abdce"


def test_chunk_concat():
    obs = torch.tensor([1, 2, 3, 4, 5, 6])
    labels = ["a", "b", "c", "d", "e", "f"]
    data = DataProto.from_dict(tensors={"obs": obs}, non_tensors={"labels": labels}, meta_info={"name": "abdce"})

    with pytest.raises(AssertionError):
        data.chunk(5)

    data_split = data.chunk(2)
    assert len(data_split) == 2
    assert torch.all(torch.eq(data_split[0].batch["obs"], torch.tensor([1, 2, 3])))
    assert np.all(data_split[0].non_tensor_batch["labels"] == np.array(["a", "b", "c"]))
    assert data_split[0].meta_info == {"name": "abdce"}

    assert torch.all(torch.eq(data_split[1].batch["obs"], torch.tensor([4, 5, 6])))
    assert np.all(data_split[1].non_tensor_batch["labels"] == np.array(["d", "e", "f"]))
    assert data_split[1].meta_info == {"name": "abdce"}

    concat_data = DataProto.concat(data_split)
    assert torch.all(torch.eq(concat_data.batch["obs"], data.batch["obs"]))
    assert np.all(concat_data.non_tensor_batch["labels"] == data.non_tensor_batch["labels"])
    assert concat_data.meta_info == data.meta_info


def test_pop():
    obs = torch.randn(100, 10)
    act = torch.randn(100, 3)
    dataset = DataProto.from_dict({"obs": obs, "act": act}, meta_info={"2": 2, "1": 1})
    poped_dataset = dataset.pop(batch_keys=["obs"], meta_info_keys=["2"])

    assert poped_dataset.batch.keys() == {"obs"}
    assert poped_dataset.meta_info.keys() == {"2"}

    assert dataset.batch.keys() == {"act"}
    assert dataset.meta_info.keys() == {"1"}


def test_repeat():
    # Create a DataProto object with some batch and non-tensor data
    obs = torch.tensor([[1, 2], [3, 4], [5, 6]])
    labels = ["a", "b", "c"]
    data = DataProto.from_dict(tensors={"obs": obs}, non_tensors={"labels": labels}, meta_info={"info": "test_info"})

    # Test interleave=True
    repeated_data_interleave = data.repeat(repeat_times=2, interleave=True)
    expected_obs_interleave = torch.tensor([[1, 2], [1, 2], [3, 4], [3, 4], [5, 6], [5, 6]])
    expected_labels_interleave = ["a", "a", "b", "b", "c", "c"]

    assert torch.all(torch.eq(repeated_data_interleave.batch["obs"], expected_obs_interleave))
    assert (repeated_data_interleave.non_tensor_batch["labels"] == expected_labels_interleave).all()
    assert repeated_data_interleave.meta_info == {"info": "test_info"}

    # Test interleave=False
    repeated_data_no_interleave = data.repeat(repeat_times=2, interleave=False)
    expected_obs_no_interleave = torch.tensor([[1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6]])
    expected_labels_no_interleave = ["a", "b", "c", "a", "b", "c"]

    assert torch.all(torch.eq(repeated_data_no_interleave.batch["obs"], expected_obs_no_interleave))
    assert (repeated_data_no_interleave.non_tensor_batch["labels"] == expected_labels_no_interleave).all()
    assert repeated_data_no_interleave.meta_info == {"info": "test_info"}


def test_dataproto_pad_unpad():
    obs = torch.tensor([[1, 2], [3, 4], [5, 6]])
    labels = ["a", "b", "c"]
    data = DataProto.from_dict(tensors={"obs": obs}, non_tensors={"labels": labels}, meta_info={"info": "test_info"})

    from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

    padded_data, pad_size = pad_dataproto_to_divisor(data, size_divisor=2)
    assert pad_size == 1

    expected_obs = torch.tensor([[1, 2], [3, 4], [5, 6], [1, 2]])
    expected_labels = ["a", "b", "c", "a"]

    assert torch.all(torch.eq(padded_data.batch["obs"], expected_obs))
    assert (padded_data.non_tensor_batch["labels"] == expected_labels).all()
    assert padded_data.meta_info == {"info": "test_info"}

    unpadd_data = unpad_dataproto(padded_data, pad_size=pad_size)
    assert torch.all(torch.eq(unpadd_data.batch["obs"], obs))
    assert (unpadd_data.non_tensor_batch["labels"] == labels).all()
    assert unpadd_data.meta_info == {"info": "test_info"}

    padded_data, pad_size = pad_dataproto_to_divisor(data, size_divisor=3)
    assert pad_size == 0

    expected_obs = torch.tensor([[1, 2], [3, 4], [5, 6]])
    expected_labels = ["a", "b", "c"]

    assert torch.all(torch.eq(padded_data.batch["obs"], expected_obs))
    assert (padded_data.non_tensor_batch["labels"] == expected_labels).all()
    assert padded_data.meta_info == {"info": "test_info"}

    unpadd_data = unpad_dataproto(padded_data, pad_size=pad_size)
    assert torch.all(torch.eq(unpadd_data.batch["obs"], obs))
    assert (unpadd_data.non_tensor_batch["labels"] == labels).all()
    assert unpadd_data.meta_info == {"info": "test_info"}

    padded_data, pad_size = pad_dataproto_to_divisor(data, size_divisor=7)
    assert pad_size == 4

    expected_obs = torch.tensor([[1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2]])
    expected_labels = ["a", "b", "c", "a", "b", "c", "a"]
    assert torch.all(torch.eq(padded_data.batch["obs"], expected_obs))
    assert (padded_data.non_tensor_batch["labels"] == expected_labels).all()
    assert padded_data.meta_info == {"info": "test_info"}

    unpadd_data = unpad_dataproto(padded_data, pad_size=pad_size)
    assert torch.all(torch.eq(unpadd_data.batch["obs"], obs))
    assert (unpadd_data.non_tensor_batch["labels"] == labels).all()
    assert unpadd_data.meta_info == {"info": "test_info"}



def test_torch_save_data_proto():
    obs = torch.tensor([[1, 2], [3, 4], [5, 6]])
    labels = ["a", "b", "c"]
    data = DataProto.from_dict(tensors={"obs": obs}, non_tensors={"labels": labels}, meta_info={"info": "test_info"})
    data.save_to_disk("test_data.pt")
    loaded_data = DataProto.load_from_disk("test_data.pt")

    assert torch.all(torch.eq(loaded_data.batch["obs"], data.batch["obs"]))
    assert (loaded_data.non_tensor_batch["labels"] == data.non_tensor_batch["labels"]).all()
    assert loaded_data.meta_info == data.meta_info

    import os

    os.remove("test_data.pt")


def test_len():
    obs = torch.tensor([[1, 2], [3, 4], [5, 6]])
    labels = np.array(["a", "b", "c"], dtype=object)

    data = tu.get_tensordict({"obs": obs, "labels": labels.tolist()}, non_tensor_dict={'info': 'test_info'})
    assert len(data) == 3

    data = tu.get_tensordict({"labels": labels.tolist()}, non_tensor_dict={'info': 'test_info'})
    assert len(data) == 3

    data = tu.get_tensordict({}, non_tensor_dict={'info': 'test_info'})
    assert len(data) == 0

    data_item = data[0]
    assert len(data_item) == 0


def test_dataproto_index():
    data_len = 100
    idx_num = 10

    obs = torch.randn(data_len, 10)
    labels = [random.choice(["abc", "cde"]) for _ in range(data_len)]
    data = DataProto.from_dict(tensors={"obs": obs}, non_tensors={"labels": labels})
    labels_np = np.array(labels)

    idx_np_int = np.random.randint(0, data_len, size=(idx_num,))
    result_np_int = data[idx_np_int]
    assert result_np_int.batch.keys() == data.batch.keys()
    assert result_np_int.non_tensor_batch.keys() == data.non_tensor_batch.keys()
    assert result_np_int.batch["obs"].shape[0] == idx_num
    assert result_np_int.non_tensor_batch["labels"].shape[0] == idx_num
    assert np.array_equal(result_np_int.batch["obs"].cpu().numpy(), obs[idx_np_int].numpy())
    assert np.array_equal(result_np_int.non_tensor_batch["labels"], labels_np[idx_np_int])

    idx_torch_int = torch.randint(0, data_len, size=(idx_num,))
    result_torch_int = data[idx_torch_int]
    assert result_torch_int.batch.keys() == data.batch.keys()
    assert result_torch_int.non_tensor_batch.keys() == data.non_tensor_batch.keys()
    assert result_torch_int.batch["obs"].shape[0] == idx_num
    assert result_torch_int.non_tensor_batch["labels"].shape[0] == idx_num
    assert np.array_equal(result_torch_int.batch["obs"].cpu().numpy(), obs[idx_torch_int].cpu().numpy())
    assert np.array_equal(result_torch_int.non_tensor_batch["labels"], labels_np[idx_torch_int.cpu().numpy()])

    idx_list_int = [np.random.randint(0, data_len) for _ in range(idx_num)]
    result_list_int = data[idx_list_int]
    assert result_list_int.batch.keys() == data.batch.keys()
    assert result_list_int.non_tensor_batch.keys() == data.non_tensor_batch.keys()
    assert result_list_int.batch["obs"].shape[0] == idx_num
    assert result_list_int.non_tensor_batch["labels"].shape[0] == idx_num
    assert np.array_equal(result_list_int.batch["obs"].cpu().numpy(), obs[idx_list_int].cpu().numpy())
    assert np.array_equal(result_list_int.non_tensor_batch["labels"], labels_np[idx_list_int])

    idx_np_bool = np.random.randint(0, 2, size=(data_len,), dtype=bool)
    result_np_bool = data[idx_np_bool]
    assert result_np_bool.batch.keys() == data.batch.keys()
    assert result_np_bool.non_tensor_batch.keys() == data.non_tensor_batch.keys()
    assert result_np_bool.batch["obs"].shape[0] == idx_np_bool.sum()
    assert result_np_bool.non_tensor_batch["labels"].shape[0] == idx_np_bool.sum()
    assert np.array_equal(result_np_bool.batch["obs"].cpu().numpy(), obs[idx_np_bool].cpu().numpy())
    assert np.array_equal(result_np_bool.non_tensor_batch["labels"], labels_np[idx_np_bool])

    idx_torch_bool = torch.randint(0, 2, size=(data_len,), dtype=torch.bool)
    result_torch_bool = data[idx_torch_bool]
    assert result_torch_bool.batch.keys() == data.batch.keys()
    assert result_torch_bool.non_tensor_batch.keys() == data.non_tensor_batch.keys()
    assert result_torch_bool.batch["obs"].shape[0] == idx_torch_bool.sum().item()
    assert result_torch_bool.non_tensor_batch["labels"].shape[0] == idx_torch_bool.sum().item()
    assert np.array_equal(result_torch_bool.batch["obs"].cpu().numpy(), obs[idx_torch_bool].cpu().numpy())
    assert np.array_equal(result_torch_bool.non_tensor_batch["labels"], labels_np[idx_torch_bool])

    idx_list_bool = [np.random.randint(0, 2, dtype=bool) for _ in range(data_len)]
    result_list_bool = data[idx_list_bool]
    assert result_list_bool.batch.keys() == data.batch.keys()
    assert result_list_bool.non_tensor_batch.keys() == data.non_tensor_batch.keys()
    assert result_list_bool.batch["obs"].shape[0] == sum(idx_list_bool)
    assert result_list_bool.non_tensor_batch["labels"].shape[0] == sum(idx_list_bool)
    assert np.array_equal(result_list_bool.batch["obs"].cpu().numpy(), obs[idx_list_bool].cpu().numpy())
    assert np.array_equal(result_list_bool.non_tensor_batch["labels"], labels_np[idx_list_bool])



def test_dataproto_no_batch():
    labels = ["a", "b", "c"]
    data = DataProto.from_dict(non_tensors={"labels": labels}, meta_info={"info": "test_info"})
    selected = data.select(non_tensor_batch_keys=["labels"])
    assert (selected.non_tensor_batch["labels"] == labels).all()
    pop_data = data.pop(non_tensor_batch_keys=["labels"])
    assert (pop_data.non_tensor_batch["labels"] == labels).all()
    assert data.non_tensor_batch == {}


def test_sample_level_repeat():
    # Create a DataProto object with some batch and non-tensor data
    obs = torch.tensor([[1, 2], [3, 4], [5, 6]])
    labels = ["a", "b", "c"]
    data = DataProto.from_dict(tensors={"obs": obs}, non_tensors={"labels": labels}, meta_info={"info": "test_info"})

    # list
    repeated_data_interleave = data.sample_level_repeat(repeat_times=[3, 1, 2])
    expected_obs_interleave = torch.tensor([[1, 2], [1, 2], [1, 2], [3, 4], [5, 6], [5, 6]])
    expected_labels_interleave = ["a", "a", "a", "b", "c", "c"]

    assert torch.all(torch.eq(repeated_data_interleave.batch["obs"], expected_obs_interleave))
    assert (repeated_data_interleave.non_tensor_batch["labels"] == expected_labels_interleave).all()
    assert repeated_data_interleave.meta_info == {"info": "test_info"}

    # torch.tensor
    repeated_data_no_interleave = data.sample_level_repeat(repeat_times=torch.tensor([1, 2, 3]))
    expected_obs_no_interleave = torch.tensor([[1, 2], [3, 4], [3, 4], [5, 6], [5, 6], [5, 6]])
    expected_labels_no_interleave = ["a", "b", "b", "c", "c", "c"]

    assert torch.all(torch.eq(repeated_data_no_interleave.batch["obs"], expected_obs_no_interleave))
    assert (repeated_data_no_interleave.non_tensor_batch["labels"] == expected_labels_no_interleave).all()
    assert repeated_data_no_interleave.meta_info == {"info": "test_info"}


def test_dataproto_chunk_after_index():
    data_len = 4
    obs = torch.randn(data_len, 4)
    labels = [f"label_{i}" for i in range(data_len)]
    data = DataProto.from_dict(tensors={"obs": obs}, non_tensors={"labels": labels}, meta_info={"name": "abc"})

    # Test with boolean numpy array
    bool_mask = np.array([True, False, True, False])
    selected = data[bool_mask]
    assert isinstance(selected.batch.batch_size, torch.Size)
    assert all(isinstance(d, int) for d in selected.batch.batch_size)  # int or List[int]

    # Test with integer numpy array
    int_mask = np.array([0, 2])
    selected = data[int_mask]
    assert isinstance(selected.batch.batch_size, torch.Size)
    assert all(isinstance(d, int) for d in selected.batch.batch_size)

    # Test with boolean list
    list_mask = [True, False, True, False]
    selected = data[list_mask]
    assert isinstance(selected.batch.batch_size, torch.Size)
    assert all(isinstance(d, int) for d in selected.batch.batch_size)

    # Test with list
    list_mask = [0, 2]
    selected = data[list_mask]
    assert isinstance(selected.batch.batch_size, torch.Size)
    assert all(isinstance(d, int) for d in selected.batch.batch_size)

    # Test with torch tensor (bool)
    torch_bool_mask = torch.tensor([True, False, True, False])
    selected = data[torch_bool_mask]
    assert isinstance(selected.batch.batch_size, torch.Size)
    assert all(isinstance(d, int) for d in selected.batch.batch_size)

    # Test with torch tensor (int)
    torch_int_mask = torch.tensor([0, 2])
    selected = data[torch_int_mask]
    assert isinstance(selected.batch.batch_size, torch.Size)
    assert all(isinstance(d, int) for d in selected.batch.batch_size)
