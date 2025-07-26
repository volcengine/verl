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

import random

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.protocol import union_numpy_dict, union_tensor_dict


def test_union_tensor_dict():
    obs = torch.randn(100, 10)

    data1 = TensorDict({"obs": obs, "act": torch.randn(100, 3)}, batch_size=[100])
    data2 = TensorDict({"obs": obs, "next_obs": torch.randn(100, 10), "rew": torch.randn(100)}, batch_size=[100])

    data_with_copied_obs = TensorDict(
        {"obs": obs.clone(), "next_obs": torch.randn(100, 10), "rew": torch.randn(100)}, batch_size=[100]
    )

    union_tensor_dict(data1, data2)
    with pytest.raises(AssertionError):
        union_tensor_dict(data1, data_with_copied_obs)


def test_union_numpy_dict():
    """
    A comprehensive test suite for union_numpy_dict, covering standard use
    cases, N-dimensional arrays, object-dtype arrays, and NaN value handling.
    """
    arr_3d = np.arange(8).reshape((2, 2, 2))
    union_numpy_dict({"a": arr_3d}, {"a": arr_3d})
    arr1 = np.array([1, "hello", np.array([2, 3])], dtype=object)
    arr2 = np.array([1, "hello", np.array([2, 3])], dtype=object)
    union_numpy_dict({"a": arr1}, {"a": arr2})
    # --- Test Case 1: The original test with mixed object/float types ---
    # This test case from the original test file is preserved.
    data = np.random.random(100)
    # This array intentionally mixes float('nan') and the string 'nan'
    nan_data = [float("nan") for _ in range(99)]
    nan_data.append("nan")
    nan_data_arr = np.array(nan_data, dtype=object)

    dict1 = {"a": data, "b": nan_data_arr}
    dict2_same = {"a": data.copy(), "b": nan_data_arr.copy()}
    dict3_different = {"a": np.random.random(100)}

    union_numpy_dict(dict1, dict2_same)  # Should pass
    with pytest.raises(AssertionError):
        union_numpy_dict(dict1, dict3_different)

    # --- Test Case 2: Standard 3D arrays (fixes the core bug) ---
    arr_3d = np.arange(24, dtype=np.int32).reshape((2, 3, 4))
    dict_3d_1 = {"nd_array": arr_3d}
    dict_3d_2_same = {"nd_array": arr_3d.copy()}
    dict_3d_3_different = {"nd_array": arr_3d + 1}

    union_numpy_dict(dict_3d_1, dict_3d_2_same)  # Should pass
    with pytest.raises(AssertionError, match="`nd_array` in tensor_dict1 and tensor_dict2 are not the same object."):
        union_numpy_dict(dict_3d_1, dict_3d_3_different)

    # --- Test Case 3: Nested 2D and 4D object-dtype arrays ---
    sub_arr1 = np.array([1, 2])
    sub_arr2 = np.array([3.0, 4.0])
    # 2D object array
    arr_2d_obj = np.array([[sub_arr1, "text"], [sub_arr2, None]], dtype=object)
    arr_2d_obj_diff = np.array([[sub_arr1, "text"], [sub_arr2, "other"]], dtype=object)

    union_numpy_dict({"data": arr_2d_obj}, {"data": arr_2d_obj.copy()})  # Should pass
    with pytest.raises(AssertionError):
        union_numpy_dict({"data": arr_2d_obj}, {"data": arr_2d_obj_diff})

    # 4D object array to ensure deep recursion is robust
    arr_4d_obj = np.array([[[[sub_arr1]]], [[[sub_arr2]]]], dtype=object)
    arr_4d_obj_diff = np.array([[[[sub_arr1]]], [[[np.array([9, 9])]]]], dtype=object)

    union_numpy_dict({"data": arr_4d_obj}, {"data": arr_4d_obj.copy()})  # Should pass
    with pytest.raises(AssertionError):
        union_numpy_dict({"data": arr_4d_obj}, {"data": arr_4d_obj_diff})

    # --- Test Case 4: Explicit NaN value comparison ---
    # This verifies that our new _deep_equal logic correctly handles NaNs.
    nan_arr = np.array([1.0, np.nan, 3.0])
    dict_nan_1 = {"data": nan_arr}
    dict_nan_2_same = {"data": np.array([1.0, np.nan, 3.0])}  # A new array with same values
    dict_nan_3_different_val = {"data": np.array([1.0, 2.0, 3.0])}
    dict_nan_4_different_pos = {"data": np.array([np.nan, 1.0, 3.0])}

    # NaNs in the same position should be considered equal for merging.
    union_numpy_dict(dict_nan_1, dict_nan_2_same)  # Should pass

    with pytest.raises(AssertionError):
        union_numpy_dict(dict_nan_1, dict_nan_3_different_val)
    with pytest.raises(AssertionError):
        union_numpy_dict(dict_nan_1, dict_nan_4_different_pos)

    # --- Test Case 5: Circular reference handling ---
    # Create two separate, but structurally identical, circular references.
    # This should pass without a RecursionError.
    circ_arr_1 = np.array([None], dtype=object)
    circ_arr_1[0] = circ_arr_1

    circ_arr_2 = np.array([None], dtype=object)
    circ_arr_2[0] = circ_arr_2

    union_numpy_dict({"data": circ_arr_1}, {"data": circ_arr_2})  # Should pass

    # Create a circular reference and a non-circular one.
    # This should fail with an AssertionError because they are different.
    non_circ_arr = np.array([None], dtype=object)

    with pytest.raises(AssertionError):
        union_numpy_dict({"data": circ_arr_1}, {"data": non_circ_arr})


def test_tensor_dict_constructor():
    obs = torch.randn(100, 10)
    act = torch.randn(100, 10, 3)
    data = DataProto.from_dict(tensors={"obs": obs, "act": act})

    assert data.batch.batch_size == torch.Size([100])

    with pytest.raises(AssertionError):
        data = DataProto.from_dict(tensors={"obs": obs, "act": act}, num_batch_dims=2)

    with pytest.raises(AssertionError):
        data = DataProto.from_dict(tensors={"obs": obs, "act": act}, num_batch_dims=3)


def test_tensor_dict_make_iterator():
    obs = torch.randn(100, 10)
    labels = [random.choice(["abc", "cde"]) for _ in range(100)]
    dataset = DataProto.from_dict(tensors={"obs": obs}, non_tensors={"labels": labels})

    data_iter_1 = dataset.make_iterator(mini_batch_size=10, epochs=2, seed=1)
    data_list_1 = []
    for data in data_iter_1:
        data_list_1.append(data)

    data_iter_2 = dataset.make_iterator(mini_batch_size=10, epochs=2, seed=1)
    data_list_2 = []
    for data in data_iter_2:
        data_list_2.append(data)

    for data1, data2 in zip(data_list_1, data_list_2, strict=True):
        assert isinstance(data1, DataProto)
        assert isinstance(data2, DataProto)
        result = torch.all(torch.eq(data1.batch["obs"], data2.batch["obs"]))
        if not result.item():
            print(data1.batch["obs"])
            print(data2.batch["obs"])
            raise AssertionError()
        non_tensor_result = np.all(np.equal(data1.non_tensor_batch["labels"], data2.non_tensor_batch["labels"]))
        if not non_tensor_result.item():
            print(data1.non_tensor_batch["labels"])
            print(data2.non_tensor_batch["labels"])


def test_reorder():
    obs = torch.tensor([1, 2, 3, 4, 5, 6])
    labels = ["a", "b", "c", "d", "e", "f"]
    data = DataProto.from_dict(tensors={"obs": obs}, non_tensors={"labels": labels}, meta_info={"name": "abdce"})
    data.reorder(torch.tensor([3, 4, 2, 0, 1, 5]))

    assert torch.all(torch.eq(data.batch["obs"], torch.tensor([4, 5, 3, 1, 2, 6])))
    assert np.all(data.non_tensor_batch["labels"] == np.array(["d", "e", "c", "a", "b", "f"]))
    assert data.meta_info == {"name": "abdce"}


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


def test_dataproto_fold_unfold():
    from verl.protocol import DataProto, fold_batch_dim, unfold_batch_dim

    obs = torch.tensor([[1, 2], [3, 4], [5, 6]])
    labels = ["a", "b", "c"]
    data = DataProto.from_dict(tensors={"obs": obs}, non_tensors={"labels": labels}, meta_info={"info": "test_info"})

    data1 = data.repeat(repeat_times=2, interleave=True)

    data2 = fold_batch_dim(data1, new_batch_size=3)

    torch.testing.assert_close(data2.batch["obs"], torch.tensor([[[1, 2], [1, 2]], [[3, 4], [3, 4]], [[5, 6], [5, 6]]]))
    assert (data2.non_tensor_batch["labels"] == [["a", "a"], ["b", "b"], ["c", "c"]]).all()

    data2.reorder(indices=torch.tensor([1, 2, 0]))

    data3 = unfold_batch_dim(data2, batch_dims=2)

    torch.testing.assert_close(data3.batch["obs"], torch.tensor([[3, 4], [3, 4], [5, 6], [5, 6], [1, 2], [1, 2]]))
    assert (data3.non_tensor_batch["labels"] == ["b", "b", "c", "c", "a", "a"]).all()
    assert data3.meta_info == {"info": "test_info"}


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
    data = DataProto.from_dict(tensors={"obs": obs}, non_tensors={"labels": labels}, meta_info={"info": "test_info"})

    assert len(data) == 3

    data = DataProto(batch=None, non_tensor_batch={"labels": labels}, meta_info={"info": "test_info"})

    assert len(data) == 3

    data = DataProto(batch=None, non_tensor_batch={}, meta_info={"info": "test_info"})

    assert len(data) == 0

    data = DataProto(batch=None, non_tensor_batch=None, meta_info={"info": "test_info"})

    assert len(data) == 0


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


def test_old_vs_new_from_single_dict():
    class CustomProto(DataProto):
        """Uses the new, fixed from_single_dict."""

        pass

    class OriginProto(DataProto):
        """Mimics the *old* from_single_dict (always returns a DataProto)."""

        @classmethod
        def from_single_dict(cls, data, meta_info=None, auto_padding=False):
            tensors, non_tensors = {}, {}
            for k, v in data.items():
                if torch.is_tensor(v):
                    tensors[k] = v
                else:
                    non_tensors[k] = v
            # always calls DataProto.from_dict, ignoring `cls`
            return DataProto.from_dict(
                tensors=tensors,
                non_tensors=non_tensors,
                meta_info=meta_info,
                auto_padding=auto_padding,
            )

    sample = {"x": torch.tensor([0])}

    orig = OriginProto.from_single_dict(sample)
    # old behavior: always DataProto, not a CustomOriginProto
    assert type(orig) is DataProto
    assert type(orig) is not OriginProto

    cust = CustomProto.from_single_dict(sample)
    # new behavior: respects subclass
    assert type(cust) is CustomProto


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


def test_dataproto_unfold_column_chunks():
    obs1 = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    obs2 = torch.tensor([[1, 2], [5, 6], [9, 10]])

    labels = ["a", "b", "c"]
    data = DataProto.from_dict(
        tensors={"obs1": obs1, "obs2": obs2}, non_tensors={"labels": labels}, meta_info={"name": "abc"}
    )
    ret = data.unfold_column_chunks(2, split_keys=["obs1"])

    expect_obs1 = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    expect_obs2 = torch.tensor([[1, 2], [1, 2], [5, 6], [5, 6], [9, 10], [9, 10]])
    expect_labels = ["a", "a", "b", "b", "c", "c"]
    assert torch.all(torch.eq(ret.batch["obs1"], expect_obs1))
    assert torch.all(torch.eq(ret.batch["obs2"], expect_obs2))
    assert (ret.non_tensor_batch["labels"] == expect_labels).all()
    assert ret.meta_info == {"name": "abc"}

    obs1 = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    obs2 = torch.tensor([[1, 2], [5, 6], [9, 10]])

    labels = [["a1", "a2"], ["b1", "b2"], ["c1", "c2"]]
    data = DataProto.from_dict(
        tensors={"obs1": obs1, "obs2": obs2}, non_tensors={"labels": labels}, meta_info={"name": "abc"}
    )
    ret = data.unfold_column_chunks(2, split_keys=["obs1", "labels"])

    expect_obs1 = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    expect_obs2 = torch.tensor([[1, 2], [1, 2], [5, 6], [5, 6], [9, 10], [9, 10]])
    expect_labels = [["a1"], ["a2"], ["b1"], ["b2"], ["c1"], ["c2"]]
    assert torch.all(torch.eq(ret.batch["obs1"], expect_obs1))
    assert torch.all(torch.eq(ret.batch["obs2"], expect_obs2))
    assert (ret.non_tensor_batch["labels"] == expect_labels).all()
    assert ret.meta_info == {"name": "abc"}

    obs1 = torch.tensor(
        [[[1, 1], [2, 2], [3, 3], [4, 4]], [[5, 5], [6, 6], [7, 7], [8, 8]], [[9, 9], [10, 10], [11, 11], [12, 12]]]
    )
    obs2 = torch.tensor([[[1, 1], [2, 2]], [[5, 5], [6, 6]], [[9, 9], [10, 10]]])

    labels = ["a", "b", "c"]
    data = DataProto.from_dict(
        tensors={"obs1": obs1, "obs2": obs2}, non_tensors={"labels": labels}, meta_info={"name": "abc"}
    )
    ret = data.unfold_column_chunks(2, split_keys=["obs1"])

    expect_obs1 = torch.tensor(
        [
            [[1, 1], [2, 2]],
            [[3, 3], [4, 4]],
            [[5, 5], [6, 6]],
            [[7, 7], [8, 8]],
            [[9, 9], [10, 10]],
            [[11, 11], [12, 12]],
        ]
    )
    expect_obs2 = torch.tensor(
        [[[1, 1], [2, 2]], [[1, 1], [2, 2]], [[5, 5], [6, 6]], [[5, 5], [6, 6]], [[9, 9], [10, 10]], [[9, 9], [10, 10]]]
    )
    expect_labels = ["a", "a", "b", "b", "c", "c"]
    assert torch.all(torch.eq(ret.batch["obs1"], expect_obs1))
    assert torch.all(torch.eq(ret.batch["obs2"], expect_obs2))
    assert (ret.non_tensor_batch["labels"] == expect_labels).all()
    assert ret.meta_info == {"name": "abc"}


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
