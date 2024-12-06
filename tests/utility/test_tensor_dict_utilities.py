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
import pytest
import torch
from tensordict import TensorDict

from verl.protocol import union_tensor_dict, union_numpy_dict

from verl import DataProto
import numpy as np


def test_union_tensor_dict():
    obs = torch.randn(100, 10)

    data1 = TensorDict({'obs': obs, 'act': torch.randn(100, 3)}, batch_size=[100])
    data2 = TensorDict({'obs': obs, 'next_obs': torch.randn(100, 10), 'rew': torch.randn(100)}, batch_size=[100])

    data_with_copied_obs = TensorDict({
        'obs': obs.clone(),
        'next_obs': torch.randn(100, 10),
        'rew': torch.randn(100)
    },
                                      batch_size=[100])

    data = union_tensor_dict(data1, data2)
    with pytest.raises(AssertionError):
        data = union_tensor_dict(data1, data_with_copied_obs)

    data = np.random.random(100)
    a = {'a': data}
    b = {'a': data}
    b_ = {'a': np.random.random(100)}
    union_numpy_dict(a, b)
    with pytest.raises(AssertionError):
        union_numpy_dict(a, b_)


def test_tensor_dict_constructor():
    obs = torch.randn(100, 10)
    act = torch.randn(100, 10, 3)
    data = DataProto.from_dict(tensors={'obs': obs, 'act': act})

    assert data.batch.batch_size == torch.Size([100])

    with pytest.raises(AssertionError):
        data = DataProto.from_dict(tensors={'obs': obs, 'act': act}, num_batch_dims=2)

    with pytest.raises(AssertionError):
        data = DataProto.from_dict(tensors={'obs': obs, 'act': act}, num_batch_dims=3)


def test_tensor_dict_make_iterator():
    obs = torch.randn(100, 10)
    labels = [random.choice(['abc', 'cde']) for _ in range(100)]
    dataset = DataProto.from_dict(tensors={'obs': obs}, non_tensors={'labels': labels})

    data_iter_1 = dataset.make_iterator(mini_batch_size=10, epochs=2, seed=1)
    data_list_1 = []
    for data in data_iter_1:
        data_list_1.append(data)

    data_iter_2 = dataset.make_iterator(mini_batch_size=10, epochs=2, seed=1)
    data_list_2 = []
    for data in data_iter_2:
        data_list_2.append(data)

    for data1, data2 in zip(data_list_1, data_list_2):
        assert isinstance(data1, DataProto)
        assert isinstance(data2, DataProto)
        result = torch.all(torch.eq(data1.batch['obs'], data2.batch['obs']))
        if not result.item():
            print(data1.batch['obs'])
            print(data2.batch['obs'])
            assert False
        non_tensor_result = np.all(np.equal(data1.non_tensor_batch['labels'], data2.non_tensor_batch['labels']))
        if not non_tensor_result.item():
            print(data1.non_tensor_batch['labels'])
            print(data2.non_tensor_batch['labels'])


def test_reorder():
    obs = torch.tensor([1, 2, 3, 4, 5, 6])
    labels = ['a', 'b', 'c', 'd', 'e', 'f']
    data = DataProto.from_dict(tensors={'obs': obs}, non_tensors={'labels': labels}, meta_info={'name': 'abdce'})
    data.reorder(torch.tensor([3, 4, 2, 0, 1, 5]))

    assert torch.all(torch.eq(data.batch['obs'], torch.tensor([4, 5, 3, 1, 2, 6])))
    assert np.all(data.non_tensor_batch['labels'] == np.array(['d', 'e', 'c', 'a', 'b', 'f']))
    assert data.meta_info == {'name': 'abdce'}


def test_chunk_concat():
    obs = torch.tensor([1, 2, 3, 4, 5, 6])
    labels = ['a', 'b', 'c', 'd', 'e', 'f']
    data = DataProto.from_dict(tensors={'obs': obs}, non_tensors={'labels': labels}, meta_info={'name': 'abdce'})

    data_split = data.chunk(2)
    assert len(data_split) == 2
    assert torch.all(torch.eq(data_split[0].batch['obs'], torch.tensor([1, 2, 3])))
    assert np.all(data_split[0].non_tensor_batch['labels'] == np.array(['a', 'b', 'c']))
    assert data_split[0].meta_info == {'name': 'abdce'}

    assert torch.all(torch.eq(data_split[1].batch['obs'], torch.tensor([4, 5, 6])))
    assert np.all(data_split[1].non_tensor_batch['labels'] == np.array(['d', 'e', 'f']))
    assert data_split[1].meta_info == {'name': 'abdce'}

    concat_data = DataProto.concat(data_split)
    assert torch.all(torch.eq(concat_data.batch['obs'], data.batch['obs']))
    assert np.all(concat_data.non_tensor_batch['labels'] == data.non_tensor_batch['labels'])
    assert concat_data.meta_info == data.meta_info


def test_pop():
    obs = torch.randn(100, 10)
    act = torch.randn(100, 3)
    dataset = DataProto.from_dict({'obs': obs, 'act': act}, meta_info={'2': 2, '1': 1})
    poped_dataset = dataset.pop(batch_keys=['obs'], meta_info_keys=['2'])

    assert poped_dataset.batch.keys() == {'obs'}
    assert poped_dataset.meta_info.keys() == {'2'}

    assert dataset.batch.keys() == {'act'}
    assert dataset.meta_info.keys() == {'1'}


def test_repeat():
    # Create a DataProto object with some batch and non-tensor data
    obs = torch.tensor([[1, 2], [3, 4], [5, 6]])
    labels = ['a', 'b', 'c']
    data = DataProto.from_dict(tensors={'obs': obs}, non_tensors={'labels': labels}, meta_info={'info': 'test_info'})

    # Test interleave=True
    repeated_data_interleave = data.repeat(repeat_times=2, interleave=True)
    expected_obs_interleave = torch.tensor([[1, 2], [1, 2], [3, 4], [3, 4], [5, 6], [5, 6]])
    expected_labels_interleave = ['a', 'a', 'b', 'b', 'c', 'c']

    assert torch.all(torch.eq(repeated_data_interleave.batch['obs'], expected_obs_interleave))
    assert (repeated_data_interleave.non_tensor_batch['labels'] == expected_labels_interleave).all()
    assert repeated_data_interleave.meta_info == {'info': 'test_info'}

    # Test interleave=False
    repeated_data_no_interleave = data.repeat(repeat_times=2, interleave=False)
    expected_obs_no_interleave = torch.tensor([[1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6]])
    expected_labels_no_interleave = ['a', 'b', 'c', 'a', 'b', 'c']

    assert torch.all(torch.eq(repeated_data_no_interleave.batch['obs'], expected_obs_no_interleave))
    assert (repeated_data_no_interleave.non_tensor_batch['labels'] == expected_labels_no_interleave).all()
    assert repeated_data_no_interleave.meta_info == {'info': 'test_info'}


def test_dataproto_pad_unpad():
    obs = torch.tensor([[1, 2], [3, 4], [5, 6]])
    labels = ['a', 'b', 'c']
    data = DataProto.from_dict(tensors={'obs': obs}, non_tensors={'labels': labels}, meta_info={'info': 'test_info'})

    from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

    padded_data, pad_size = pad_dataproto_to_divisor(data, size_divisor=2)
    assert pad_size == 1

    expected_obs = torch.tensor([[1, 2], [3, 4], [5, 6], [1, 2]])
    expected_labels = ['a', 'b', 'c', 'a']

    assert torch.all(torch.eq(padded_data.batch['obs'], expected_obs))
    assert (padded_data.non_tensor_batch['labels'] == expected_labels).all()
    assert padded_data.meta_info == {'info': 'test_info'}

    unpadd_data = unpad_dataproto(padded_data, pad_size=pad_size)
    assert torch.all(torch.eq(unpadd_data.batch['obs'], obs))
    assert (unpadd_data.non_tensor_batch['labels'] == labels).all()
    assert unpadd_data.meta_info == {'info': 'test_info'}

    padded_data, pad_size = pad_dataproto_to_divisor(data, size_divisor=3)
    assert pad_size == 0

    expected_obs = torch.tensor([[1, 2], [3, 4], [5, 6]])
    expected_labels = ['a', 'b', 'c']

    assert torch.all(torch.eq(padded_data.batch['obs'], expected_obs))
    assert (padded_data.non_tensor_batch['labels'] == expected_labels).all()
    assert padded_data.meta_info == {'info': 'test_info'}

    unpadd_data = unpad_dataproto(padded_data, pad_size=pad_size)
    assert torch.all(torch.eq(unpadd_data.batch['obs'], obs))
    assert (unpadd_data.non_tensor_batch['labels'] == labels).all()
    assert unpadd_data.meta_info == {'info': 'test_info'}


def test_dataproto_unfold_column_chunks():
    obs1 = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    obs2 = torch.tensor([[1, 2], [5, 6], [9, 10]])

    labels = ['a', 'b', 'c']
    data = DataProto.from_dict(tensors={
        'obs1': obs1,
        'obs2': obs2
    },
                               non_tensors={'labels': labels},
                               meta_info={'name': 'abc'})
    ret = data.unfold_column_chunks(2, split_keys=['obs1'])

    expect_obs1 = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    expect_obs2 = torch.tensor([[1, 2], [1, 2], [5, 6], [5, 6], [9, 10], [9, 10]])
    expect_labels = ['a', 'a', 'b', 'b', 'c', 'c']
    assert torch.all(torch.eq(ret.batch['obs1'], expect_obs1))
    assert torch.all(torch.eq(ret.batch['obs2'], expect_obs2))
    assert (ret.non_tensor_batch['labels'] == expect_labels).all()
    assert ret.meta_info == {'name': 'abc'}

    obs1 = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    obs2 = torch.tensor([[1, 2], [5, 6], [9, 10]])

    labels = [['a1', 'a2'], ['b1', 'b2'], ['c1', 'c2']]
    data = DataProto.from_dict(tensors={
        'obs1': obs1,
        'obs2': obs2
    },
                               non_tensors={'labels': labels},
                               meta_info={'name': 'abc'})
    ret = data.unfold_column_chunks(2, split_keys=['obs1', 'labels'])

    expect_obs1 = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    expect_obs2 = torch.tensor([[1, 2], [1, 2], [5, 6], [5, 6], [9, 10], [9, 10]])
    expect_labels = [[
        'a1',
    ], [
        'a2',
    ], [
        'b1',
    ], [
        'b2',
    ], [
        'c1',
    ], [
        'c2',
    ]]
    assert torch.all(torch.eq(ret.batch['obs1'], expect_obs1))
    assert torch.all(torch.eq(ret.batch['obs2'], expect_obs2))
    assert (ret.non_tensor_batch['labels'] == expect_labels).all()
    assert ret.meta_info == {'name': 'abc'}

    obs1 = torch.tensor([[[1, 1], [2, 2], [3, 3], [4, 4]], [[5, 5], [6, 6], [7, 7], [8, 8]],
                         [[9, 9], [10, 10], [11, 11], [12, 12]]])
    obs2 = torch.tensor([[[1, 1], [2, 2]], [[5, 5], [6, 6]], [[9, 9], [10, 10]]])

    labels = ['a', 'b', 'c']
    data = DataProto.from_dict(tensors={
        'obs1': obs1,
        'obs2': obs2
    },
                               non_tensors={'labels': labels},
                               meta_info={'name': 'abc'})
    ret = data.unfold_column_chunks(2, split_keys=['obs1'])

    expect_obs1 = torch.tensor([[[1, 1], [2, 2]], [[3, 3], [4, 4]], [[5, 5], [6, 6]], [[7, 7], [8, 8]],
                                [[9, 9], [10, 10]], [[11, 11], [12, 12]]])
    expect_obs2 = torch.tensor([[[1, 1], [2, 2]], [[1, 1], [2, 2]], [[5, 5], [6, 6]], [[5, 5], [6, 6]],
                                [[9, 9], [10, 10]], [[9, 9], [10, 10]]])
    expect_labels = ['a', 'a', 'b', 'b', 'c', 'c']
    assert torch.all(torch.eq(ret.batch['obs1'], expect_obs1))
    assert torch.all(torch.eq(ret.batch['obs2'], expect_obs2))
    assert (ret.non_tensor_batch['labels'] == expect_labels).all()
    assert ret.meta_info == {'name': 'abc'}
