# Copyright 2025 Individual Contributor: furunding
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

import io

import torch


def chunk_list(lst, n_chunks):
    """Split a list into chunks of equal length"""
    size = len(lst) // n_chunks
    for i, start in enumerate(range(0, len(lst), size)):
        if i == n_chunks - 1:
            yield lst[start:]
            return
        else:
            yield lst[start : start + size]


def serialize(data):
    buffer = io.BytesIO()
    torch.save(data, buffer)
    return buffer.getbuffer()


def deserialize(message):
    buffer = io.BytesIO(message)
    return torch.load(buffer)


if __name__ == "__main__":
    lst = list(range(12))
    sub_lsts = list(chunk_list(lst, 3))

    assert len(sub_lsts) == 3
    assert sub_lsts[0] == [0, 1, 2, 3]
    assert sub_lsts[1] == [4, 5, 6, 7]
    assert sub_lsts[2] == [8, 9, 10, 11]

    lst = list(range(11))
    sub_lsts = list(chunk_list(lst, 3))
    assert len(sub_lsts) == 3
    assert sub_lsts[0] == [0, 1, 2]
    assert sub_lsts[1] == [3, 4, 5]
    assert sub_lsts[2] == [6, 7, 8, 9, 10]

    lst = list(range(11))
    sub_lsts = list(chunk_list(lst, 1))
    assert len(sub_lsts) == 1
    assert sub_lsts[0] == list(range(11))
