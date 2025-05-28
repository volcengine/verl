# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

from verl.utils.megatron.pipeline_parallel import make_batch_generator


def test_make_batch_generator_no_vpp():
    batches = [1, 2, 3]
    vpp_size = 1
    generator = make_batch_generator(batches, vpp_size)
    assert list(generator) == batches


def test_make_batch_generator_with_vpp():
    batches = [{"data": 1}, {"data": 2}]
    vpp_size = 2
    generators = make_batch_generator(batches, vpp_size)
    assert isinstance(generators, list)
    assert len(generators) == vpp_size

    # Check each generator yields the original batches
    for gen in generators:
        assert list(gen) == batches


def test_make_batch_generator_empty():
    batches = []
    vpp_size = 1
    generator = make_batch_generator(batches, vpp_size)
    assert list(generator) == []

    vpp_size = 3
    generators = make_batch_generator(batches, vpp_size)
    assert len(generators) == vpp_size
    for gen in generators:
        assert list(gen) == []
