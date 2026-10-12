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

import random
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from verl.trainer.ppo.utils import get_balance_mini_batch_size, need_minibatch_shuffle
from verl.utils.seqlen_balancing import calculate_workload, get_minibatch_balanced_partitions


def _worker_mini_batches(partitions, mini_batch_size):
    """Mimic workers: each rank slices its shard into mini-batches sequentially, without shuffling."""
    share = mini_batch_size // len(partitions)
    num_mini_batches = len(partitions[0]) // share
    return [[partition[i * share : (i + 1) * share] for partition in partitions] for i in range(num_mini_batches)]


def _check_layout(partitions, num_samples, k_partitions):
    assert len(partitions) == k_partitions
    assert sorted(i for partition in partitions for i in partition) == list(range(num_samples))
    assert len({len(partition) for partition in partitions}) == 1


def test_keeps_mini_batches_without_shuffle():
    random.seed(0)
    num_samples, mini_batch_size, k_partitions = 64, 16, 4
    workloads = calculate_workload(torch.tensor([random.randint(100, 20000) for _ in range(num_samples)]))

    partitions = get_minibatch_balanced_partitions(workloads, k_partitions, mini_batch_size)

    _check_layout(partitions, num_samples, k_partitions)
    for i, shares in enumerate(_worker_mini_batches(partitions, mini_batch_size)):
        members = sorted(j for share in shares for j in share)
        assert members == list(range(i * mini_batch_size, (i + 1) * mini_batch_size))


def test_balances_each_mini_batch():
    # Rank totals of the whole batch can be balanced while every mini-batch is not; per-mini-batch
    # balancing gives each rank one heavy and one light sample of every mini-batch.
    workloads = [10, 10, 1, 1, 20, 20, 2, 2]

    partitions = get_minibatch_balanced_partitions(workloads, k_partitions=2, mini_batch_size=4)

    for shares in _worker_mini_batches(partitions, mini_batch_size=4):
        sums = [sum(workloads[j] for j in share) for share in shares]
        assert sums[0] == sums[1]


def test_shuffle_is_seeded_and_keeps_mini_batches_aligned():
    num_samples, mini_batch_size, k_partitions = 48, 12, 3
    workloads = list(range(1, num_samples + 1))

    partitions = get_minibatch_balanced_partitions(workloads, k_partitions, mini_batch_size, shuffle=True, seed=[42, 7])

    _check_layout(partitions, num_samples, k_partitions)
    assert partitions == get_minibatch_balanced_partitions(
        workloads, k_partitions, mini_batch_size, shuffle=True, seed=[42, 7]
    )
    assert partitions != get_minibatch_balanced_partitions(
        workloads, k_partitions, mini_batch_size, shuffle=True, seed=[42, 8]
    )
    mini_batches = [sorted(j for share in shares for j in share) for shares in _worker_mini_batches(partitions, 12)]
    assert all(len(members) == mini_batch_size for members in mini_batches)
    assert mini_batches != [list(range(i * 12, (i + 1) * 12)) for i in range(4)]


@pytest.mark.parametrize("mini_batch_size", [0, 7])
def test_falls_back_to_whole_batch(mini_batch_size):
    workloads = list(range(1, 13))

    partitions = get_minibatch_balanced_partitions(workloads, k_partitions=3, mini_batch_size=mini_batch_size)

    _check_layout(partitions, 12, 3)


def test_make_iterator_shuffles_mini_batch_order_per_epoch():
    from tensordict import TensorDict

    from verl.utils import tensordict_utils as tu

    data = TensorDict({"index": torch.arange(16)}, batch_size=[16])

    def visits(seed):
        iterator = tu.make_iterator(data, mini_batch_size=2, epochs=3, seed=seed, shuffle_mini_batch_order=True)
        return [mini_batch["index"].tolist() for mini_batch in iterator]

    batches = visits(seed=1)
    epochs = [batches[i * 8 : (i + 1) * 8] for i in range(3)]
    contiguous = [[2 * i, 2 * i + 1] for i in range(8)]
    for epoch in epochs:
        assert sorted(epoch) == contiguous
    assert epochs[0] != epochs[1] or epochs[1] != epochs[2]
    # Every DP rank passes the same seed, so all ranks visit the same global mini-batch at each step.
    assert visits(seed=1) == batches


def _config(actor_mini_batch_size=4, critic_mini_batch_size=6, actor_shuffle=False, critic_shuffle=False):
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {"n": 2},
                "actor": {"ppo_mini_batch_size": actor_mini_batch_size, "shuffle": actor_shuffle},
            },
            "critic": {"ppo_mini_batch_size": critic_mini_batch_size, "shuffle": critic_shuffle},
        }
    )


def test_balance_mini_batch_size():
    config = _config()
    assert get_balance_mini_batch_size(config, use_critic=False, update_actor=True) == 8
    assert get_balance_mini_batch_size(config, use_critic=True, update_actor=False) == 12
    assert get_balance_mini_batch_size(config, use_critic=True, update_actor=True) == 4
    assert get_balance_mini_batch_size(config, use_critic=False, update_actor=False) == 0


@pytest.mark.parametrize("shuffle", [False, True])
def test_ray_trainer_balance_batch(shuffle):
    from verl import DataProto
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    random.seed(0)
    num_samples, k_partitions = 32, 2
    lengths = [random.randint(10, 100) for _ in range(num_samples)]
    attention_mask = torch.zeros(num_samples, 128, dtype=torch.long)
    for i, length in enumerate(lengths):
        attention_mask[i, :length] = 1
    batch = DataProto.from_single_dict({"attention_mask": attention_mask, "index": torch.arange(num_samples)})

    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = _config(actor_shuffle=shuffle)
    trainer.config.trainer = {"balance_batch": True, "critic_warmup": 0}
    trainer.config.actor_rollout_ref.actor.data_loader_seed = 42
    trainer.use_critic, trainer.global_steps = False, 1
    trainer.actor_rollout_wg = SimpleNamespace(_dispatch_info={"actor": list(range(k_partitions))})
    trainer._balance_batch(batch, metrics={})

    assert batch.meta_info["minibatch_balanced"]
    order = batch.batch["index"].tolist()
    shards = [order[: num_samples // k_partitions], order[num_samples // k_partitions :]]
    mini_batches = [sorted(j for share in shares for j in share) for shares in _worker_mini_batches(shards, 8)]
    if shuffle:
        assert mini_batches != [list(range(i * 8, (i + 1) * 8)) for i in range(4)]
    else:
        assert mini_batches == [list(range(i * 8, (i + 1) * 8)) for i in range(4)]


def test_need_minibatch_shuffle():
    assert not need_minibatch_shuffle(_config(), use_critic=True)
    assert need_minibatch_shuffle(_config(actor_shuffle=True), use_critic=False)
    assert need_minibatch_shuffle(_config(critic_shuffle=True), use_critic=True)
    assert not need_minibatch_shuffle(_config(critic_shuffle=True), use_critic=False)
