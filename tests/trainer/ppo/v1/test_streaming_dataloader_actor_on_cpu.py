# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

import asyncio
import inspect
from collections import defaultdict
from contextlib import suppress
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

import verl.trainer.ppo.v1.streaming_dataloader as streaming_module
import verl.trainer.ppo.v1.trainer_base as trainer_module
from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer, ReplayBufferAsync
from verl.trainer.ppo.v1.streaming_dataloader import StreamingDataloader, StreamingDataloaderAsync
from verl.trainer.ppo.v1.trainer_base import PPOTrainer


def _sync_stub():
    return object.__new__(StreamingDataloader)


def _async_stub():
    actor = object.__new__(StreamingDataloaderAsync)
    actor._validation_active = False
    return actor


async def _advance_and_maintain_watermark(actor):
    if actor._advance_once():
        actor._refill_to_watermark()


def _replay_state(*, pending=(), running=(), finished=(), failure=()):
    replay = SimpleNamespace(
        pending_keys=defaultdict(set),
        running_keys=defaultdict(set),
        finished_keys=defaultdict(set),
        failure_keys=defaultdict(set),
    )
    replay.pending_keys["train"].update(pending)
    replay.running_keys["train"].update(running)
    replay.finished_keys["train"].update(finished)
    replay.failure_keys["train"].update(failure)
    return replay


def test_async_dataloader_inherits_sync_dataloader():
    assert issubclass(StreamingDataloaderAsync, StreamingDataloader)


def test_dataloader_subclasses_build_matching_replay_buffers():
    config = OmegaConf.create(
        {
            "data": {"train_batch_size": 8},
            "algorithm": {"filter_groups": {"enable": False}},
            "trainer": {
                "v1": {
                    "sync": {},
                    "colocate_async": {},
                    "sampler": {
                        "custom_sampler": None,
                        "max_off_policy_threshold": 8,
                        "max_off_policy_strategy": "drop",
                        "sampler_kwargs": {},
                        "sync_refill_failed_groups": False,
                    },
                }
            },
        }
    )
    sync_actor = _sync_stub()
    sync_actor.config = config
    sync_actor.trainer_mode = "sync"
    sync_actor.filter_groups_metric = None
    sync_actor.gen_batch_size = 8
    async_actor = _async_stub()
    async_actor.config = config
    async_actor.trainer_mode = "colocate_async"
    async_actor.filter_groups_metric = None

    assert type(sync_actor._build_replay_buffer()) is ReplayBuffer
    assert type(async_actor._build_replay_buffer()) is ReplayBufferAsync


def test_async_custom_sampler_receives_no_refill_callback(monkeypatch):
    received_kwargs = {}

    class CustomAsyncSampler:
        def __init__(self, **kwargs):
            received_kwargs.update(kwargs)

        def poll_once(self, global_steps, partition_id, target_count):
            del global_steps, partition_id, target_count

        def materialize_sampleable(self, global_steps, partition_id, sampleable_keys, batch_size):
            del global_steps, partition_id, sampleable_keys, batch_size

        def unfinished_session_count(self, partition_id, default_rollout_n):
            del partition_id, default_rollout_n
            return 0

    config = OmegaConf.create(
        {
            "trainer": {
                "v1": {
                    "colocate_async": {},
                    "sampler": {
                        "custom_sampler": {"path": "custom.py", "name": "CustomAsyncSampler"},
                        "max_off_policy_threshold": 8,
                        "max_off_policy_strategy": "drop",
                        "sampler_kwargs": {},
                    },
                }
            }
        }
    )
    actor = _async_stub()
    actor.config = config
    actor.trainer_mode = "colocate_async"
    actor.filter_groups_metric = None
    monkeypatch.setattr(streaming_module, "load_extern_type", lambda *_args: CustomAsyncSampler)

    replay_buffer = actor._build_replay_buffer()

    assert isinstance(replay_buffer, CustomAsyncSampler)
    assert "refill_fn" not in received_kwargs


def test_session_watermark_refills_complete_prompt_groups():
    actor = _async_stub()
    actor.watermark_sessions = 5
    actor.default_rollout_n = 2
    actor.gen_batch_size = 1
    actor.replay_buffer = SimpleNamespace(unfinished_session_count=lambda *_args: 1)
    requested = []
    actor._refill_exact = lambda count: requested.append(count) or count

    assert actor._refill_to_watermark() == 2

    assert requested == [2]


def test_sync_sample_submits_batch_before_blocking_replay():
    actor = _sync_stub()
    actor.global_steps = 0
    actor._refill_exact = lambda count: count
    actor.replay_buffer = SimpleNamespace(
        sample=lambda global_steps, partition_id, batch_size: (["a", "b"], {"sample": 1})
    )
    result = actor.sample(global_steps=3, batch_size=2)

    assert result == (["a", "b"], {"sample": 1}, 2)


def test_async_sample_waits_for_coordinator_and_refills_after_delivery():
    async def run():
        actor = _async_stub()
        actor.global_steps = 0
        actor.config = SimpleNamespace(data=SimpleNamespace(train_batch_size=2))
        actor.replay_buffer = SimpleNamespace(
            poll_interval=0.01,
            poll_once=lambda *_args, **_kwargs: ({"uid"}, {}, True, 0),
            materialize_sampleable=lambda *_args: ["trajectory"],
        )
        actor._pending_sample = None
        actor._wakeup = asyncio.Event()
        actor._coordinator_task = None
        actor._train_state_ready = True
        actor._pending_metrics = {}
        refills = []
        actor._refill_to_watermark = lambda: refills.append(True) or 0

        result = await asyncio.wait_for(actor.sample(1, 1), timeout=1)
        assert result == (["trajectory"], {}, 1)
        assert refills

        actor._coordinator_task.cancel()
        with suppress(asyncio.CancelledError):
            await actor._coordinator_task

    asyncio.run(run())


def test_coordinator_failure_fails_pending_sample_and_terminates_actor(monkeypatch):
    async def run():
        actor = _async_stub()
        future = asyncio.get_running_loop().create_future()
        actor._pending_sample = streaming_module._SampleRequest(
            global_steps=1,
            batch_size=1,
            partition_id="train",
            min_sampleable=None,
            consume=True,
            future=future,
        )
        actor._coordinator_task = None

        async def fail():
            raise RuntimeError("coordinator failed")

        actor._coordinator_loop = fail
        exits = []
        monkeypatch.setattr(streaming_module.ray.actor, "exit_actor", lambda: exits.append(True))

        actor._ensure_coordinator()
        with pytest.raises(RuntimeError, match="coordinator failed"):
            await actor._coordinator_task
        await asyncio.sleep(0)

        with pytest.raises(RuntimeError, match="coordinator failed"):
            await future
        assert exits == [True]

    asyncio.run(run())


def test_train_request_uses_one_poll_for_maintenance_and_fulfillment():
    async def run():
        actor = _async_stub()
        actor.global_steps = 3
        actor._pending_sample = None
        actor._pending_metrics = {}
        actor._train_state_ready = True
        polls = []
        refills = []

        def poll_once(global_steps, partition_id, target_count):
            polls.append((global_steps, partition_id, target_count))
            return {"uid-0", "uid-1"}, {}, True, 0

        actor.replay_buffer = SimpleNamespace(
            poll_once=poll_once,
            materialize_sampleable=lambda *_args: ["trajectory-0", "trajectory-1"],
        )
        actor._refill_to_watermark = lambda: refills.append("after") or 0
        future = asyncio.get_running_loop().create_future()
        actor._pending_sample = streaming_module._SampleRequest(
            global_steps=3,
            batch_size=2,
            partition_id="train",
            min_sampleable=1,
            consume=True,
            future=future,
        )

        await _advance_and_maintain_watermark(actor)

        assert polls == [(3, "train", 2)]
        assert await future == (["trajectory-0", "trajectory-1"], {}, 2)
        assert refills == ["after"]
        assert actor._pending_sample is None

    asyncio.run(run())


def test_train_eviction_refills_without_fulfilling_request():
    async def run():
        actor = _async_stub()
        actor.global_steps = 3
        actor._pending_metrics = {}
        actor._train_state_ready = True
        polls = []
        refills = []

        def poll_once(global_steps, partition_id, target_count):
            polls.append((global_steps, partition_id, target_count))
            return {"uid-0"}, {}, False, 1

        actor.replay_buffer = SimpleNamespace(
            poll_once=poll_once,
            materialize_sampleable=lambda *_args: pytest.fail("eviction pass must not materialize"),
        )
        actor._refill_to_watermark = lambda: refills.append(True) or 1
        future = asyncio.get_running_loop().create_future()
        request = streaming_module._SampleRequest(
            global_steps=3,
            batch_size=1,
            partition_id="train",
            min_sampleable=None,
            consume=True,
            future=future,
        )
        actor._pending_sample = request

        await _advance_and_maintain_watermark(actor)

        assert polls == [(3, "train", 1)]
        assert refills == [True]
        assert actor._pending_sample is request
        assert not future.done()
        future.cancel()

    asyncio.run(run())


def test_validation_request_pauses_train_inventory_maintenance():
    async def run():
        actor = _async_stub()
        actor.global_steps = 5
        actor._pending_metrics = {}
        actor._train_state_ready = True
        polls = []
        refills = []

        def poll_once(global_steps, partition_id, target_count):
            polls.append((global_steps, partition_id, target_count))
            return ({"val-uid"} if partition_id == "val" else set()), {}, True, 0

        actor.replay_buffer = SimpleNamespace(
            poll_once=poll_once,
            materialize_sampleable=lambda *_args: ["validation-trajectory"],
        )
        actor._refill_to_watermark = lambda: refills.append(True) or 0
        future = asyncio.get_running_loop().create_future()
        actor._pending_sample = streaming_module._SampleRequest(
            global_steps=5,
            batch_size=1,
            partition_id="val",
            min_sampleable=None,
            consume=True,
            future=future,
        )

        await _advance_and_maintain_watermark(actor)

        assert polls == [(5, "val", 1)]
        assert refills == []
        assert await future == (["validation-trajectory"], {}, 1)

    asyncio.run(run())


def test_validation_mode_pauses_periodic_train_maintenance_between_requests():
    async def run():
        actor = _async_stub()
        actor._pending_sample = None
        actor._train_state_ready = True
        actor._validation_active = True
        actor.replay_buffer = SimpleNamespace(
            poll_once=lambda *_args: pytest.fail("train poll must remain paused during validation")
        )
        actor._refill_to_watermark = lambda: pytest.fail("train refill must remain paused during validation")

        await _advance_and_maintain_watermark(actor)

    asyncio.run(run())


def test_validation_before_load_does_not_maintain_train_inventory():
    async def run():
        actor = _async_stub()
        actor.global_steps = 0
        actor._pending_metrics = {}
        actor._train_state_ready = False
        polls = []

        def poll_once(global_steps, partition_id, target_count):
            polls.append((global_steps, partition_id, target_count))
            return {"val-uid"}, {}, True, 0

        actor.replay_buffer = SimpleNamespace(
            poll_once=poll_once,
            materialize_sampleable=lambda *_args: ["validation-trajectory"],
        )
        actor._refill_to_watermark = lambda: pytest.fail("train refill must remain gated before load")
        future = asyncio.get_running_loop().create_future()
        actor._pending_sample = streaming_module._SampleRequest(
            global_steps=0,
            batch_size=1,
            partition_id="val",
            min_sampleable=None,
            consume=True,
            future=future,
        )

        await _advance_and_maintain_watermark(actor)

        assert polls == [(0, "val", 1)]
        assert await future == (["validation-trajectory"], {}, 1)

    asyncio.run(run())


def test_materialize_failure_completes_sample_future_with_exception():
    async def run():
        actor = _async_stub()
        actor._pending_sample = None
        actor._pending_metrics = {}
        actor._train_state_ready = True
        actor.replay_buffer = SimpleNamespace(
            poll_once=lambda *_args, **_kwargs: ({"uid"}, {}, True, 0),
            materialize_sampleable=lambda *_args: (_ for _ in ()).throw(RuntimeError("materialize failed")),
        )
        actor._refill_to_watermark = lambda: 0
        future = asyncio.get_running_loop().create_future()
        actor._pending_sample = streaming_module._SampleRequest(
            global_steps=1,
            batch_size=1,
            partition_id="train",
            min_sampleable=None,
            consume=True,
            future=future,
        )

        await _advance_and_maintain_watermark(actor)

        with pytest.raises(RuntimeError, match="materialize failed"):
            await future
        assert actor._pending_sample is None

    asyncio.run(run())


def test_consuming_sample_never_uses_minimum_below_batch_size():
    async def run():
        actor = _async_stub()
        actor._pending_sample = None
        actor._pending_metrics = {}
        actor._train_state_ready = True
        targets = []
        refills = []

        def poll_once(_step, _partition, target, **_kwargs):
            targets.append(target)
            return {"uid"}, {}, False, 0

        actor.replay_buffer = SimpleNamespace(poll_once=poll_once)
        actor._refill_to_watermark = lambda: refills.append(True) or 0
        future = asyncio.get_running_loop().create_future()
        actor._pending_sample = streaming_module._SampleRequest(
            global_steps=1,
            batch_size=4,
            partition_id="train",
            min_sampleable=1,
            consume=True,
            future=future,
        )

        await _advance_and_maintain_watermark(actor)

        assert targets == [4]
        assert refills == [True]
        assert not future.done()
        future.cancel()

    asyncio.run(run())


def test_non_consuming_count_request_uses_watermark_coordinator():
    async def run():
        actor = _async_stub()
        actor._pending_metrics = {}
        actor._train_state_ready = True
        targets = []
        refills = []

        def poll_once(_step, _partition, target):
            targets.append(target)
            return {"uid-0", "uid-1"}, {}, False, 0

        actor.replay_buffer = SimpleNamespace(poll_once=poll_once)
        actor._refill_to_watermark = lambda: refills.append("after") or 0
        future = asyncio.get_running_loop().create_future()
        actor._pending_sample = streaming_module._SampleRequest(
            global_steps=1,
            batch_size=0,
            partition_id="train",
            min_sampleable=None,
            consume=False,
            future=future,
        )

        await _advance_and_maintain_watermark(actor)

        assert targets == [0]
        assert await future == (None, {}, 2)
        assert refills == ["after"]

    asyncio.run(run())


def test_non_consuming_wait_request_refills_while_below_target():
    async def run():
        actor = _async_stub()
        actor._pending_metrics = {}
        actor._train_state_ready = True
        targets = []
        refills = []

        def poll_once(_step, _partition, target):
            targets.append(target)
            return {"uid-0"}, {}, False, 0

        actor.replay_buffer = SimpleNamespace(poll_once=poll_once)
        actor._refill_to_watermark = lambda: refills.append(True) or 0
        future = asyncio.get_running_loop().create_future()
        actor._pending_sample = streaming_module._SampleRequest(
            global_steps=1,
            batch_size=0,
            partition_id="train",
            min_sampleable=2,
            consume=False,
            future=future,
        )

        await _advance_and_maintain_watermark(actor)

        assert targets == [2]
        assert refills == [True]
        assert not future.done()
        future.cancel()

    asyncio.run(run())


def test_async_checkpoint_rpc_methods_are_synchronous():
    assert not inspect.iscoroutinefunction(StreamingDataloaderAsync.save_checkpoint)
    assert not inspect.iscoroutinefunction(StreamingDataloaderAsync.load_checkpoint)


def test_sync_checkpoint_contains_only_dataloader(monkeypatch, tmp_path):
    actor = _sync_stub()
    actor.train_dataloader = SimpleNamespace(state_dict=lambda: {"cursor": 7})
    tq_saves = []
    monkeypatch.setattr(
        streaming_module.tq,
        "save_checkpoint",
        lambda path, metadata: tq_saves.append((path, metadata)),
        raising=False,
    )

    actor.save_checkpoint(str(tmp_path), global_steps=7)

    assert torch.load(tmp_path / "data.pt", weights_only=False) == {"cursor": 7}
    assert tq_saves == []


def test_async_checkpoint_contains_dataloader_and_transfer_queue(monkeypatch, tmp_path):
    actor = _async_stub()
    actor.train_dataloader = SimpleNamespace(state_dict=lambda: {"cursor": 7})
    tq_saves = []
    monkeypatch.setattr(
        streaming_module.tq,
        "save_checkpoint",
        lambda path, metadata: tq_saves.append((path, metadata)),
        raising=False,
    )
    monkeypatch.setattr(streaming_module.tq, "load_checkpoint", lambda _path: None, raising=False)
    monkeypatch.setattr(streaming_module.tq, "__version__", "0.1.9")

    actor.save_checkpoint(str(tmp_path), global_steps=7)

    assert torch.load(tmp_path / "data.pt", weights_only=False) == {"cursor": 7}
    assert tq_saves == [(str(tmp_path / "transfer_queue"), {"global_steps": 7})]


def test_async_checkpoint_preserves_dataloader_without_tq_support(monkeypatch, tmp_path):
    actor = _async_stub()
    actor.train_dataloader = SimpleNamespace(state_dict=lambda: {"cursor": 7})
    monkeypatch.setattr(streaming_module.tq, "save_checkpoint", None, raising=False)
    monkeypatch.setattr(streaming_module.tq, "load_checkpoint", None, raising=False)

    actor.save_checkpoint(str(tmp_path), global_steps=7)

    assert torch.load(tmp_path / "data.pt", weights_only=False) == {"cursor": 7}
    assert not (tmp_path / "transfer_queue").exists()


def test_async_restore_preserves_dataloader_without_tq_support(monkeypatch, tmp_path):
    actor = _async_stub()
    loaded_states = []
    actor.train_dataloader = SimpleNamespace(load_state_dict=loaded_states.append)
    actor._reissue_inflight_prompts = lambda: 0
    actor._refill_to_watermark = lambda: pytest.fail("restore must leave watermark refill to the coordinator")
    actor._ensure_coordinator = lambda: None
    actor._wakeup = SimpleNamespace(set=lambda: None)
    torch.save({"cursor": 5}, tmp_path / "data.pt")
    monkeypatch.setattr(streaming_module.tq, "save_checkpoint", None, raising=False)
    monkeypatch.setattr(streaming_module.tq, "load_checkpoint", None, raising=False)

    metrics = actor.load_checkpoint(str(tmp_path), global_steps=5)

    assert loaded_states == [{"cursor": 5}]
    assert actor.global_steps == 5
    assert actor._train_state_ready is True
    assert metrics == {"streaming_dataloader/reissued_groups": 0.0}


def test_trainer_sample_delegates_to_dataloader_actor(monkeypatch):
    calls = []

    class _RemoteSample:
        def remote(self, *args):
            calls.append(args)
            return ("batch", {"metric": 1}, 2)

    stub = SimpleNamespace(streaming_dataloader=SimpleNamespace(sample=_RemoteSample()))
    monkeypatch.setattr(trainer_module.ray, "get", lambda value: value)

    result = PPOTrainer._sample_replay(
        stub,
        global_steps=4,
        partition_id="train",
        batch_size=2,
    )

    assert result == ("batch", {"metric": 1}, 2)
    assert calls == [(4, 2, "train", None, None, True)]
