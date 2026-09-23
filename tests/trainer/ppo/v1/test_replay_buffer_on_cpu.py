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
"""CPU tests for ReplayBuffer against a real TransferQueue instance.

Producers write every trajectory before marking its prompt terminal. Consumers run in a separate
thread when a test needs to observe blocking behavior. Off-policy staleness is measured in model
versions as ``global_steps - prompt_global_steps + 1``; ``drop`` and ``wait`` apply only to async
trainers.
"""

import threading
import time
import uuid
from dataclasses import dataclass, field

import pytest
import torch
import transfer_queue as tq
from transfer_queue import KVBatchMeta

from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer, ReplayBufferAsync

# Small poll interval so the blocking consumer reacts to producer writes quickly.
POLL_INTERVAL = 0.05


@pytest.fixture(scope="module")
def tq_init():
    tq.init()
    yield
    tq.close()


@pytest.fixture
def partition_id():
    """A unique partition per test to isolate TransferQueue state across tests."""
    return f"test-{uuid.uuid4().hex}"


def _make_rb(
    *,
    max_off_policy_threshold: int = 8,
    max_off_policy_strategy: str = "drop",
    poll_interval: float = POLL_INTERVAL,
    refill_fn=None,
    trainer_mode: str = "sync",
    filter_groups_metric: str | None = None,
    train_batch_size: int = 2,
    gen_batch_size: int = 1,
    max_inflight_gen_batches: int = 1,
    sync_refill_failed_groups: bool = False,
) -> ReplayBuffer:
    """Construct a ReplayBuffer with defaults that keep generic samples on-policy."""
    common_kwargs = dict(
        trainer_mode=trainer_mode,
        trainer_config={},
        max_off_policy_threshold=max_off_policy_threshold,
        max_off_policy_strategy=max_off_policy_strategy,
        sampler_kwargs={},
        poll_interval=poll_interval,
        filter_groups_metric=filter_groups_metric,
    )
    if trainer_mode != "sync":
        assert refill_fn is None
        assert sync_refill_failed_groups is False
        return ReplayBufferAsync(**common_kwargs)
    return ReplayBuffer(
        **common_kwargs,
        refill_fn=refill_fn,
        train_batch_size=train_batch_size,
        gen_batch_size=gen_batch_size,
        max_inflight_gen_batches=max_inflight_gen_batches,
        sync_refill_failed_groups=sync_refill_failed_groups,
    )


class FakeRefiller:
    """Produce fresh terminal prompts when the replay buffer requests replacements."""

    def __init__(self, partition_id: str, global_steps: int, sessions: int = 1, rewards: list[float] | None = None):
        self.partition_id = partition_id
        self.global_steps = global_steps
        self.sessions = sessions
        # When set, refilled groups carry these per-session rewards so they survive zero-variance filtering.
        self.rewards = rewards
        self.calls: list[int] = []
        self.produced_uids: list[str] = []

    def __call__(self, num_prompts: int) -> int:
        self.calls.append(num_prompts)
        specs = [
            PromptSpec(
                uid=_uid(),
                status="finished",
                sessions=self.sessions,
                global_steps=self.global_steps,
                rewards=self.rewards,
            )
            for _ in range(num_prompts)
        ]
        # Synchronous produce (already-terminal) so the sample loop sees them on its next poll.
        _produce(self.partition_id, specs).join_and_check()
        self.produced_uids.extend(spec.uid for spec in specs)
        return num_prompts


def _sample(rb: ReplayBuffer, partition_id: str, batch_size: int, global_steps: int = 0) -> KVBatchMeta:
    """Call ``sample`` and return just the batch (dropping the metrics dict)."""
    batch, _metrics = rb.sample(global_steps=global_steps, partition_id=partition_id, batch_size=batch_size)
    return batch


def _uid() -> str:
    # uid must not contain "_" because ReplayBuffer derives it via key.split("_")[0].
    return uuid.uuid4().hex


def _trajectory_key(uid: str, session_id: int = 0, index: int = 0) -> str:
    return f"{uid}_{session_id}_{index}"


def _set_prompt_status(partition_id: str, uid: str, status: str, global_steps: int) -> None:
    """Transition an existing prompt to a new status (e.g. running -> finished).

    Mirrors the rollout side flipping a GRPO group's status once it terminates.
    The prompt tag is updated in place; its trajectory values are untouched.
    """
    tq.kv_put(
        key=uid,
        partition_id=partition_id,
        tag={"is_prompt": True, "status": status, "global_steps": global_steps},
    )


@dataclass
class PromptSpec:
    """A prompt group and the trajectories that precede its status update."""

    uid: str
    status: str
    sessions: int = 1
    global_steps: int = 0
    rewards: list[float] | None = None
    trajectory_keys: list[str] = field(default_factory=list)


class RolloutProducer(threading.Thread):
    """Write complete trajectory groups before publishing their prompt status."""

    def __init__(self, partition_id: str, specs: list[PromptSpec]):
        super().__init__(daemon=True)
        self.partition_id = partition_id
        self.specs = specs
        self.error: Exception | None = None

    def run(self) -> None:
        try:
            for spec in self.specs:
                for session_id in range(spec.sessions):
                    key = _trajectory_key(spec.uid, session_id)
                    fields = {"input_ids": torch.tensor([1, 2, 3])}
                    tag = {"is_prompt": False, "seq_len": 3, "global_steps": spec.global_steps}
                    if spec.rewards is not None:
                        fields["extra_fields"] = {"reward_extra_info": {"acc": float(spec.rewards[session_id])}}
                    tq.kv_put(
                        key=key,
                        partition_id=self.partition_id,
                        fields=fields,
                        tag=tag,
                    )
                    spec.trajectory_keys.append(key)
                tq.kv_put(
                    key=spec.uid,
                    partition_id=self.partition_id,
                    tag={"is_prompt": True, "status": spec.status, "global_steps": spec.global_steps},
                )
        except Exception as e:  # surfaced to the test via join_and_check()
            self.error = e

    def join_and_check(self, timeout: float = 10.0) -> None:
        self.join(timeout)
        assert not self.is_alive(), "RolloutProducer thread did not finish in time"
        if self.error is not None:
            raise self.error


class SampleConsumer(threading.Thread):
    """Runs the blocking ``ReplayBuffer.sample`` in a background thread so the test
    can assert that it stays blocked until the producer supplies enough data."""

    def __init__(self, rb: ReplayBuffer, partition_id: str, batch_size: int, global_steps: int = 0):
        super().__init__(daemon=True)
        self.rb = rb
        self.partition_id = partition_id
        self.batch_size = batch_size
        self.global_steps = global_steps
        self.result: KVBatchMeta | None = None
        self.metrics: dict | None = None
        self.error: Exception | None = None

    def run(self) -> None:
        try:
            self.result, self.metrics = self.rb.sample(
                global_steps=self.global_steps,
                partition_id=self.partition_id,
                batch_size=self.batch_size,
            )
        except Exception as e:
            self.error = e

    def result_or_raise(self, timeout: float = 10.0) -> KVBatchMeta:
        self.join(timeout)
        assert not self.is_alive(), "SampleConsumer thread did not finish in time"
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


def _produce(partition_id: str, specs: list[PromptSpec]) -> RolloutProducer:
    producer = RolloutProducer(partition_id, specs)
    producer.start()
    return producer


def _clear_partition(partition_id: str) -> None:
    """Best-effort cleanup of every key written into a partition."""
    keys = list(tq.kv_list(partition_id=partition_id).get(partition_id, {}).keys())
    if keys:
        tq.kv_clear(keys=keys, partition_id=partition_id)


def _uids_of(keys: list[str]) -> set[str]:
    return {key.split("_")[0] for key in keys}


def test_async_poll_once_and_materialize_are_nonblocking_primitives(tq_init, partition_id):
    finished = PromptSpec(uid=_uid(), status="finished")
    _produce(partition_id, [finished]).join_and_check()
    rb = _make_rb(trainer_mode="separate_async")

    try:
        sampleable, metrics, ready, evicted_count = rb.poll_once(
            global_steps=0,
            partition_id=partition_id,
            target_count=1,
        )
        assert ready is True
        assert metrics == {}
        assert evicted_count == 0
        batch = rb.materialize_sampleable(0, partition_id, sampleable, 1)
        assert _uids_of(batch.keys) == {finished.uid}
    finally:
        _clear_partition(partition_id)


# --------------------------------------------------------------------------- #
# __init__: configuration validation.
# --------------------------------------------------------------------------- #


def test_init_rejects_non_positive_threshold():
    """max_off_policy_threshold must be a positive integer."""
    with pytest.raises(AssertionError, match="max off policy threshold"):
        _make_rb(max_off_policy_threshold=0)


def test_init_rejects_unknown_strategy():
    """max_off_policy_strategy must be one of {drop, wait}."""
    with pytest.raises(AssertionError, match="max off policy strategy"):
        _make_rb(max_off_policy_strategy="bogus")


def test_init_rejects_non_positive_sync_dapo_inflight_limit():
    with pytest.raises(ValueError, match="max_inflight_gen_batches"):
        _make_rb(
            refill_fn=lambda _n: None,
            filter_groups_metric="acc",
            max_inflight_gen_batches=0,
        )


# --------------------------------------------------------------------------- #
# _sync_metadata_from_transfer_queue: classification of polled metadata.
# --------------------------------------------------------------------------- #


def test_sync_metadata_classifies_keys(tq_init, partition_id):
    """The poll splits prompts by status and collects trajectory tags."""
    pending = PromptSpec(uid=_uid(), status="pending", sessions=0)
    running = PromptSpec(uid=_uid(), status="running", sessions=1)
    finished = PromptSpec(uid=_uid(), status="finished", sessions=2)
    failure = PromptSpec(uid=_uid(), status="failure", sessions=1)
    _produce(partition_id, [pending, running, finished, failure]).join_and_check()

    rb = _make_rb()
    try:
        rb._sync_metadata_from_transfer_queue()

        assert rb.pending_keys[partition_id] == {pending.uid}
        assert rb.running_keys[partition_id] == {running.uid}
        assert rb.finished_keys[partition_id] == {finished.uid}
        assert rb.failure_keys[partition_id] == {failure.uid}

        # All trajectory keys (and only those) land in the partition value map.
        expected_traj = set(running.trajectory_keys) | set(finished.trajectory_keys) | set(failure.trajectory_keys)
        assert set(rb.partitions[partition_id].keys()) == expected_traj
        for key in expected_traj:
            assert rb.partitions[partition_id][key] == {"is_prompt": False, "seq_len": 3, "global_steps": 0}
    finally:
        _clear_partition(partition_id)


def test_sync_metadata_unknown_status_raises(tq_init, partition_id):
    """An unrecognized prompt status is a hard error during the poll."""
    _produce(partition_id, [PromptSpec(uid=_uid(), status="bogus", sessions=0)]).join_and_check()

    rb = _make_rb()
    try:
        with pytest.raises(ValueError, match="Unknown status"):
            rb._sync_metadata_from_transfer_queue()
    finally:
        # The bogus prompt must be removed: every poll lists *all* partitions, so
        # leaving it behind would break unrelated tests.
        _clear_partition(partition_id)


def test_async_unfinished_session_count_uses_prompt_progress():
    rb = _make_rb(trainer_mode="colocate_async")
    partition_id = "train"
    rb.pending_keys[partition_id] = {"pending"}
    rb.running_keys[partition_id] = {"running"}
    rb.partitions[partition_id] = {
        "running_0_0": {"status": "success"},
        "running_1_0": {"status": "success"},
    }
    rb.finished_session_keys[partition_id] = {"running_0", "running_1"}
    rb.failure_session_keys[partition_id] = {"running_2"}

    assert rb.unfinished_session_count(partition_id, default_rollout_n=4) == 5


def test_async_unfinished_session_count_rejects_excess_progress():
    rb = _make_rb(trainer_mode="colocate_async")
    partition_id = "train"
    rb.running_keys[partition_id] = {"running"}
    rb.finished_session_keys[partition_id] = {"running_0", "running_1"}

    with pytest.raises(RuntimeError, match="inconsistent session progress"):
        rb.unfinished_session_count(partition_id, default_rollout_n=1)


def test_session_metadata_is_not_materialized(tq_init, partition_id):
    uid = _uid()
    trajectory_key = _trajectory_key(uid, session_id=0)
    success_session_key = f"{uid}_0"
    failure_session_key = f"{uid}_1"
    tq.kv_put(
        key=uid,
        partition_id=partition_id,
        tag={"is_prompt": True, "status": "running", "global_steps": 0},
    )
    tq.kv_put(
        key=trajectory_key,
        partition_id=partition_id,
        fields={"input_ids": torch.tensor([1, 2, 3])},
        tag={"status": "success"},
    )
    tq.kv_put(
        key=success_session_key,
        partition_id=partition_id,
        tag={"is_session": True, "status": "finished", "session_id": 0},
    )
    tq.kv_put(
        key=failure_session_key,
        partition_id=partition_id,
        tag={"is_session": True, "status": "failure", "session_id": 1},
    )

    rb = _make_rb(trainer_mode="colocate_async")
    try:
        rb._sync_metadata_from_transfer_queue()

        assert set(rb.partitions[partition_id]) == {trajectory_key}
        assert rb.finished_session_keys[partition_id] == {success_session_key}
        assert rb.failure_session_keys[partition_id] == {failure_session_key}
        assert rb.unfinished_session_count(partition_id, default_rollout_n=2) == 0
    finally:
        _clear_partition(partition_id)


def test_clear_groups_updates_active_snapshot(monkeypatch):
    rb = _make_rb()
    pid, dropped_uid, kept_uid = "p", "drop", "keep"
    dropped_trajectory = _trajectory_key(dropped_uid)
    kept_trajectory = _trajectory_key(kept_uid)
    rb.partitions[pid] = {dropped_trajectory: {}, kept_trajectory: {}}
    for status_keys in (rb.pending_keys, rb.running_keys, rb.finished_keys, rb.failure_keys):
        status_keys[pid] |= {dropped_uid, kept_uid}
    rb.prompt_global_steps[pid] = {dropped_uid: 1, kept_uid: 2}
    rb._dapo_classification_cache[pid] = {dropped_uid: 0.0, kept_uid: None}

    cleared: list[set[str]] = []
    monkeypatch.setattr(tq, "kv_clear", lambda *, keys, partition_id: cleared.append(set(keys)))

    rb._clear_groups(pid, {dropped_uid})

    assert cleared == [{dropped_uid, dropped_trajectory}]
    assert rb.partitions[pid] == {kept_trajectory: {}}
    for status_keys in (rb.pending_keys, rb.running_keys, rb.finished_keys, rb.failure_keys):
        assert status_keys[pid] == {kept_uid}
    assert rb.prompt_global_steps[pid] == {kept_uid: 2}
    assert rb._dapo_classification_cache[pid] == {kept_uid: None}


# --------------------------------------------------------------------------- #
# sample: end-to-end against a real TransferQueue.
# --------------------------------------------------------------------------- #


def test_sample_returns_finished_and_failure_trajectories(tq_init, partition_id):
    """sample picks trajectories belonging to finished/failure prompts and clears
    the sampled prompt keys from TransferQueue."""
    finished = PromptSpec(uid=_uid(), status="finished", sessions=2)
    failure = PromptSpec(uid=_uid(), status="failure", sessions=1)
    # Running prompt's trajectory must NOT be sampled.
    running = PromptSpec(uid=_uid(), status="running", sessions=1)

    _produce(partition_id, [finished, failure, running]).join_and_check()
    expected_keys = set(finished.trajectory_keys) | set(failure.trajectory_keys)

    rb = _make_rb()
    try:
        batch = _sample(rb, partition_id, batch_size=2)

        assert batch.partition_id == partition_id
        assert set(batch.keys) == expected_keys
        assert len(batch.tags) == len(batch.keys)

        # The two sampled prompt keys are consumed from TransferQueue; the running
        # prompt and all trajectory values remain.
        remaining = tq.kv_list(partition_id=partition_id).get(partition_id, {})
        assert finished.uid not in remaining
        assert failure.uid not in remaining
        assert running.uid in remaining
    finally:
        _clear_partition(partition_id)


def test_sync_failure_without_trajectories_uses_padding_path_by_default(tq_init, partition_id):
    finished = PromptSpec(uid=_uid(), status="finished", sessions=1)
    empty_failure = PromptSpec(uid=_uid(), status="failure", sessions=0)
    _produce(partition_id, [finished, empty_failure]).join_and_check()

    rb = _make_rb()
    try:
        batch, metrics = rb.sample(global_steps=0, partition_id=partition_id, batch_size=2)

        assert _uids_of(batch.keys) == {finished.uid}
        assert metrics == {}
    finally:
        _clear_partition(partition_id)


def test_sync_all_empty_failures_raise_without_refill(tq_init, partition_id):
    failures = [PromptSpec(uid=_uid(), status="failure", sessions=0) for _ in range(2)]
    _produce(partition_id, failures).join_and_check()

    rb = _make_rb()
    try:
        with pytest.raises(RuntimeError, match="sync_refill_failed_groups"):
            rb.sample(global_steps=0, partition_id=partition_id, batch_size=2)
    finally:
        _clear_partition(partition_id)


def test_sample_prioritizes_smallest_global_steps(tq_init, partition_id):
    """When more prompts are ready than ``batch_size``, sample must hand out the
    oldest ones first (smallest ``global_steps``), leaving the newer surplus."""
    # Produced out of step order on purpose; selection must follow global_steps.
    oldest = PromptSpec(uid=_uid(), status="finished", sessions=1, global_steps=1)
    middle = PromptSpec(uid=_uid(), status="finished", sessions=1, global_steps=2)
    newest = PromptSpec(uid=_uid(), status="finished", sessions=1, global_steps=5)
    _produce(partition_id, [newest, oldest, middle]).join_and_check()

    rb = _make_rb()
    try:
        batch = _sample(rb, partition_id, batch_size=2)

        # The two smallest-step prompts are picked; the newest is left behind.
        assert _uids_of(batch.keys) == {oldest.uid, middle.uid}

        remaining = tq.kv_list(partition_id=partition_id).get(partition_id, {})
        assert oldest.uid not in remaining
        assert middle.uid not in remaining
        assert newest.uid in remaining
    finally:
        _clear_partition(partition_id)


def test_sample_blocks_until_enough_then_unblocks(tq_init, partition_id):
    """sample stays blocked while fewer than batch_size prompts are ready and
    returns once the producer thread supplies the missing group."""
    # One group ready up front -> not enough for batch_size=2.
    _produce(partition_id, [PromptSpec(uid=_uid(), status="finished", sessions=1)]).join_and_check()

    rb = _make_rb()
    consumer = SampleConsumer(rb, partition_id, batch_size=2)
    try:
        consumer.start()

        # Give the consumer time to poll a few times; it must still be blocked.
        time.sleep(POLL_INTERVAL * 5)
        assert consumer.is_alive(), "sample returned before batch_size prompts were ready"

        # The producer thread supplies a second group; sample can now complete.
        _produce(partition_id, [PromptSpec(uid=_uid(), status="finished", sessions=1)]).join_and_check()

        batch = consumer.result_or_raise()
        assert len(batch.keys) == 2
    finally:
        consumer.join(timeout=10.0)
        _clear_partition(partition_id)


def test_sync_grpo_step_returns_complete_groups(tq_init, partition_id):
    """A synchronous PPO/GRPO step submits batch_size prompts, each a GRPO group of
    n sessions; one sample must return every trajectory as whole groups."""
    n_prompts = 3
    n_sessions = 4  # GRPO rollout.n
    specs = [PromptSpec(uid=_uid(), status="finished", sessions=n_sessions) for _ in range(n_prompts)]
    _produce(partition_id, specs).join_and_check()

    rb = _make_rb()
    try:
        batch = _sample(rb, partition_id, batch_size=n_prompts)

        # Every prompt's full GRPO group is present, nothing more, nothing less.
        assert len(batch.keys) == n_prompts * n_sessions
        assert set(batch.keys) == {k for spec in specs for k in spec.trajectory_keys}
        # Each sampled prompt contributes exactly n_sessions trajectories.
        per_uid: dict[str, int] = {}
        for key in batch.keys:
            per_uid[key.split("_")[0]] = per_uid.get(key.split("_")[0], 0) + 1
        assert set(per_uid.values()) == {n_sessions}
    finally:
        _clear_partition(partition_id)


def test_async_overproduction_drains_in_batches_without_duplicates(tq_init, partition_id):
    """An async rollouter over-produces; sequential samples drain the surplus
    batch_size complete groups at a time without ever re-selecting a prompt.

    ``sample`` only re-polls TransferQueue while it is under-filled, and it clears
    just the sampled *prompt* keys (trajectory values stay). The real trainer
    re-polls metadata in the gap between two ``sample`` calls; we reproduce that
    gap deterministically by re-syncing so a follow-up ``sample`` cannot re-select
    consumed prompts.
    """
    n_prompts = 5
    n_sessions = 2
    batch_size = 2
    specs = [PromptSpec(uid=_uid(), status="finished", sessions=n_sessions) for _ in range(n_prompts)]
    _produce(partition_id, specs).join_and_check()
    all_uids = {spec.uid for spec in specs}

    rb = _make_rb()
    try:
        collected_keys: list[str] = []
        consumed_uids: set[str] = set()

        # Drain 2 + 2 + 1 prompts across three samples.
        for bs in (batch_size, batch_size, n_prompts - 2 * batch_size):
            batch = _sample(rb, partition_id, batch_size=bs)

            sampled_uids = _uids_of(batch.keys)
            assert len(sampled_uids) == bs
            assert len(batch.keys) == bs * n_sessions
            assert not (sampled_uids & consumed_uids), "a prompt was handed out twice"

            collected_keys.extend(batch.keys)
            consumed_uids |= sampled_uids
            # Mimic the trainer's metadata refresh between two sample() calls.
            rb._sync_metadata_from_transfer_queue()

        # The whole surplus was drained exactly once.
        assert consumed_uids == all_uids
        assert len(collected_keys) == n_prompts * n_sessions
        assert len(set(collected_keys)) == len(collected_keys)
    finally:
        _clear_partition(partition_id)


# --------------------------------------------------------------------------- #
# sample: off-policy "drop" strategy.
# --------------------------------------------------------------------------- #


def test_drop_poll_evicts_stale_groups_and_reports_metrics(tq_init, partition_id):
    """A drop poll evicts stale terminal groups without owning replacement dispatch."""
    # staleness = (global_steps - prompt_global_steps + 1); drop when staleness > threshold(=2).
    # At global_steps=5:
    #   stale    gs=0 -> 6 > 2 -> dropped (and refilled)
    #   boundary gs=4 -> 2 not > 2 -> kept (boundary is inclusive on the keep side)
    #   fresh    gs=5 -> 1 -> kept
    stale = PromptSpec(uid=_uid(), status="finished", sessions=1, global_steps=0)
    boundary = PromptSpec(uid=_uid(), status="finished", sessions=1, global_steps=4)
    fresh = PromptSpec(uid=_uid(), status="finished", sessions=1, global_steps=5)
    inflight = PromptSpec(uid=_uid(), status="running", sessions=1, global_steps=0)
    _produce(partition_id, [stale, boundary, fresh, inflight]).join_and_check()

    rb = _make_rb(
        trainer_mode="colocate_async",
        max_off_policy_strategy="drop",
        max_off_policy_threshold=2,
    )
    try:
        sampleable, metrics, ready, evicted_count = rb.poll_once(
            global_steps=5,
            partition_id=partition_id,
            target_count=3,
        )

        assert sampleable == {boundary.uid, fresh.uid}
        assert ready is False
        assert evicted_count == 1
        assert rb.refill_fn is None

        # The dropped group's prompt and trajectory keys are gone from TransferQueue.
        remaining = tq.kv_list(partition_id=partition_id).get(partition_id, {})
        assert stale.uid not in remaining
        assert stale.trajectory_keys[0] not in remaining

        # partition_id is a random test id (not "train") -> "validation" prefix.
        assert metrics["validation/off_policy/evicted_samples"] == 1
        assert metrics["validation/off_policy/evicted_samples_staleness/mean"] == 6
        assert metrics["validation/off_policy/evicted_samples_staleness/max"] == 6
        assert metrics["validation/off_policy/evicted_samples_staleness/min"] == 6
    finally:
        _clear_partition(partition_id)


def test_drop_uses_one_snapshot_per_poll_iteration(tq_init, partition_id):
    """A stale running prompt that finishes between polls is dropped on the next snapshot."""
    stale_finished = PromptSpec(uid=_uid(), status="finished", sessions=1, global_steps=0)
    stale_running = PromptSpec(uid=_uid(), status="running", sessions=1, global_steps=0)
    _produce(partition_id, [stale_finished, stale_running]).join_and_check()

    rb = _make_rb(
        trainer_mode="colocate_async",
        max_off_policy_strategy="drop",
        max_off_policy_threshold=1,
    )
    try:
        sampleable, first_metrics, ready, evicted_count = rb.poll_once(5, partition_id, 1)
        assert sampleable == set()
        assert ready is False
        assert evicted_count == 1
        assert first_metrics["validation/off_policy/evicted_samples"] == 1

        _set_prompt_status(partition_id, stale_running.uid, "finished", global_steps=0)
        sampleable, second_metrics, ready, evicted_count = rb.poll_once(5, partition_id, 1)
        assert sampleable == set()
        assert ready is False
        assert evicted_count == 1
        assert second_metrics["validation/off_policy/evicted_samples"] == 1
    finally:
        _clear_partition(partition_id)


# --------------------------------------------------------------------------- #
# sample: off-policy "wait" (dropless) strategy.
# --------------------------------------------------------------------------- #


def test_wait_poll_is_not_ready_until_stale_inflight_finishes(tq_init, partition_id):
    """wait holds back a full batch while a stale in-flight prompt exists."""
    threshold, g = 2, 5
    fresh = [PromptSpec(uid=_uid(), status="finished", sessions=1, global_steps=g) for _ in range(2)]
    # In-flight prompt: staleness (5 - 0 + 1) = 6 >= 2 -> blocks sampling.
    stale = PromptSpec(uid=_uid(), status="running", sessions=1, global_steps=0)
    _produce(partition_id, fresh + [stale]).join_and_check()

    rb = _make_rb(trainer_mode="colocate_async", max_off_policy_strategy="wait", max_off_policy_threshold=threshold)
    try:
        sampleable, metrics, ready, evicted_count = rb.poll_once(g, partition_id, 2)
        assert len(sampleable) == 2
        assert metrics == {}
        assert ready is False
        assert evicted_count == 0

        _set_prompt_status(partition_id, stale.uid, "finished", global_steps=0)
        sampleable, metrics, ready, evicted_count = rb.poll_once(g, partition_id, 2)
        assert ready is True
        assert evicted_count == 0
        batch = rb.materialize_sampleable(g, partition_id, sampleable, 2)
        assert len(_uids_of(batch.keys)) == 2
    finally:
        _clear_partition(partition_id)


def test_wait_keeps_stale_terminal_trajectories(tq_init, partition_id):
    """wait is dropless: a finished-but-very-stale group that ``drop`` would
    discard is still returned, with no drop metrics."""
    # staleness (100 - 0 + 1) = 101, far above threshold=2; "drop" would
    # remove it, "wait" keeps it. No in-flight prompts, so sampling never blocks.
    stale = PromptSpec(uid=_uid(), status="finished", sessions=1, global_steps=0)
    _produce(partition_id, [stale]).join_and_check()

    rb = _make_rb(trainer_mode="colocate_async", max_off_policy_strategy="wait", max_off_policy_threshold=2)
    try:
        sampleable, metrics, ready, evicted_count = rb.poll_once(100, partition_id, 1)
        assert ready is True
        assert evicted_count == 0
        batch = rb.materialize_sampleable(100, partition_id, sampleable, 1)

        assert set(batch.keys) == set(stale.trajectory_keys)
        assert metrics == {}
    finally:
        _clear_partition(partition_id)


def test_wait_does_not_block_when_inflight_is_fresh(tq_init, partition_id):
    """wait proceeds immediately when every in-flight prompt is below threshold,
    and never drops (no drop metrics)."""
    threshold, g = 4, 3
    finished = [PromptSpec(uid=_uid(), status="finished", sessions=1, global_steps=g) for _ in range(2)]
    # In-flight prompt staleness (3 - 3 + 1) = 1 < 4 -> does not block.
    inflight = PromptSpec(uid=_uid(), status="running", sessions=1, global_steps=g)
    _produce(partition_id, finished + [inflight]).join_and_check()

    rb = _make_rb(trainer_mode="colocate_async", max_off_policy_strategy="wait", max_off_policy_threshold=threshold)
    try:
        sampleable, metrics, ready, evicted_count = rb.poll_once(g, partition_id, 2)
        assert ready is True
        assert evicted_count == 0
        batch = rb.materialize_sampleable(g, partition_id, sampleable, 2)

        assert _uids_of(batch.keys) == {spec.uid for spec in finished}
        assert metrics == {}
    finally:
        _clear_partition(partition_id)


# --------------------------------------------------------------------------- #
# mode matrix: stale drop, DAPO, and rollout failure.
# --------------------------------------------------------------------------- #


def test_init_rejects_dapo_without_refill_fn():
    with pytest.raises(ValueError, match="requires refill_fn"):
        _make_rb(filter_groups_metric="acc", refill_fn=None)


def test_init_rejects_sync_failure_refill_without_refill_fn():
    with pytest.raises(ValueError, match="requires refill_fn"):
        _make_rb(sync_refill_failed_groups=True, refill_fn=None)


def test_init_rejects_batched_sync_failure_refill():
    with pytest.raises(ValueError, match="requires gen_batch_size=1"):
        _make_rb(sync_refill_failed_groups=True, refill_fn=lambda _n: None, gen_batch_size=2)


def test_validation_never_drops_terminal_groups(tq_init):
    partition_id = "val"
    _clear_partition(partition_id)
    stale_dapo = PromptSpec(uid=_uid(), status="finished", sessions=2, global_steps=0, rewards=[1.0, 1.0])
    stale_failure = PromptSpec(uid=_uid(), status="failure", global_steps=0)
    _produce(partition_id, [stale_dapo, stale_failure]).join_and_check()

    rb = _make_rb(
        trainer_mode="colocate_async",
        max_off_policy_strategy="drop",
        max_off_policy_threshold=1,
        filter_groups_metric="acc",
    )
    try:
        sampleable, metrics, ready, evicted_count = rb.poll_once(5, partition_id, 2)
        assert ready is True
        assert evicted_count == 0
        batch = rb.materialize_sampleable(5, partition_id, sampleable, 2)
        assert _uids_of(batch.keys) == {stale_dapo.uid, stale_failure.uid}
        assert metrics == {}
    finally:
        _clear_partition(partition_id)


def test_validation_ignores_sync_failure_refill(tq_init):
    partition_id = "val"
    _clear_partition(partition_id)
    finished = PromptSpec(uid=_uid(), status="finished")
    empty_failure = PromptSpec(uid=_uid(), status="failure", sessions=0)
    _produce(partition_id, [finished, empty_failure]).join_and_check()

    refiller = FakeRefiller(partition_id, global_steps=1)
    rb = _make_rb(sync_refill_failed_groups=True, refill_fn=refiller)
    try:
        batch, metrics = rb.sample(global_steps=1, partition_id=partition_id, batch_size=2)

        assert _uids_of(batch.keys) == {finished.uid}
        assert refiller.calls == []
        assert metrics == {}
    finally:
        _clear_partition(partition_id)


@pytest.mark.parametrize("strategy", ["drop", "wait"])
def test_sync_ignores_off_policy_strategy(tq_init, partition_id, strategy):
    stale = PromptSpec(uid=_uid(), status="finished", global_steps=0)
    fresh = PromptSpec(uid=_uid(), status="finished", global_steps=5)
    _produce(partition_id, [stale, fresh]).join_and_check()

    rb = _make_rb(max_off_policy_strategy=strategy, max_off_policy_threshold=1)
    try:
        batch, metrics = rb.sample(global_steps=5, partition_id=partition_id, batch_size=2)
        assert _uids_of(batch.keys) == {stale.uid, fresh.uid}
        assert metrics == {}
    finally:
        _clear_partition(partition_id)


def test_async_failure_poll_evicts_without_refilling(tq_init, partition_id):
    finished = PromptSpec(uid=_uid(), status="finished")
    failure = PromptSpec(uid=_uid(), status="failure")
    _produce(partition_id, [finished, failure]).join_and_check()

    rb = _make_rb(trainer_mode="colocate_async")
    try:
        sampleable, metrics, ready, evicted_count = rb.poll_once(1, partition_id, 2)
        assert sampleable == {finished.uid}
        assert ready is False
        assert evicted_count == 1
        assert rb.refill_fn is None
        assert metrics["validation/rollout_failure/evicted_samples"] == 1
    finally:
        _clear_partition(partition_id)


def test_sync_failure_refill_replaces_empty_failed_groups(tq_init, partition_id):
    finished = PromptSpec(uid=_uid(), status="finished")
    failure = PromptSpec(uid=_uid(), status="failure", sessions=0)
    _produce(partition_id, [finished, failure]).join_and_check()

    refiller = FakeRefiller(partition_id, global_steps=1)
    rb = _make_rb(sync_refill_failed_groups=True, refill_fn=refiller)
    try:
        batch, metrics = rb.sample(global_steps=1, partition_id=partition_id, batch_size=2)
        sampled_uids = _uids_of(batch.keys)

        assert len(sampled_uids) == 2
        assert finished.uid in sampled_uids
        assert failure.uid not in sampled_uids
        assert refiller.calls == [1]
        assert metrics["validation/rollout_failure/evicted_samples"] == 1
    finally:
        _clear_partition(partition_id)


def test_sync_failure_refill_replaces_multiple_empty_failed_groups(tq_init, partition_id):
    failures = [PromptSpec(uid=_uid(), status="failure", sessions=0) for _ in range(3)]
    _produce(partition_id, failures).join_and_check()

    refiller = FakeRefiller(partition_id, global_steps=1)
    rb = _make_rb(sync_refill_failed_groups=True, refill_fn=refiller, train_batch_size=3)
    try:
        batch, metrics = rb.sample(global_steps=1, partition_id=partition_id, batch_size=3)

        assert _uids_of(batch.keys) == set(refiller.produced_uids)
        assert refiller.calls == [3]
        assert metrics["validation/rollout_failure/evicted_samples"] == 3
    finally:
        _clear_partition(partition_id)


def test_sync_failure_refill_keeps_materializable_failed_groups(tq_init, partition_id):
    finished = PromptSpec(uid=_uid(), status="finished")
    failure = PromptSpec(uid=_uid(), status="failure", sessions=1)
    _produce(partition_id, [finished, failure]).join_and_check()

    refiller = FakeRefiller(partition_id, global_steps=1)
    rb = _make_rb(sync_refill_failed_groups=True, refill_fn=refiller)
    try:
        batch, metrics = rb.sample(global_steps=1, partition_id=partition_id, batch_size=2)

        assert _uids_of(batch.keys) == {finished.uid, failure.uid}
        assert refiller.calls == []
        assert metrics == {}
    finally:
        _clear_partition(partition_id)


def test_sync_failure_refill_does_not_drain_unrelated_inflight(tq_init, partition_id):
    finished = PromptSpec(uid=_uid(), status="finished")
    empty_failure = PromptSpec(uid=_uid(), status="failure", sessions=0)
    running = PromptSpec(uid=_uid(), status="running", sessions=0)
    _produce(partition_id, [finished, empty_failure, running]).join_and_check()

    refiller = FakeRefiller(partition_id, global_steps=1)
    rb = _make_rb(sync_refill_failed_groups=True, refill_fn=refiller)
    consumer = SampleConsumer(rb, partition_id, batch_size=1, global_steps=1)
    try:
        consumer.start()
        batch = consumer.result_or_raise(timeout=2)

        assert _uids_of(batch.keys) == {finished.uid}
        assert refiller.calls == []
        assert consumer.metrics is not None
        assert consumer.metrics["validation/rollout_failure/evicted_samples"] == 1
        remaining = tq.kv_list(partition_id=partition_id).get(partition_id, {})
        assert running.uid in remaining
    finally:
        if consumer.is_alive():
            _set_prompt_status(partition_id, running.uid, "finished", global_steps=1)
            consumer.join(timeout=2)
        _clear_partition(partition_id)


def test_async_dapo_poll_evicts_without_refilling(tq_init, partition_id):
    all_same = PromptSpec(uid=_uid(), status="finished", sessions=2, rewards=[1.0, 1.0])
    mixed = PromptSpec(uid=_uid(), status="finished", sessions=2, rewards=[0.0, 1.0])
    _produce(partition_id, [all_same, mixed]).join_and_check()

    rb = _make_rb(
        trainer_mode="colocate_async",
        filter_groups_metric="acc",
    )
    try:
        sampleable, metrics, ready, evicted_count = rb.poll_once(1, partition_id, 2)
        assert sampleable == {mixed.uid}
        assert ready is False
        assert evicted_count == 1
        assert rb.refill_fn is None
        assert metrics["validation/filter_groups/evicted_samples"] == 1
    finally:
        _clear_partition(partition_id)


def test_dapo_classification_cache_fetches_only_new_finished_groups(tq_init, partition_id, monkeypatch):
    first_filtered = PromptSpec(uid=_uid(), status="finished", sessions=2, rewards=[1.0, 1.0])
    mixed = PromptSpec(uid=_uid(), status="finished", sessions=2, rewards=[0.0, 1.0])
    _produce(partition_id, [first_filtered, mixed]).join_and_check()

    requested_keys: list[set[str]] = []
    original_kv_batch_get = tq.kv_batch_get

    def recording_kv_batch_get(*args, **kwargs):
        requested_keys.append(set(kwargs["keys"]))
        return original_kv_batch_get(*args, **kwargs)

    monkeypatch.setattr(tq, "kv_batch_get", recording_kv_batch_get)
    rb = _make_rb(filter_groups_metric="acc", refill_fn=lambda _n: None)
    try:
        rb._sync_metadata_from_transfer_queue()
        first_result = rb._dapo_filtered_keys(partition_id)
        assert first_result[0] == {first_filtered.uid}
        assert dict(first_result[1]) == {1.0: 1}
        assert rb._dapo_filtered_keys(partition_id) == first_result

        second_filtered = PromptSpec(uid=_uid(), status="finished", sessions=2, rewards=[0.0, 0.0])
        _produce(partition_id, [second_filtered]).join_and_check()
        rb._sync_metadata_from_transfer_queue()
        filtered_uids, filtered_counts = rb._dapo_filtered_keys(partition_id)

        assert filtered_uids == {first_filtered.uid, second_filtered.uid}
        assert dict(filtered_counts) == {0.0: 1, 1.0: 1}
        assert requested_keys == [
            set(first_filtered.trajectory_keys + mixed.trajectory_keys),
            set(second_filtered.trajectory_keys),
        ]
    finally:
        _clear_partition(partition_id)


def test_terminal_eviction_reasons_has_no_cross_call_hidden_state(tq_init, partition_id):
    """An interleaved validation lookup must not overwrite a pending training breakdown."""
    all_zero = PromptSpec(uid=_uid(), status="finished", sessions=2, rewards=[0.0, 0.0])
    all_one = PromptSpec(uid=_uid(), status="finished", sessions=2, rewards=[1.0, 1.0])
    _produce(partition_id, [all_zero, all_one]).join_and_check()

    rb = _make_rb(trainer_mode="colocate_async", filter_groups_metric="acc")
    try:
        rb._sync_metadata_from_transfer_queue()
        train_reasons = rb._terminal_eviction_reasons(global_steps=0, partition_id=partition_id)
        rb._terminal_eviction_reasons(global_steps=0, partition_id="val")

        _evicted, _stale_count, dapo_count, metrics = rb._evict_terminal_groups(
            global_steps=0, partition_id=partition_id, eviction_reasons=train_reasons
        )
        from verl.trainer.ppo.v1.replay_buffer import DAPO_FILTERED_REWARD_COUNTS_KEY

        assert dapo_count == 2
        assert metrics[DAPO_FILTERED_REWARD_COUNTS_KEY] == {0.0: 1, 1.0: 1}
    finally:
        _clear_partition(partition_id)


def test_accumulate_eviction_metrics_merges_filtered_reward_counts():
    """The dict-valued DAPO diagnostic accumulates additively across poll iterations."""
    from verl.trainer.ppo.v1.replay_buffer import (
        DAPO_FILTERED_REWARD_COUNTS_KEY,
        _accumulate_eviction_metrics,
    )

    acc: dict = {}
    _accumulate_eviction_metrics(
        acc,
        {DAPO_FILTERED_REWARD_COUNTS_KEY: {0.0: 3, 1.0: 1}, "training/filter_groups/evicted_samples": 4},
    )
    _accumulate_eviction_metrics(
        acc,
        {DAPO_FILTERED_REWARD_COUNTS_KEY: {0.0: 2}, "training/filter_groups/evicted_samples": 2},
    )

    assert acc[DAPO_FILTERED_REWARD_COUNTS_KEY] == {0.0: 5, 1.0: 1}
    assert acc["training/filter_groups/evicted_samples"] == 6


def test_accumulate_eviction_metrics_weights_staleness_by_stale_count():
    from verl.trainer.ppo.v1.replay_buffer import _accumulate_eviction_metrics

    acc: dict = {}
    _accumulate_eviction_metrics(
        acc,
        {
            "training/off_policy/evicted_samples": 1,
            "training/off_policy/evicted_samples_staleness/mean": 10.0,
            "training/off_policy/evicted_samples_staleness/max": 10.0,
            "training/off_policy/evicted_samples_staleness/min": 10.0,
        },
    )
    _accumulate_eviction_metrics(acc, {"training/filter_groups/evicted_samples": 2})
    _accumulate_eviction_metrics(
        acc,
        {
            "training/off_policy/evicted_samples": 3,
            "training/off_policy/evicted_samples_staleness/mean": 2.0,
            "training/off_policy/evicted_samples_staleness/max": 3.0,
            "training/off_policy/evicted_samples_staleness/min": 1.0,
        },
    )

    assert acc["training/off_policy/evicted_samples"] == 4
    assert acc["training/off_policy/evicted_samples_staleness/mean"] == pytest.approx(4.0)
    assert acc["training/off_policy/evicted_samples_staleness/max"] == 10.0
    assert acc["training/off_policy/evicted_samples_staleness/min"] == 1.0
    assert acc["training/filter_groups/evicted_samples"] == 2


def test_dapo_reports_filtered_reward_value_breakdown(tq_init, partition_id):
    """The DAPO diagnostic maps each no-signal group's shared metric value to a count."""
    from verl.trainer.ppo.v1.replay_buffer import DAPO_FILTERED_REWARD_COUNTS_KEY

    # Two groups collapse to acc=0.0, one to acc=1.0; one mixed group survives filtering.
    all_zero_a = PromptSpec(uid=_uid(), status="finished", sessions=2, rewards=[0.0, 0.0])
    all_zero_b = PromptSpec(uid=_uid(), status="finished", sessions=2, rewards=[0.0, 0.0])
    all_one = PromptSpec(uid=_uid(), status="finished", sessions=2, rewards=[1.0, 1.0])
    mixed = PromptSpec(uid=_uid(), status="finished", sessions=2, rewards=[0.0, 1.0])
    _produce(partition_id, [all_zero_a, all_zero_b, all_one, mixed]).join_and_check()

    rb = _make_rb(trainer_mode="colocate_async", filter_groups_metric="acc")
    try:
        sampleable, metrics, ready, evicted_count = rb.poll_once(1, partition_id, 1)
        assert sampleable == {mixed.uid}
        assert ready is False
        assert evicted_count == 3
        assert metrics["validation/filter_groups/evicted_samples"] == 3
        assert metrics[DAPO_FILTERED_REWARD_COUNTS_KEY] == {0.0: 2, 1.0: 1}
    finally:
        _clear_partition(partition_id)


def test_sync_dapo_refills_twice_and_clears_surplus(tq_init, partition_id):
    all_same = PromptSpec(uid=_uid(), status="finished", sessions=2, rewards=[0.0, 0.0])
    mixed = PromptSpec(uid=_uid(), status="finished", sessions=2, rewards=[0.0, 1.0])
    _produce(partition_id, [all_same, mixed]).join_and_check()

    refiller = FakeRefiller(partition_id, global_steps=1, sessions=2, rewards=[0.0, 1.0])
    rb = _make_rb(refill_fn=refiller, filter_groups_metric="acc")
    try:
        batch, metrics = rb.sample(global_steps=1, partition_id=partition_id, batch_size=2)
        sampled_uids = _uids_of(batch.keys)
        refill_uids = set(refiller.produced_uids)

        assert all_same.uid not in sampled_uids
        assert mixed.uid in sampled_uids
        assert refiller.calls == [2]
        assert len(sampled_uids & refill_uids) == 1
        assert metrics["validation/filter_groups/evicted_samples"] == 1
        assert metrics["validation/filter_groups/discarded_surplus_samples"] == 1

        surplus_uid = (refill_uids - sampled_uids).pop()
        remaining = tq.kv_list(partition_id=partition_id).get(partition_id, {})
        assert surplus_uid not in remaining
        assert _trajectory_key(surplus_uid, 0) not in remaining
        assert _trajectory_key(surplus_uid, 1) not in remaining
    finally:
        _clear_partition(partition_id)


def test_sync_dapo_and_failure_refill_credits_are_combined(tq_init, partition_id):
    all_same = PromptSpec(uid=_uid(), status="finished", sessions=2, rewards=[0.0, 0.0])
    empty_failure = PromptSpec(uid=_uid(), status="failure", sessions=0)
    mixed = PromptSpec(uid=_uid(), status="finished", sessions=2, rewards=[0.0, 1.0])
    _produce(partition_id, [all_same, empty_failure, mixed]).join_and_check()

    refiller = FakeRefiller(partition_id, global_steps=1, sessions=2, rewards=[0.0, 1.0])
    rb = _make_rb(
        refill_fn=refiller,
        filter_groups_metric="acc",
        sync_refill_failed_groups=True,
        train_batch_size=3,
    )
    try:
        batch, metrics = rb.sample(global_steps=1, partition_id=partition_id, batch_size=3)
        sampled_uids = _uids_of(batch.keys)

        assert len(sampled_uids) == 3
        assert mixed.uid in sampled_uids
        assert all_same.uid not in sampled_uids
        assert empty_failure.uid not in sampled_uids
        assert refiller.calls == [3]
        assert metrics["validation/filter_groups/evicted_samples"] == 1
        assert metrics["validation/rollout_failure/evicted_samples"] == 1
        assert metrics["validation/filter_groups/discarded_surplus_samples"] == 1
    finally:
        _clear_partition(partition_id)


def test_sync_dapo_does_not_refill_when_filtered_surplus_leaves_a_full_batch(tq_init, partition_id):
    all_same = PromptSpec(uid=_uid(), status="finished", sessions=2, rewards=[0.0, 0.0])
    mixed = [PromptSpec(uid=_uid(), status="finished", sessions=2, rewards=[0.0, 1.0]) for _ in range(2)]
    _produce(partition_id, [all_same, *mixed]).join_and_check()

    refiller = FakeRefiller(partition_id, global_steps=1, sessions=2, rewards=[0.0, 1.0])
    rb = _make_rb(
        refill_fn=refiller,
        filter_groups_metric="acc",
        train_batch_size=3,
    )
    try:
        batch, metrics = rb.sample(global_steps=1, partition_id=partition_id, batch_size=2)

        assert _uids_of(batch.keys) == {spec.uid for spec in mixed}
        assert refiller.calls == []
        assert metrics["validation/filter_groups/evicted_samples"] == 1
    finally:
        _clear_partition(partition_id)


def test_sync_dapo_streams_refill_credit_with_bounded_inflight(tq_init, partition_id):
    all_same = PromptSpec(uid=_uid(), status="finished", sessions=2, rewards=[0.0, 0.0])
    slow_valid = PromptSpec(uid=_uid(), status="running", sessions=2, rewards=[0.0, 1.0])
    _produce(partition_id, [all_same, slow_valid]).join_and_check()

    refiller = FakeRefiller(partition_id, global_steps=1, sessions=2, rewards=[0.0, 1.0])
    # Sync DAPO is on-policy, so max_off_policy_strategy is irrelevant here; the default (drop) is a NO-OP.
    rb = _make_rb(
        refill_fn=refiller,
        filter_groups_metric="acc",
        train_batch_size=2,
        gen_batch_size=1,
        max_inflight_gen_batches=1,
    )
    consumer = SampleConsumer(rb, partition_id, batch_size=2, global_steps=1)
    try:
        consumer.start()
        deadline = time.time() + 5
        while len(refiller.calls) < 2 and consumer.is_alive() and time.time() < deadline:
            time.sleep(POLL_INTERVAL)

        # One original prompt remains in flight, so the 2x credit is dispatched one slot at a time.
        assert refiller.calls == [1, 1]
        assert consumer.is_alive()

        _set_prompt_status(partition_id, slow_valid.uid, "finished", global_steps=0)
        batch = consumer.result_or_raise()

        assert len(_uids_of(batch.keys)) == 2
        assert all_same.uid not in _uids_of(batch.keys)
        assert consumer.metrics is not None
        assert consumer.metrics["validation/filter_groups/evicted_samples"] == 1
        assert consumer.metrics["validation/filter_groups/discarded_surplus_samples"] == 1
    finally:
        _clear_partition(partition_id)


def test_sync_dapo_discards_unsent_credit_when_batch_fills(tq_init, partition_id):
    all_same = PromptSpec(uid=_uid(), status="finished", sessions=2, rewards=[0.0, 0.0])
    slow_valid = PromptSpec(uid=_uid(), status="running", sessions=2, rewards=[0.0, 1.0])
    _produce(partition_id, [all_same, slow_valid]).join_and_check()

    refiller = FakeRefiller(partition_id, global_steps=1, sessions=2, rewards=[0.0, 1.0])

    def refill_and_finish_original(num_prompts: int) -> int:
        result = refiller(num_prompts)
        _set_prompt_status(partition_id, slow_valid.uid, "finished", global_steps=0)
        return result

    rb = _make_rb(
        refill_fn=refill_and_finish_original,
        filter_groups_metric="acc",
        train_batch_size=2,
        gen_batch_size=1,
        max_inflight_gen_batches=1,
    )
    try:
        batch, metrics = rb.sample(global_steps=1, partition_id=partition_id, batch_size=2)

        assert len(_uids_of(batch.keys)) == 2
        assert all_same.uid not in _uids_of(batch.keys)
        assert refiller.calls == [1]
        assert metrics["validation/filter_groups/evicted_samples"] == 1
        assert "validation/filter_groups/discarded_surplus_samples" not in metrics
    finally:
        _clear_partition(partition_id)


def test_overlapping_async_eviction_reasons_evict_once(tq_init, partition_id):
    stale_dapo = PromptSpec(uid=_uid(), status="finished", sessions=2, global_steps=0, rewards=[1.0, 1.0])
    _produce(partition_id, [stale_dapo]).join_and_check()

    rb = _make_rb(
        trainer_mode="colocate_async",
        max_off_policy_strategy="drop",
        max_off_policy_threshold=1,
        filter_groups_metric="acc",
    )
    try:
        sampleable, metrics, ready, evicted_count = rb.poll_once(5, partition_id, 1)
        assert sampleable == set()
        assert ready is False
        assert evicted_count == 1
        assert metrics["validation/off_policy/evicted_samples"] == 1
        assert metrics["validation/filter_groups/evicted_samples"] == 1
    finally:
        _clear_partition(partition_id)


# --------------------------------------------------------------------------- #
# Non-consuming async polls used by separate_async switching.
# --------------------------------------------------------------------------- #


def test_count_poll_evicts_stale_groups_without_consuming_fresh_groups(tq_init, partition_id):
    fresh = PromptSpec(uid=_uid(), status="finished", sessions=1, global_steps=5)
    stale = PromptSpec(uid=_uid(), status="finished", sessions=1, global_steps=0)
    _produce(partition_id, [fresh, stale]).join_and_check()

    rb = _make_rb(
        trainer_mode="colocate_async",
        max_off_policy_strategy="drop",
        max_off_policy_threshold=1,
    )
    try:
        sampleable, metrics, ready, evicted_count = rb.poll_once(5, partition_id, 0)
        assert sampleable == {fresh.uid}
        assert ready is False
        assert evicted_count == 1
        assert metrics["validation/off_policy/evicted_samples"] == 1
        remaining = tq.kv_list(partition_id=partition_id)[partition_id]
        assert fresh.uid in remaining
        assert stale.uid not in remaining
    finally:
        _clear_partition(partition_id)


def test_non_consuming_poll_leaves_groups_for_materialization(tq_init, partition_id):
    specs = [PromptSpec(uid=_uid(), status="finished", sessions=1) for _ in range(2)]
    _produce(partition_id, specs).join_and_check()

    rb = _make_rb(trainer_mode="colocate_async")
    try:
        sampleable, metrics, ready, evicted_count = rb.poll_once(0, partition_id, 2)
        assert metrics == {}
        assert ready is True
        assert evicted_count == 0
        assert sampleable == {spec.uid for spec in specs}

        sampleable, _, ready, _ = rb.poll_once(0, partition_id, 2)
        assert ready is True
        batch = rb.materialize_sampleable(0, partition_id, sampleable, 2)
        assert _uids_of(batch.keys) == {spec.uid for spec in specs}
    finally:
        _clear_partition(partition_id)


def test_non_consuming_poll_reports_not_ready_below_target(tq_init, partition_id):
    ready = PromptSpec(uid=_uid(), status="finished", sessions=1)
    pending = PromptSpec(uid=_uid(), status="running", sessions=1)
    _produce(partition_id, [ready, pending]).join_and_check()

    rb = _make_rb(trainer_mode="colocate_async")
    try:
        sampleable, metrics, is_ready, evicted_count = rb.poll_once(0, partition_id, 2)
        assert sampleable == {ready.uid}
        assert metrics == {}
        assert is_ready is False
        assert evicted_count == 0
        _set_prompt_status(partition_id, pending.uid, "finished", global_steps=0)
        sampleable, metrics, is_ready, evicted_count = rb.poll_once(0, partition_id, 2)
        assert sampleable == {ready.uid, pending.uid}
        assert metrics == {}
        assert is_ready is True
        assert evicted_count == 0
    finally:
        _clear_partition(partition_id)


def test_async_blocking_sample_api_is_rejected():
    rb = _make_rb(trainer_mode="colocate_async")
    with pytest.raises(RuntimeError, match="coordinator-driven"):
        rb.sample(global_steps=0, partition_id="train", batch_size=1)
