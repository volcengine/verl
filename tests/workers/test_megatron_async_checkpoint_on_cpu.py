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

"""Exercise checkpoint control flow without importing GPU-only Megatron kernels.

Compile the production methods with a fake training operation and async writer.
This tests queue progress and publication ordering, not distributed GPU I/O.
"""

import ast
import copy
import os
import queue
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from tensordict import TensorDict

from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import BaseCheckpointManager, get_checkpoint_tracker_filename
from verl.utils.fs import local_mkdir_safe
from verl.workers.engine_workers import TrainingWorker

ROOT = Path(__file__).resolve().parents[2]


def _load_methods(path, class_name, methods, namespace, base=object):
    tree = ast.parse((ROOT / path).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    cls.bases = [ast.Name(id="TestBase", ctx=ast.Load())]
    cls.decorator_list = []
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    assert {node.name for node in cls.body} == set(methods)
    module = ast.parse("from __future__ import annotations")
    module.body.append(cls)
    scope = dict(namespace, TestBase=base)
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    return scope[class_name]


def _load_noop_writer():
    """Use the production empty-rank writer in the simulated persistent caller."""
    path = ROOT / "verl/utils/checkpoint/megatron_checkpoint_manager.py"
    tree = ast.parse(path.read_text())
    writer = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_noop_async_checkpoint_write"
    )
    namespace = {}
    exec(compile(ast.Module(body=[writer], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_noop_async_checkpoint_write"]


class _Request:
    def __init__(self):
        self.async_fn = None
        self.async_fn_args = ()
        self.async_fn_kwargs = {}
        self.finalize_fns = []
        self.ready = False

    def add_finalize_fn(self, fn):
        self.finalize_fns.append(fn)

    def _replace(self, **updates):
        result = copy.copy(self)
        for name, value in updates.items():
            setattr(result, name, value)
        return result


class _Queue:
    def __init__(self):
        self.pending = []
        self.scheduled = 0

    def schedule_async_request(self, request):
        self.pending.append(request)
        self.scheduled += 1

    def maybe_finalize_async_calls(self, blocking=False):
        while self.pending and (blocking or self.pending[0].ready):
            for fn in self.pending.pop(0).finalize_fns:
                fn()


def _manager():
    namespace = dict(
        os=os,
        logger=None,
        log_with_rank=lambda *args, **kwargs: None,
        local_mkdir_safe=local_mkdir_safe,
        get_checkpoint_tracker_filename=get_checkpoint_tracker_filename,
        _noop_async_checkpoint_write=_load_noop_writer(),
    )
    for content in ("model", "optimizer", "extra"):
        namespace[f"get_{content}_dist_checkpoint_path"] = lambda root, content=content: os.path.join(
            root, content, "dist_ckpt"
        )
    cls = _load_methods(
        "verl/utils/checkpoint/megatron_checkpoint_manager.py",
        "MegatronCheckpointManager",
        [
            "save_checkpoint",
            "_schedule_dist_checkpoint_saves",
            "_dispatch_finalize",
            "_maybe_finalize_async_save",
            "_finalize_save",
        ],
        namespace,
    )
    manager = cls()
    manager.checkpoint_config = SimpleNamespace(async_save=True)
    manager.rank = 0
    manager.use_megatron_fsdp = False
    manager.should_save_optimizer = False
    manager.should_save_dist_ckpt_model = False
    manager.should_save_hf_model = False
    manager.should_save_extra = True
    manager._build_sharded_state_dict_metadata = lambda: {}
    manager._build_extra_state_dict = lambda: {"rng_state": 1}
    manager._save_transformer_config = lambda path: None
    manager._save_dist_checkpoint = lambda path, state: _Request()
    manager._write_checkpoint_manifest = lambda path, *args: (Path(path) / "ckpt_contents.json").write_text("ready")
    manager.register_checkpoint = MagicMock()
    manager._async_calls_queue = _Queue()
    return manager


def _engine(manager, output=None):
    class BaseEngine:
        def train_batch(self, data, loss_function):
            loss_function()
            return output

    cls = _load_methods(
        "verl/workers/engine/megatron/transformer_impl.py",
        "MegatronEngine",
        ["train_batch", "save_checkpoint"],
        dict(
            tu=tu,
            torch=SimpleNamespace(distributed=SimpleNamespace(barrier=lambda: None)),
            get_megatron_module_device=lambda module: "cuda",
            load_megatron_model_to_gpu=lambda *args, **kwargs: None,
            offload_megatron_model_to_cpu=lambda module: None,
        ),
        base=BaseEngine,
    )
    engine = cls()
    engine.checkpoint_config = manager.checkpoint_config
    engine.checkpoint_mananager = manager
    engine._is_offload_param = False
    engine.module = []
    return engine


@pytest.mark.parametrize("relative_path", ["global_step_3", "global_step_3/actor"])
def test_blocking_save_waits_for_older_and_current_publication(tmp_path, relative_path):
    manager = _manager()
    previous = str(tmp_path / relative_path.replace("step_3", "step_2"))
    current = str(tmp_path / relative_path)
    manager.save_checkpoint(previous, global_step=2)
    tracker = tmp_path / "latest_checkpointed_iteration.txt"
    assert not tracker.exists()
    worker = SimpleNamespace(engine=_engine(manager))

    TrainingWorker.save_checkpoint(worker, current, global_step=3, blocking=True)

    assert not manager._async_calls_queue.pending
    assert manager._async_calls_queue.scheduled == 2
    assert (Path(previous) / "ckpt_contents.json").exists()
    assert (Path(current) / "ckpt_contents.json").exists()
    assert tracker.read_text() == "3"
    assert manager.register_checkpoint.call_count == 2


@pytest.mark.parametrize("has_output", [False, True])
@pytest.mark.parametrize("opt_in", [False, True])
def test_training_progresses_ready_save_without_scheduling_another(tmp_path, has_output, opt_in):
    manager = _manager()
    manager.save_checkpoint(str(tmp_path / "global_step_2"), global_step=2)
    tracker = tmp_path / "latest_checkpointed_iteration.txt"
    output = {"metrics": {}} if has_output else None
    engine = _engine(manager, output)
    data = TensorDict({}, batch_size=[])
    if opt_in:
        tu.assign_non_tensor(data, finalize_async_checkpoint=True)

    # An unfinished write must not hold up the training step or be published.
    assert engine.train_batch(data, lambda: None) is output
    assert not tracker.exists()

    def finish_writes_during_training():
        assert not tracker.exists()
        for request in manager._async_calls_queue.pending:
            request.ready = True

    assert engine.train_batch(data, finish_writes_during_training) is output
    assert tracker.exists() is opt_in
    if opt_in:
        assert tracker.read_text() == "2"
    # RL requests without the SFT opt-in leave completion to their coordinator.
    assert manager._async_calls_queue.scheduled == 1


def test_failed_training_does_not_enter_checkpoint_collectives():
    manager = _manager()
    manager._maybe_finalize_async_save = MagicMock()
    engine = _engine(manager)
    data = TensorDict({}, batch_size=[])
    tu.assign_non_tensor(data, finalize_async_checkpoint=True)

    def fail():
        raise RuntimeError("training failed")

    with pytest.raises(RuntimeError, match="training failed"):
        engine.train_batch(data, fail)
    manager._maybe_finalize_async_save.assert_not_called()


@pytest.mark.parametrize("checkpoint_config", [None, SimpleNamespace(async_save=False)])
def test_training_without_async_checkpointing_does_not_progress_queue(checkpoint_config):
    manager = _manager()
    manager.checkpoint_config = checkpoint_config
    manager._maybe_finalize_async_save = MagicMock()
    engine = _engine(manager)
    data = TensorDict({}, batch_size=[])
    tu.assign_non_tensor(data, finalize_async_checkpoint=True)

    engine.train_batch(data, lambda: None)
    manager._maybe_finalize_async_save.assert_not_called()


def test_manager_initialization_without_checkpoint_config():
    manager_cls = _load_methods(
        "verl/utils/checkpoint/megatron_checkpoint_manager.py",
        "MegatronCheckpointManager",
        ["__init__", "should_save_hf_model", "should_load_hf_model"],
        dict(torch=torch),
        base=BaseCheckpointManager,
    )
    with (
        patch.object(torch.distributed, "get_rank", return_value=0),
        patch.object(torch.distributed, "get_world_size", return_value=1),
    ):
        manager = manager_cls(
            config=SimpleNamespace(model=SimpleNamespace(path="unused")),
            checkpoint_config=None,
            model_config=None,
            transformer_config=None,
            role="actor",
            model=[],
            arch="unused",
            hf_config=None,
            param_dtype=torch.float32,
            share_embeddings_and_output_weights=False,
            processing_class=None,
            optimizer=None,
            optimizer_scheduler=None,
            use_distributed_optimizer=False,
            bridge=object(),
        )

    assert manager._async_calls_queue is None
    assert manager.checkpoint_save_contents == ["model", "optimizer", "extra"]


def test_async_content_results_match_their_write_plans():
    """A faster extra write must not supply the optimizer's storage offsets.

    Model MCore's shared result queue and FIFO finalizers, with an earlier
    optimizer write deliberately delayed until the extra write can finish.
    Exercise two checkpoints so ordering also holds across save calls.
    """
    results = queue.Queue()
    second_written = threading.Event()
    metadata = {}
    published = []

    class SharedResultsQueue:
        def __init__(self, persistent=False):
            self.executor = ThreadPoolExecutor(max_workers=1 if persistent else 4)
            self.pending = []

        def schedule_async_request(self, request):
            self.pending.append((self.executor.submit(request.async_fn), request))

        def maybe_finalize_async_calls(self, blocking=False):
            while self.pending and (blocking or self.pending[0][0].done()):
                future, request = self.pending.pop(0)
                future.result()
                for finalize in request.finalize_fns:
                    finalize()

    module_name = "megatron.core.dist_checkpointing.strategies.async_utils"
    module = ModuleType(module_name)
    module.AsyncCallsQueue = SharedResultsQueue
    manager_cls = _load_methods(
        "verl/utils/checkpoint/megatron_checkpoint_manager.py",
        "MegatronCheckpointManager",
        [
            "__init__",
            "should_save_hf_model",
            "should_load_hf_model",
            "_dispatch_finalize",
            "_maybe_finalize_async_save",
        ],
        dict(torch=torch, _noop_async_checkpoint_write=lambda: None),
        base=BaseCheckpointManager,
    )
    with (
        patch.dict(sys.modules, {module_name: module}),
        patch.object(torch.distributed, "get_rank", return_value=0),
        patch.object(torch.distributed, "get_world_size", return_value=1),
    ):
        manager = manager_cls(
            config=SimpleNamespace(model=SimpleNamespace(path="unused")),
            checkpoint_config=CheckpointConfig(
                async_save=True, save_contents=["optimizer", "extra"], load_contents=["optimizer", "extra"]
            ),
            model_config=None,
            transformer_config=None,
            role="actor",
            model=[],
            arch="unused",
            hf_config=None,
            param_dtype=torch.float32,
            share_embeddings_and_output_weights=False,
            processing_class=None,
            optimizer=None,
            optimizer_scheduler=None,
            use_distributed_optimizer=False,
        )

    def request_for(content):
        request = _Request()

        def write():
            if content == "step1/optimizer":
                second_written.wait(timeout=0.2)
            results.put(content)
            if content == "step1/extra":
                second_written.set()

        request.async_fn = write
        request.add_finalize_fn(lambda: metadata.update({content: results.get_nowait()}))
        return request

    try:
        for step in (1, 2):
            manager._dispatch_finalize(
                [request_for(f"step{step}/optimizer"), request_for(f"step{step}/extra")],
                lambda step=step: published.append(step),
            )
        manager._maybe_finalize_async_save(blocking=True)
    finally:
        manager._async_calls_queue.executor.shutdown(wait=True)

    assert metadata == {f"step{s}/{c}": f"step{s}/{c}" for s in (1, 2) for c in ("optimizer", "extra")}
    assert published == [1, 2]
    assert results.empty()


@pytest.mark.parametrize(
    "args,kwargs",
    [((), {}), (("unused",), {}), (("unused",), {"ignored": True})],
)
def test_empty_rank_requests_keep_persistent_completion_events_aligned(args, kwargs):
    """An empty content must still complete before publishing on this rank."""
    completed = queue.Queue()

    class PersistentQueue:
        def __init__(self):
            self.pending = []

        def schedule_async_request(self, request):
            self.pending.append(request)
            # MCore's persistent caller does not enqueue async_fn=None.
            if request.async_fn is not None:
                request.async_fn(*request.async_fn_args, **request.async_fn_kwargs)
                completed.put(True)

        def maybe_finalize_async_calls(self, blocking=False):
            while self.pending:
                if completed.empty():
                    raise TimeoutError("Persistent caller is waiting for an absent completion event")
                completed.get_nowait()
                for fn in self.pending.pop(0).finalize_fns:
                    fn()

    manager = _manager()
    manager._async_calls_queue = PersistentQueue()
    requests = [_Request() for _ in range(3)]
    requests[0].async_fn = lambda: None
    requests[1].async_fn_args = args
    requests[1].async_fn_kwargs = kwargs
    requests[2].async_fn = lambda: None
    published = []

    manager._dispatch_finalize(requests, lambda: published.append(True))
    manager._maybe_finalize_async_save(blocking=True)

    assert published == [True]
    assert not manager._async_calls_queue.pending
    assert completed.empty()


@pytest.mark.parametrize("blocking", [False, True])
def test_async_save_without_distributed_requests_publishes_immediately(tmp_path, blocking):
    manager = _manager()
    manager._build_extra_state_dict = lambda: {}
    manager.save_checkpoint(str(tmp_path / "global_step_2"), global_step=2, blocking=blocking)

    assert manager._async_calls_queue.scheduled == 0
    assert (tmp_path / "latest_checkpointed_iteration.txt").read_text() == "2"


def test_non_megatron_async_save_rejected_before_engine_construction():
    config = SimpleNamespace(
        model_config={},
        engine_config=SimpleNamespace(strategy="fsdp"),
        optimizer_config=None,
        checkpoint_config=SimpleNamespace(async_save=True),
    )
    with (
        patch("verl.workers.engine_workers.Worker.__init__", return_value=None),
        patch("verl.workers.engine_workers.initialize_global_process_group_ray"),
        patch("verl.workers.engine_workers.set_numa_affinity"),
        pytest.raises(NotImplementedError, match="fsdp does not support checkpoint.async_save"),
    ):
        TrainingWorker(config)
