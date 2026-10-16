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
"""Exercise production micro-batch loops without optional accelerator dependencies.

AST loading compiles the original method bodies, replacing only imports and
decorators. Forward/backward and native profiler calls are test doubles: these
tests verify control flow, not distributed collectives or hardware trace output.
"""

import ast
import functools
import importlib.util
import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]


def load_class(path, name, namespace, members=None):
    """Load original class bodies without importing their optional backends."""
    tree = ast.parse((ROOT / path).read_text())
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
    node.decorator_list = []
    if members is not None:
        node.body = [
            child
            for child in node.body
            if isinstance(child, ast.Assign | ast.AnnAssign)
            or isinstance(child, ast.FunctionDef)
            and child.name in members
        ]
    for child in node.body:
        if isinstance(child, ast.FunctionDef):
            child.decorator_list = [
                decorator
                for decorator in child.decorator_list
                if isinstance(decorator, ast.Name) and decorator.id in ("classmethod", "staticmethod", "property")
            ]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(ROOT / path), "exec"), namespace)
    return namespace[name]


def schedule_config(active=1):
    values = dict(skip_first=3, wait=0, warmup=1, active=active, repeat=1)
    return SimpleNamespace(enabled=active > 0, to_torch_kwargs=lambda: values)


class MicroBatchCallbackTests(unittest.TestCase):
    def setUp(self):
        self.base = load_class("verl/workers/engine/base.py", "BaseEngine", {})
        self.events = []
        self.prepare = Mock(side_effect=lambda data, **kwargs: (data["micro_batches"], None))
        self.tu = SimpleNamespace(
            assign_non_tensor=lambda data, **kwargs: data.update(kwargs),
            assign_non_tensor_data=Mock(),
            get_non_tensor_data=lambda data, key, default=None: data.get(key, default),
            pop=lambda data, key, default=None: data.pop(key, default),
        )
        self.namespace = {
            "BaseEngine": self.base,
            "nullcontext": nullcontext,
            "torch": SimpleNamespace(
                no_grad=nullcontext,
                distributed=SimpleNamespace(all_reduce=Mock(), ReduceOp=SimpleNamespace(SUM="sum")),
            ),
            "tu": self.tu,
            "get_device_id": lambda: "cpu",
            "prepare_micro_batches": self.prepare,
            "postprocess_batch_func": lambda output_lst, **kwargs: output_lst,
            "prepare_for_grad_accumulation": Mock(),
            "prepare_for_final_backward": Mock(),
            "parallel_state": SimpleNamespace(get_parallel_state=lambda: SimpleNamespace(ulysses_size=1)),
            "RouterReplayAction": SimpleNamespace(RECORD="record"),
        }

    def make_engine(self, backend, fail=None, scaled=False):
        names = dict(
            fsdp="FSDPEngine", automodel="AutomodelEngine", torchtitan="TorchTitanEngine", veomni="VeOmniEngine"
        )
        if backend == "veomni":
            self.namespace["FSDPEngine"] = load_class(
                "verl/workers/engine/fsdp/transformer_impl.py",
                "FSDPEngine",
                dict(self.namespace),
                members=["forward_backward_batch"],
            )
        cls = load_class(
            f"verl/workers/engine/{backend}/transformer_impl.py",
            names[backend],
            dict(self.namespace),
            members=["forward_backward_batch"],
        )
        engine = cls()
        engine.get_data_parallel_group = lambda: None
        engine.get_data_parallel_size = lambda: 1
        engine.get_data_parallel_rank = lambda: 0
        engine.is_mp_src_rank_with_outputs = lambda: False
        engine.train_mode = lambda **kwargs: nullcontext()
        engine.engine_config = SimpleNamespace(ep_size=1, tensor_parallel_size=1)
        engine.ulysses_sequence_parallel_size = 1
        engine.module = object()
        engine.trainer = SimpleNamespace(train_context=nullcontext)
        engine.model_fwd_context = nullcontext()
        engine.model_bwd_context = nullcontext()
        engine.enable_routing_replay = True
        engine._router_replay_mode = "R3"
        engine._router_replay = Mock(action="replay")
        engine.scaler = SimpleNamespace(scale=lambda loss: loss) if scaled else None

        def forward(micro_batch, **kwargs):
            self.events.append(("forward", micro_batch))
            if fail == ("forward", micro_batch):
                raise RuntimeError("forward failed")

            def backward():
                self.events.append(("backward", micro_batch))
                if fail == ("backward", micro_batch):
                    raise RuntimeError("backward failed")

            return SimpleNamespace(backward=backward), {"loss": 1}

        engine.forward_step = forward
        return engine

    @staticmethod
    def data(count=3):
        return {"loss_mask": Mock(), "micro_batches": list(range(count))}

    def test_success_inference_and_failures_on_all_backends(self):
        for backend in ("fsdp", "automodel", "torchtitan", "veomni"):
            for forward_only, fail in ((False, None), (True, None), (False, ("forward", 1)), (False, ("backward", 1))):
                with self.subTest(backend=backend, forward_only=forward_only, fail=fail):
                    self.events.clear()
                    engine = self.make_engine(backend, fail=fail, scaled=backend == "fsdp")
                    engine.set_micro_batch_callback(lambda: self.events.append(("step", None)))
                    data = self.data()
                    if backend == "veomni":
                        data["enable_routing_replay"] = True
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, "failed"):
                            engine.forward_backward_batch(data, None, forward_only=forward_only)
                        self.assertEqual(self.events.count(("step", None)), 1)
                    else:
                        engine.forward_backward_batch(data, None, forward_only=forward_only)
                        expected = []
                        for i in range(3):
                            expected.append(("forward", i))
                            if not forward_only:
                                expected.extend([("backward", i), ("step", None)])
                        self.assertEqual(self.events, expected)
                    self.assertTrue(self.prepare.call_args.kwargs["same_micro_num_in_dp"])
                    if backend == "veomni":
                        engine._router_replay.clear.assert_called_once()

    def test_callback_defaults_clear_and_instance_isolation(self):
        first = self.make_engine("fsdp")
        second = self.make_engine("fsdp")
        callback = Mock()
        first.set_micro_batch_callback(callback)
        second.forward_backward_batch(self.data(), None)
        callback.assert_not_called()
        first.forward_backward_batch(self.data(), None)
        self.assertEqual(callback.call_count, 3)
        first.set_micro_batch_callback(None)
        first.forward_backward_batch(self.data(), None)
        self.assertEqual(callback.call_count, 3)
        with self.assertRaises(NotImplementedError):
            self.base().set_micro_batch_callback(callback)

    def make_worker(self, engine, enable=True, active=1, forward_only=False, rank=0):
        class Worker:
            def __init__(self):
                self.rank = rank

            def _register_dispatch_collect_info(self, **kwargs):
                pass

        class Extension:
            def __init__(self, profiler):
                self.profiler = profiler

        namespace = dict(
            self.namespace,
            Worker=Worker,
            DistProfilerExtension=Extension,
            initialize_global_process_group_ray=lambda **kwargs: None,
            set_numa_affinity=lambda: None,
            get_device_name=lambda: "cpu",
            DistProfiler=lambda **kwargs: Mock(),
            maybe_fix_3d_position_ids=lambda data: None,
            NonTensorData=lambda value: value,
            Timer=lambda **kwargs: nullcontext(),
        )
        worker_cls = load_class(
            "verl/workers/engine_workers.py", "TrainingWorker", namespace, members=["__init__", "train_mini_batch"]
        )
        tool = SimpleNamespace(schedule=schedule_config(active))
        config = SimpleNamespace(
            model_type="language_model",
            model_config=SimpleNamespace(get=lambda key, default=None: default),
            engine_config=SimpleNamespace(strategy="test", forward_only=forward_only),
            optimizer_config=None,
            checkpoint_config=None,
            profiler_config=SimpleNamespace(enable=enable, tool="torch", tool_config={"torch": tool}),
        )
        module = SimpleNamespace(BaseEngine=self.base, EngineRegistry=SimpleNamespace(new=lambda **kwargs: engine))
        with patch.dict(sys.modules, {"verl.workers.engine": module}):
            return worker_cls(config)

    def test_worker_registers_on_all_ranks_without_mini_batch_double_counting(self):
        for rank in (0, 1):
            with self.subTest(rank=rank):
                engine = self.make_engine("fsdp")
                worker = self.make_worker(engine, rank=rank)
                worker.train_batch = lambda data, engine=engine: engine.forward_backward_batch(data, None)
                self.tu.make_iterator = lambda data, epochs, **kwargs: [
                    self.data(count) for _ in range(epochs) for count in (2, 3)
                ]

                class Batch(dict):
                    shape = (6,)

                worker.train_mini_batch(Batch(num_mini_batch=2, epochs=2))
                self.assertEqual(worker.profiler.step.call_count, 10)

    def test_worker_rejects_unsupported_schedules_only_for_enabled_training(self):
        for enable, active, forward_only in ((False, 1, False), (True, 0, False), (True, 1, True), (True, 1, False)):
            with self.subTest(enable=enable, active=active, forward_only=forward_only):
                engine = self.make_engine("fsdp")
                engine.supports_micro_batch_callback = False
                if enable and active and not forward_only:
                    with self.assertRaisesRegex(NotImplementedError, "schedule.active=0"):
                        self.make_worker(engine, enable=enable, active=active, forward_only=forward_only)
                else:
                    worker = self.make_worker(engine, enable=enable, active=active, forward_only=forward_only)
                    engine.on_micro_batch_end()
                    worker.profiler.step.assert_not_called()


class ProfilerLifecycleTests(unittest.TestCase):
    def make_profiler(self, backend, discrete, schedule):
        instances = []

        def create(**kwargs):
            instance = Mock()
            instances.append(instance)
            return instance

        factory = Mock(side_effect=create)
        factory.instances = instances
        namespace = dict(
            DistProfiler=object,
            functools=functools,
            torch=SimpleNamespace(profiler=SimpleNamespace(record_function=lambda name: nullcontext())),
            get_torch_profiler=factory,
            get_npu_profiler=factory,
            mark_start_range=Mock(),
            mark_end_range=Mock(),
        )
        filename, name = ("torch_profile", "Profiler") if backend == "torch" else ("mstx_profile", "NPUProfiler")
        cls = load_class(f"verl/utils/profiler/{filename}.py", name, namespace)
        tool = SimpleNamespace(discrete=discrete, schedule=schedule, contents=["cpu"], level="level0", analysis=False)
        config = SimpleNamespace(enable=True, save_path="unused")
        return cls, lambda: cls(rank=0, config=config, tool_config=tool), factory

    def test_nested_discrete_scope_steps_and_cleans_up_after_failure(self):
        for backend in ("torch", "npu"):
            for fail in (False, True):
                with self.subTest(backend=backend, fail=fail):
                    cls, build, factory = self.make_profiler(backend, True, schedule_config())
                    outer, inner = build(), build()

                    @inner.annotate(role="train_batch")
                    def mini_batch(inner=inner, fail=fail):
                        inner.step()
                        if fail:
                            raise RuntimeError("backward failed")
                        inner.step()

                    @outer.annotate(role="update")
                    def update(mini_batch=mini_batch):
                        mini_batch()
                        mini_batch()

                    if fail:
                        with self.assertRaisesRegex(RuntimeError, "backward failed"):
                            update()
                    else:
                        update()
                    factory.assert_called_once()
                    self.assertEqual(factory.call_args.kwargs["schedule"], schedule_config().to_torch_kwargs())
                    active = factory.instances[0]
                    self.assertEqual(active.step.call_count, 1 if fail else 4)
                    active.stop.assert_called_once()
                    self.assertIsNone(cls._active_prof)
                    # A new scope must be able to start after success or failure.
                    with self.assertRaises(RuntimeError) if fail else nullcontext():
                        update()
                    self.assertEqual(factory.call_count, 2)
                    self.assertIsNone(cls._active_prof)

    def test_disabled_schedule_no_active_profiler_and_restart(self):
        for backend in ("torch", "npu"):
            for schedule in (None, schedule_config(0), schedule_config()):
                with self.subTest(backend=backend, schedule=schedule):
                    cls, build, factory = self.make_profiler(backend, False, schedule)
                    outer, inner = build(), build()
                    inner.step()
                    factory.assert_not_called()
                    for _ in range(2):
                        outer.start()
                        active = cls._active_prof
                        for _ in range(7):
                            inner.step()
                        outer.stop()
                        self.assertEqual(active.step.call_count, 7 if schedule and schedule.enabled else 0)
                        active.stop.assert_called_once()
                        self.assertIsNone(cls._active_prof)
                        inner.step()
                    self.assertEqual(factory.call_count, 2)

    def test_dispatcher_rank_gate_allows_nested_worker_steps(self):
        cls = load_class("verl/utils/profiler/profile.py", "DistProfiler", {})
        for enable, rank_selected in ((False, True), (True, False), (True, True)):
            with self.subTest(enable=enable, rank_selected=rank_selected):
                dispatcher = object.__new__(cls)
                dispatcher._enable = enable
                dispatcher._this_rank = rank_selected
                dispatcher._this_step = False  # Only the outer worker was started.
                dispatcher._impl = Mock()
                dispatcher.step()
                self.assertEqual(dispatcher._impl.step.call_count, int(enable and rank_selected))

    def test_schedule_only_annotation_preserves_unscheduled_scopes(self):
        cls = load_class(
            "verl/utils/profiler/profile.py",
            "DistProfiler",
            dict(
                functools=functools,
                RLInsightLogger=SimpleNamespace(trace_state=lambda *args, **kwargs: nullcontext()),
            ),
        )
        for schedule in (None, schedule_config(0), schedule_config()):
            with self.subTest(schedule=schedule):
                dispatcher = object.__new__(cls)
                dispatcher.rank = 0
                dispatcher._enable = dispatcher._this_rank = dispatcher._this_step = True
                dispatcher.tool_config = SimpleNamespace(schedule=schedule)
                dispatcher._impl = Mock()
                dispatcher._impl.annotate.return_value = lambda fn: fn
                body = Mock(return_value="result")

                @cls.annotate(schedule_only=True)
                def wrapped(worker, body=body):
                    return body(worker)

                self.assertEqual(wrapped(SimpleNamespace(profiler=dispatcher)), "result")
                body.assert_called_once()
                self.assertEqual(dispatcher._impl.annotate.call_count, int(bool(schedule and schedule.enabled)))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is required for native CPU trace validation")
    def test_native_torch_recording_windows_and_reset(self):
        import torch

        for discrete, active in ((False, 1), (True, 1), (False, 0), (True, 0)):
            with self.subTest(discrete=discrete, active=active):
                windows = []

                def create(windows=windows, **kwargs):
                    schedule = kwargs["schedule"]
                    return torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU],
                        on_trace_ready=lambda prof: windows.append(
                            {event.name for event in prof.events() if event.name.startswith("micro_")}
                        ),
                        **({"schedule": torch.profiler.schedule(**schedule)} if schedule else {}),
                    )

                cls = load_class(
                    "verl/utils/profiler/torch_profile.py",
                    "Profiler",
                    dict(DistProfiler=object, functools=functools, torch=torch, get_torch_profiler=create),
                )
                tool = SimpleNamespace(discrete=discrete, schedule=schedule_config(active), contents=["cpu"])
                config = SimpleNamespace(enable=True, save_path="unused")
                outer = cls(rank=0, config=config, tool_config=tool)
                inner = cls(rank=0, config=config, tool_config=tool)

                @outer.annotate(role="update")
                def update(inner=inner):
                    for i in range(7):
                        with torch.profiler.record_function(f"micro_{i}"):
                            torch.ones(4).sum()
                        inner.step()

                for _ in range(2):
                    outer.start()
                    update()
                    outer.stop()
                expected = {"micro_4"} if active else {f"micro_{i}" for i in range(7)}
                self.assertEqual(windows, [expected, expected])


if __name__ == "__main__":
    unittest.main()
