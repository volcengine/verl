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
import logging
import os
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np
import ray
import torch
import transfer_queue as tq
from packaging.version import InvalidVersion, Version
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData
from torchdata.stateful_dataloader import StatefulDataLoader

from verl.trainer.ppo.utils import create_rl_dataset, create_rl_sampler
from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopManagerTQ
from verl.trainer.ppo.v1.replay_buffer import (
    ReplayBuffer,
    ReplayBufferAsync,
    _accumulate_eviction_metrics,
)
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.import_utils import load_class_from_fqn, load_extern_type
from verl.utils.skip import SkipManager

logger = logging.getLogger(__name__)


def _tq_supports_checkpoint() -> bool:
    try:
        version_supported = Version(getattr(tq, "__version__", "")) >= Version("0.1.9")
    except InvalidVersion:
        return False
    return (
        version_supported
        and callable(getattr(tq, "save_checkpoint", None))
        and callable(getattr(tq, "load_checkpoint", None))
    )


@dataclass
class _SampleRequest:
    global_steps: int
    batch_size: int
    partition_id: str
    min_sampleable: int | None
    consume: bool
    future: asyncio.Future


class StreamingDataloader:
    """Synchronous rollout data actor implementation."""

    _persist_prompt_fields = False

    def __init__(
        self,
        *,
        config,
        tokenizer,
        processor,
        llm_client,
        teacher_client=None,
        reward_loop_worker_handles=None,
        filter_groups_metric: str | None = None,
    ):
        tq.init()
        SkipManager.init(config)
        self.config = config
        self.trainer_mode = config.trainer.v1.trainer_mode
        self.global_steps = 0
        self.filter_groups_metric = filter_groups_metric

        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        configured_gen_batch_size = config.data.get("gen_batch_size", None)
        if self.trainer_mode == "sync":
            self.gen_batch_size = int(configured_gen_batch_size or config.data.train_batch_size)
        else:
            if configured_gen_batch_size not in (None, 1):
                raise ValueError("Async trainers require data.gen_batch_size to be 1 or null")
            self.gen_batch_size = 1
        self.train_dataloader = StatefulDataLoader(
            dataset=train_dataset,
            batch_size=self.gen_batch_size,
            num_workers=config.data["dataloader_num_workers"],
            drop_last=True,
            collate_fn=collate_fn,
            sampler=create_rl_sampler(config.data, train_dataset),
        )
        self.train_dataloader_it = None

        manager_class_fqn = config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        manager_cls = (
            load_class_from_fqn(manager_class_fqn, "AgentLoopManager") if manager_class_fqn else AgentLoopManagerTQ
        )
        self.agent_loop_manager = manager_cls.create(
            config=config,
            llm_client=llm_client,
            teacher_client=teacher_client,
            reward_loop_worker_handles=reward_loop_worker_handles,
        )
        self.replay_buffer = self._build_replay_buffer()

    def _build_replay_buffer(self) -> ReplayBuffer:
        sampler_config = self.config.trainer.v1.sampler
        custom_sampler = sampler_config.get("custom_sampler", None)
        if custom_sampler is not None and custom_sampler.get("path") and custom_sampler.get("name"):
            sampler_cls = load_extern_type(custom_sampler.path, custom_sampler.name)
            return sampler_cls(
                trainer_mode=self.trainer_mode,
                trainer_config=self.config.trainer.v1.get(self.trainer_mode, {}),
                max_off_policy_threshold=sampler_config.max_off_policy_threshold,
                max_off_policy_strategy=sampler_config.max_off_policy_strategy,
                sampler_kwargs=sampler_config.sampler_kwargs,
                refill_fn=self._refill_exact,
            )
        filter_groups = self.config.algorithm.get("filter_groups", None)
        max_inflight_gen_batches = (
            filter_groups.get("max_inflight_gen_batches", 1) if self.filter_groups_metric is not None else 1
        )
        sync_refill_failed_groups = bool(sampler_config.get("sync_refill_failed_groups", False))
        return ReplayBuffer(
            trainer_mode=self.trainer_mode,
            trainer_config=self.config.trainer.v1.get(self.trainer_mode, {}),
            max_off_policy_threshold=sampler_config.max_off_policy_threshold,
            max_off_policy_strategy=sampler_config.max_off_policy_strategy,
            sampler_kwargs=sampler_config.sampler_kwargs,
            refill_fn=self._refill_exact,
            filter_groups_metric=self.filter_groups_metric,
            train_batch_size=self.config.data.train_batch_size,
            gen_batch_size=1
            if self.filter_groups_metric is not None or sync_refill_failed_groups
            else self.gen_batch_size,
            max_inflight_gen_batches=max_inflight_gen_batches,
            sync_refill_failed_groups=sync_refill_failed_groups,
        )

    def sample(
        self,
        global_steps: int,
        batch_size: int,
        partition_id: str = "train",
        prompts: TensorDict | None = None,
        min_sampleable: int | None = None,
        consume: bool = True,
    ) -> tuple[Any, dict, int]:
        """Run synchronous rollout and materialize one requested batch."""
        if not consume or min_sampleable is not None:
            raise ValueError("count/wait sample operations are only supported by async trainers")
        self.global_steps = global_steps
        SkipManager.set_step(global_steps)
        if prompts is not None:
            self._submit_batch_to_rollout(prompts, partition_id)
        elif partition_id == "train":
            self._refill_exact(batch_size)
        batch, metrics = self.replay_buffer.sample(global_steps, partition_id, batch_size)
        return batch, metrics, len(batch)

    def save_checkpoint(self, local_global_step_folder: str, global_steps: int) -> None:
        """Persist the synchronous dataloader cursor."""
        del global_steps
        os.makedirs(local_global_step_folder, exist_ok=True)
        torch.save(self.train_dataloader.state_dict(), os.path.join(local_global_step_folder, "data.pt"))

    def load_checkpoint(self, local_global_step_folder: str | None, global_steps: int) -> dict:
        """Restore the synchronous dataloader cursor."""
        self.global_steps = global_steps
        SkipManager.set_step(global_steps)
        if local_global_step_folder is not None:
            dataloader_path = os.path.join(local_global_step_folder, "data.pt")
            if os.path.exists(dataloader_path):
                self.train_dataloader.load_state_dict(torch.load(dataloader_path, weights_only=False))
            else:
                logger.warning("No dataloader checkpoint found at %s", dataloader_path)
        self.train_dataloader_it = None
        return {}

    def _fetch_one_gen_batch(self) -> TensorDict:
        try:
            if self.train_dataloader_it is None:
                self.train_dataloader_it = iter(self.train_dataloader)
            batch_dict = next(self.train_dataloader_it)
        except StopIteration:
            self.train_dataloader_it = iter(self.train_dataloader)
            batch_dict = next(self.train_dataloader_it)
        batch_dict["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch_dict["raw_prompt"]))], dtype=object)
        return tu.get_tensordict(batch_dict)

    def _next_train_batch(self, num_prompts: int) -> TensorDict:
        if num_prompts <= 0 or num_prompts % self.gen_batch_size != 0:
            raise ValueError(
                f"num_prompts ({num_prompts}) must be a positive multiple of gen_batch_size ({self.gen_batch_size})"
            )
        chunks = [self._fetch_one_gen_batch() for _ in range(num_prompts // self.gen_batch_size)]
        batch = chunks[0] if len(chunks) == 1 else tu.concat_tensordict(chunks)
        tu.assign_non_tensor_data(batch, "global_steps", self.global_steps)
        return batch

    def _default_rollout_n(self, partition_id: str) -> int:
        rollout_config = self.config.actor_rollout_ref.rollout
        return int(rollout_config.n if partition_id == "train" else rollout_config.val_kwargs.n)

    def _rollout_n_values(self, batch: TensorDict, partition_id: str) -> list[int]:
        if "__rollout_n__" not in batch:
            return [self._default_rollout_n(partition_id)] * len(batch)

        values = batch["__rollout_n__"]
        if isinstance(values, NonTensorData):
            values = [values.data] * len(batch)
        rollout_ns = []
        for i in range(len(batch)):
            value = values[i]
            if isinstance(value, NonTensorData):
                value = value.data
            if isinstance(value, torch.Tensor):
                value = value.item()
            rollout_n = int(value)
            if rollout_n <= 0:
                raise ValueError(f"rollout_n must be positive, got {rollout_n}")
            rollout_ns.append(rollout_n)
        return rollout_ns

    def _submit_batch_to_rollout(self, batch: TensorDict, partition_id: str) -> int:
        tags = [{"is_prompt": True, "status": "pending", "global_steps": self.global_steps} for _ in range(len(batch))]
        if partition_id == "train" and self._persist_prompt_fields:
            tq.kv_batch_put(
                keys=list(batch["uid"]),
                partition_id=partition_id,
                tags=tags,
                fields=batch.select(*[key for key in batch.keys() if not isinstance(batch.get(key), NonTensorData)]),
            )
        else:
            tq.kv_batch_put(keys=list(batch["uid"]), partition_id=partition_id, tags=tags)
        self.agent_loop_manager.generate_sequences(batch)
        return len(batch)

    def _refill_exact(self, num_prompts: int) -> int:
        if num_prompts <= 0:
            return 0
        return self._submit_batch_to_rollout(self._next_train_batch(num_prompts), "train")


class StreamingDataloaderAsync(StreamingDataloader):
    """Autonomous async rollout data actor with serialized state transitions."""

    # Restore reissues in-flight prompts from TransferQueue, so their fields must be stored there.
    _persist_prompt_fields = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if int(self.config.trainer.get("save_freq", 0) or 0) > 0 and not _tq_supports_checkpoint():
            logger.warning(
                "TransferQueue checkpoint APIs are unavailable; async checkpoints will preserve the "
                "dataloader cursor but not in-flight rollout state"
            )
        watermark_batches = float(self.config.trainer.v1.get(self.trainer_mode, {}).get("watermark_batches", 1))
        if watermark_batches <= 0:
            raise ValueError(f"watermark_batches must be positive, got {watermark_batches}")
        self.default_rollout_n = self._default_rollout_n("train")
        self.watermark_sessions = round(watermark_batches * self.config.data.train_batch_size * self.default_rollout_n)
        self._pending_sample: _SampleRequest | None = None
        # Wake local requests immediately. If sample batch is not ready at once, wait for next poll.
        self._wakeup = asyncio.Event()
        self._coordinator_task: asyncio.Task | None = None
        # Validation may start the coordinator before load_checkpoint establishes train state.
        self._train_state_ready = False
        self._validation_active = False
        self._pending_metrics: dict = {}

    def _build_replay_buffer(self) -> ReplayBufferAsync:
        sampler_config = self.config.trainer.v1.sampler
        custom_sampler = sampler_config.get("custom_sampler", None)
        sampler_cls = ReplayBufferAsync
        replay_buffer_kwargs = dict(
            trainer_mode=self.trainer_mode,
            trainer_config=self.config.trainer.v1.get(self.trainer_mode, {}),
            max_off_policy_threshold=sampler_config.max_off_policy_threshold,
            max_off_policy_strategy=sampler_config.max_off_policy_strategy,
            sampler_kwargs=sampler_config.sampler_kwargs,
        )
        if custom_sampler is not None and custom_sampler.get("path") and custom_sampler.get("name"):
            sampler_cls = load_extern_type(custom_sampler.path, custom_sampler.name)
        else:
            replay_buffer_kwargs["filter_groups_metric"] = self.filter_groups_metric
        replay_buffer = sampler_cls(**replay_buffer_kwargs)
        required_methods = ("poll_once", "materialize_sampleable", "unfinished_session_count")
        if any(not hasattr(replay_buffer, method) for method in required_methods):
            raise TypeError(
                f"{type(replay_buffer).__name__} must implement {required_methods} for async coordinator sampling"
            )
        return replay_buffer

    def _submit_batch_to_rollout(self, batch: TensorDict, partition_id: str) -> int:
        # TODO: support custom `__rollout_n__`
        if partition_id == "train":
            rollout_ns = self._rollout_n_values(batch, partition_id)
            if any(rollout_n != self.default_rollout_n for rollout_n in rollout_ns):
                raise ValueError("Async trainers require a fixed train rollout_n")
        return super()._submit_batch_to_rollout(batch, partition_id)

    async def sample(
        self,
        global_steps: int,
        batch_size: int,
        partition_id: str = "train",
        prompts: TensorDict | None = None,
        min_sampleable: int | None = None,
        consume: bool = True,
    ) -> tuple[Any, dict, int]:
        self._ensure_coordinator()
        future = asyncio.get_running_loop().create_future()
        if self._pending_sample is not None:
            raise RuntimeError("Concurrent sample calls are not supported")
        self._validation_active = partition_id == "val"
        self.global_steps = global_steps
        SkipManager.set_step(global_steps)
        if prompts is not None:
            # only for validation
            self._submit_batch_to_rollout(prompts, partition_id)
        self._pending_sample = _SampleRequest(
            global_steps=global_steps,
            batch_size=batch_size,
            partition_id=partition_id,
            min_sampleable=min_sampleable,
            consume=consume,
            future=future,
        )
        self._wakeup.set()
        # wait for train batch ready
        return await future

    def save_checkpoint(self, local_global_step_folder: str, global_steps: int) -> None:
        super().save_checkpoint(local_global_step_folder, global_steps)
        if _tq_supports_checkpoint():
            tq.save_checkpoint(
                os.path.join(local_global_step_folder, "transfer_queue"),
                metadata={"global_steps": global_steps},
            )
        else:
            logger.warning(
                "Skipping TransferQueue checkpoint because its checkpoint APIs are unavailable; "
                "in-flight rollout state will not be recoverable"
            )

    def load_checkpoint(self, local_global_step_folder: str | None, global_steps: int) -> dict:
        self.global_steps = global_steps
        SkipManager.set_step(global_steps)
        if local_global_step_folder is not None:
            tq_path = os.path.join(local_global_step_folder, "transfer_queue")
            dataloader_path = os.path.join(local_global_step_folder, "data.pt")
            if not os.path.exists(dataloader_path):
                raise RuntimeError(f"StreamingDataloader restore requires {dataloader_path}")
            self.train_dataloader.load_state_dict(torch.load(dataloader_path, weights_only=False))
            if _tq_supports_checkpoint() and os.path.exists(tq_path):
                tq.load_checkpoint(tq_path)
            else:
                logger.warning(
                    "Skipping TransferQueue restore because its checkpoint APIs or checkpoint data are unavailable; "
                    "in-flight rollout state will not be recovered"
                )
        self.train_dataloader_it = None
        reissued = self._reissue_inflight_prompts()
        self._train_state_ready = True
        self._validation_active = False
        self._ensure_coordinator()
        self._wakeup.set()
        return {"streaming_dataloader/reissued_groups": float(reissued)}

    def _ensure_coordinator(self) -> None:
        if self._coordinator_task is None:
            self._coordinator_task = asyncio.create_task(
                self._coordinator_loop(),
                name="streaming-dataloader-coordinator",
            )
            self._coordinator_task.add_done_callback(self._on_coordinator_done)

    def _on_coordinator_done(self, task: asyncio.Task) -> None:
        if task.cancelled():
            return
        error = task.exception() or RuntimeError("Streaming dataloader coordinator exited unexpectedly")
        request, self._pending_sample = self._pending_sample, None
        if request is not None and not request.future.done():
            request.future.set_exception(error)
        logger.critical(
            "Streaming dataloader coordinator failed; terminating the actor",
            exc_info=(type(error), error, error.__traceback__),
        )
        # The coordinator exclusively advances replay state, so the actor cannot recover without it.
        ray.actor.exit_actor()

    async def _coordinator_loop(self) -> None:
        while True:
            # Clear before doing work so a concurrent producer cannot have its wakeup erased.
            self._wakeup.clear()
            if self._advance_once():
                # poll_once already refreshed replay metadata for this coordinator pass.
                self._refill_to_watermark()
            try:
                await asyncio.wait_for(self._wakeup.wait(), timeout=self.replay_buffer.poll_interval)
            except TimeoutError:
                pass

    def _advance_once(self) -> bool:
        request = self._pending_sample
        if request is not None:
            self._poll_once(request)
            return self._train_state_ready and request.partition_id == "train"
        if self._train_state_ready and not self._validation_active:
            self._poll_once()
            return True
        return False

    def _poll_once(self, request: _SampleRequest | None = None) -> None:
        if request is None:
            global_steps = self.global_steps
            partition_id = "train"
            target_count = 0
        else:
            global_steps = request.global_steps
            partition_id = request.partition_id
            target_count = (
                max(request.batch_size, request.min_sampleable or 0)
                if request.consume
                else (request.min_sampleable or 0)
            )

        sampleable, metrics, ready, evicted_count = self.replay_buffer.poll_once(
            global_steps,
            partition_id,
            target_count,
        )
        if partition_id == "train":
            _accumulate_eviction_metrics(self._pending_metrics, metrics)
            if evicted_count:
                return
        if request is None:
            return
        if not request.consume and request.min_sampleable is None:
            ready = True
        if not ready:
            return

        try:
            if request.consume:
                batch = self.replay_buffer.materialize_sampleable(
                    request.global_steps,
                    request.partition_id,
                    sampleable,
                    request.batch_size,
                )
                result_count = len(batch)
            else:
                batch = None
                result_count = len(sampleable)
            if request.partition_id == "train":
                result_metrics = dict(self._pending_metrics)
                self._pending_metrics.clear()
            else:
                result_metrics = metrics
            result = (batch, result_metrics, result_count)
        except BaseException as error:
            self._pending_sample = None
            if not request.future.done():
                request.future.set_exception(error)
            return

        self._pending_sample = None
        if not request.future.done():
            request.future.set_result(result)

    def _refill_to_watermark(self) -> int:
        inflight_sessions = self.replay_buffer.unfinished_session_count("train", self.default_rollout_n)
        free_slots = max(0, self.watermark_sessions - inflight_sessions)
        if free_slots <= 0:
            return 0

        num_prompts = free_slots // self.default_rollout_n
        return self._refill_exact(num_prompts) if num_prompts else 0

    def _reissue_inflight_prompts(self, partition_id: str = "train") -> int:
        data = tq.kv_list(partition_id)
        if not data:
            return 0
        items = data.get(partition_id, {})
        inflight_tags = {
            key: tag
            for key, tag in items.items()
            if tag.get("is_prompt", False) and tag.get("status") in ("pending", "running")
        }
        inflight_uids = list(inflight_tags)
        if not inflight_uids:
            return 0
        batch = tq.kv_batch_get(keys=inflight_uids, partition_id=partition_id)
        if "__rollout_n__" in batch:
            batch.pop("__rollout_n__")
        inflight_uid_set = set(inflight_uids)
        old_trajectory_keys = [
            key
            for key, tag in items.items()
            if not tag.get("is_prompt", False) and key.split("_", 1)[0] in inflight_uid_set
        ]
        if old_trajectory_keys:
            tq.kv_clear(keys=old_trajectory_keys, partition_id=partition_id)
        tu.assign_non_tensor_data(batch, "global_steps", self.global_steps)
        tags = [{"is_prompt": True, "status": "pending", "global_steps": self.global_steps} for _ in inflight_uids]
        tq.kv_batch_put(keys=inflight_uids, partition_id=partition_id, tags=tags)
        self.agent_loop_manager.generate_sequences(batch)
        return len(inflight_uids)


StreamingDataloaderActor = ray.remote(StreamingDataloader)
StreamingDataloaderAsyncActor = ray.remote(max_concurrency=1)(StreamingDataloaderAsync)
