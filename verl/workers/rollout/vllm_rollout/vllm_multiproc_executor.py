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

import logging
import os
import pickle
import signal
import threading
import weakref
from multiprocessing.synchronize import Lock as LockType
from threading import Thread
from types import MethodType

import torch
import zmq
from vllm import envs
from vllm.config import VllmConfig
from vllm.distributed.device_communicators.shm_broadcast import Handle, MessageQueue
from vllm.utils import get_mp_context
from vllm.utils import (
    get_distributed_init_method,
    get_loopback_ip,
    get_open_port,
)
from vllm.v1.executor.abstract import FailureCallback
from vllm.v1.executor.multiproc_executor import UnreadyWorkerProcHandle, WorkerProc, MultiprocExecutor, set_multiprocessing_worker_envs

try:
    from vllm.worker.worker_base import WorkerWrapperBase
except ModuleNotFoundError:
    # https://github.com/vllm-project/vllm/commit/6a113d9aed8221a9c234535958e70e34ab6cac5b
    from vllm.v1.worker.worker_base import WorkerWrapperBase


from verl.utils.vllm.vllm_fp8_utils import apply_vllm_fp8_patches

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

# https://github.com/vllm-project/vllm/issues/13175
def _monkey_patch_compute_logits(model, vocab_size: int):
    original_compute_logits = model.compute_logits

    def compute_logits(
        self,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        logits = original_compute_logits(*args, **kwargs)
        logits[..., vocab_size:] = float("-inf")
        return logits

    model.compute_logits = MethodType(compute_logits, model)

class vLLMWorkerProc(WorkerProc):
    READY_STR = "READY"

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        input_shm_handle: Handle,
        shared_worker_lock: LockType,
    ):
        self.rank = rank
        all_kwargs: list[dict] = [
            {} for _ in range(vllm_config.parallel_config.world_size)
        ]
        all_kwargs[rank] = {
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
            "is_driver_worker": True,
        }
        if os.environ.get("VERL_VLLM_FP8_QUANT_ENABLED", "0") == "1":
            apply_vllm_fp8_patches()
        rank_offset = int(os.environ.get("VERL_VLLM_MULTIPROC_RANK_OFFSET", "0"))
        wrapper = WorkerWrapperBase(vllm_config=vllm_config, rpc_rank=local_rank - rank_offset)
        wrapper.init_worker(all_kwargs)
        self.worker = wrapper

        self.rpc_broadcast_mq = MessageQueue.create_from_handle(
            input_shm_handle, self.worker.rank)
        self.worker_response_mq = MessageQueue(1, 1)

        self.mm_receiver_cache = None

        self.worker.init_device()

        self.setup_proc_title_and_log_prefix(
            enable_ep=vllm_config.parallel_config.enable_expert_parallel)

        # Load model
        self.worker.load_model()

        vocab_size = int(os.environ.get("VERL_VLLM_VOCAB_SIZE", "0"))
        assert vocab_size != 0, "VERL_VLLM_VOCAB_SIZE must be set for patching!"
        _monkey_patch_compute_logits(self.worker.worker.model_runner.model, vocab_size)

    # modify from https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/v1/executor/multiproc_executor.py
    # NOTE: Only modify the WorkerProc to vLLMWorkerProc. It is best practice
    # to modify this method to the class method of the parent class WorkerProc,
    # and a PR will be submitted to the vllm community.
    @staticmethod
    def make_worker_process(
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        input_shm_handle,  # Receive SchedulerOutput
        shared_worker_lock: LockType,
    ) -> UnreadyWorkerProcHandle:
        context = get_mp_context()
        # (reader, writer)
        reader, writer = context.Pipe(duplex=False)

        # Create death pipe to detect parent process exit
        death_reader, death_writer = context.Pipe(duplex=False)

        process_kwargs = {
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
            "input_shm_handle": input_shm_handle,
            "ready_pipe": (reader, writer),
            "death_pipe": death_reader,
            "shared_worker_lock": shared_worker_lock,
        }
        # Run EngineCore busy loop in background process.
        proc = context.Process(target=vLLMWorkerProc.worker_main,
                               kwargs=process_kwargs,
                               name=f"VllmWorker-{rank}",
                               daemon=True)

        proc.start()
        writer.close()
        # Keep death_writer open in parent - when parent exits,
        # death_reader in child will get EOFError
        return UnreadyWorkerProcHandle(proc, rank, reader, death_writer)
    
    # modify from https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/v1/executor/multiproc_executor.py
    # NOTE: Only modify the WorkerProc to vLLMWorkerProc. It is best practice
    # to modify this method to the class method of the parent class WorkerProc,
    # and a PR will be submitted to the vllm community.
    @staticmethod
    def worker_main(*args, **kwargs):
        """ Worker initialization and execution loops.
        This runs a background process """

        # Signal handler used for graceful termination.
        # SystemExit exception is only raised once to allow this and worker
        # processes to terminate without error
        shutdown_requested = False

        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                raise SystemExit()

        # Either SIGTERM or SIGINT will terminate the worker
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        worker = None
        # tuple[Connection, Connection]
        reader, ready_writer = kwargs.pop("ready_pipe")
        death_pipe = kwargs.pop("death_pipe", None)
        shutdown_event = threading.Event()
        # Start death monitoring thread if death_pipe is provided
        if death_pipe is not None:

            def monitor_parent_death():
                try:
                    # This will block until parent process exits (pipe closes)
                    death_pipe.recv()
                except EOFError:
                    # Parent process has exited, terminate this worker
                    logger.info("Parent process exited, terminating worker")
                    # Send signal to self to trigger clean shutdown
                    shutdown_event.set()
                except Exception as e:
                    logger.warning("Death monitoring error: %s", e)

            death_monitor = Thread(target=monitor_parent_death,
                                   daemon=True,
                                   name="WorkerDeathMonitor")
            death_monitor.start()

        try:
            reader.close()
            worker = vLLMWorkerProc(*args, **kwargs)

            # Send READY once we know everything is loaded
            ready_writer.send({
                "status":
                vLLMWorkerProc.READY_STR,
                "handle":
                worker.worker_response_mq.export_handle(),
            })

            # Ensure message queues are ready. Will deadlock if re-ordered.
            # Must be kept consistent with the Executor
            worker.rpc_broadcast_mq.wait_until_ready()
            worker.worker_response_mq.wait_until_ready()
            ready_writer.close()
            ready_writer = None

            worker.worker_busy_loop(cancel=shutdown_event)

        except Exception:
            # NOTE: if an Exception arises in busy_loop, we send
            # a FAILURE message over the MQ RPC to notify the Executor,
            # which triggers system shutdown.
            # TODO(rob): handle case where the MQ itself breaks.

            if ready_writer is not None:
                logger.exception("vLLMWorkerProc failed to start.")
            elif shutdown_event.is_set():
                logger.info("vLLMWorkerProc shutting down.")
            else:
                logger.exception("vLLMWorkerProc failed.")

            # The parent sends a SIGTERM to all worker processes if
            # any worker dies. Set this value so we don't re-throw
            # SystemExit() to avoid zmq exceptions in __del__.
            shutdown_requested = True

        finally:
            if ready_writer is not None:
                ready_writer.close()
            if death_pipe is not None:
                death_pipe.close()
            # Clean up once worker exits busy loop
            if worker is not None:
                worker.shutdown()


class vLLMMultiprocExecutor(MultiprocExecutor):
    """
    vLLM multiproc executor specialized for verl.
    Inherits from MultiprocExecutor and adds ZMQ server functionality
    for communicating with training workers.
    """

    uses_ray: bool = False

    def __init__(self, vllm_config):
        self.zmq_context = None
        self.zmq_socket = None
        self.zmq_thread = None
        self.zmq_is_running = False
        self.executor_zmq_address = None
        super().__init__(vllm_config)

    def _init_executor(self) -> None:
        # Call self.shutdown at exit to clean up
        # and ensure workers will be terminated.
        self._finalizer = weakref.finalize(self, self.shutdown)
        self.is_failed = False
        self.shutdown_event = threading.Event()
        self.failure_callback: FailureCallback | None = None

        self.world_size = self.parallel_config.world_size
        tensor_parallel_size = self.parallel_config.tensor_parallel_size
        rank_offset = int(os.environ.get("VERL_VLLM_MULTIPROC_RANK_OFFSET", "0"))

        # Set multiprocessing envs
        set_multiprocessing_worker_envs()

        distributed_init_method = get_distributed_init_method(
            get_loopback_ip(), get_open_port()
        )
        max_chunk_bytes = envs.VLLM_MQ_MAX_CHUNK_BYTES_MB * 1024 * 1024
        self.rpc_broadcast_mq = MessageQueue(tensor_parallel_size,
                                             tensor_parallel_size,
                                             max_chunk_bytes=max_chunk_bytes)
        scheduler_output_handle = self.rpc_broadcast_mq.export_handle()

        # Create workers
        context = get_mp_context()
        shared_worker_lock = context.Lock()
        unready_workers: list[UnreadyWorkerProcHandle] = []
        success = False
        try:
            for rank in range(tensor_parallel_size):
                unready_workers.append(
                    vLLMWorkerProc.make_worker_process(
                        vllm_config=self.vllm_config,
                        local_rank=rank + rank_offset,
                        rank=rank,
                        distributed_init_method=distributed_init_method,
                        input_shm_handle=scheduler_output_handle,
                        shared_worker_lock=shared_worker_lock,
                    ))
            self.workers = vLLMWorkerProc.wait_for_ready(unready_workers)

            # Ensure message queues are ready. Will deadlock if re-ordered
            # Must be kept consistent with the WorkerProc.
            self.rpc_broadcast_mq.wait_until_ready()
            for w in self.workers:
                w.worker_response_mq.wait_until_ready()

            self.start_worker_monitor()
            success = True
        finally:
            if not success:
                # Clean up the worker procs if there was a failure.
                # Close death_writers first to signal workers to exit
                for uw in unready_workers:
                    if uw.death_writer is not None:
                        uw.death_writer.close()
                self._ensure_worker_termination(
                    [uw.proc for uw in unready_workers])

        self.output_rank = self._get_output_rank()
        self.has_connector = self.vllm_config.kv_transfer_config is not None

        # Initialize ZMQ server to communicate with training workers
        self._init_zmq_server()
        self._start_zmq_server()


    def _handle_zmq_request(self, request: dict) -> dict:
        """Handle ZMQ requests from vLLMAsyncRollout."""
        method = request.get("method")
        args = request.get("args", ())
        kwargs = request.get("kwargs", {})

        try:
            # Use collective_rpc to distribute the method call to workers
            results = self.collective_rpc(method, *args, **kwargs)
            return {"status": "success", "result": results}

        except Exception as e:
            logger.error(f"Error handling ZMQ request {method}: {e}")
            return {"status": "error", "error": str(e)}

    def _init_zmq_server(self):
        """Initialize ZMQ server."""
        # Read ZMQ address from environment variable (set by vLLMHttpServerBase)
        self.executor_zmq_address = os.environ.get("VERL_VLLM_EXECUTOR_ZMQ_ADDRESS")
        if not self.executor_zmq_address:
            raise ValueError("VERL_VLLM_EXECUTOR_ZMQ_ADDRESS environment variable not set")

        logger.info(f"vLLMMultiprocExecutor ZMQ server will bind to: {self.executor_zmq_address}")

    def _start_zmq_server(self):
        """Start ZMQ server."""
        import threading

        if self.zmq_is_running:
            return

        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.REP)

        # Bind to address
        self.zmq_socket.bind(self.executor_zmq_address)
        logger.info(f"vLLMMultiprocExecutor ZMQ server bound to: {self.executor_zmq_address}")

        # Start service thread
        self.zmq_is_running = True
        self.zmq_thread = threading.Thread(
            target=self._zmq_service_loop,
            daemon=True,
            name="vLLMMultiprocExecutor-ZMQ"
        )
        self.zmq_thread.start()
        logger.info("vLLMMultiprocExecutor ZMQ server started")

    def _zmq_service_loop(self):
        """ZMQ service loop to handle requests from vLLMAsyncRollout."""
        while self.zmq_is_running:
            try:
                message = self.zmq_socket.recv()
                request = pickle.loads(message)

                response = self._handle_zmq_request(request)

                self.zmq_socket.send(pickle.dumps(response))

            except zmq.Again:
                continue
            except zmq.ZMQError as e:
                if e.errno == zmq.ETERM:
                    logger.info("vLLMMultiprocExecutor ZMQ context terminated")
                    break
                logger.error(f"ZMQ error in vLLMMultiprocExecutor: {e}")
            except Exception as e:
                logger.exception(f"Error in ZMQ service loop: {e}")

    def _shutdown_zmq_server(self):
        """Shutdown ZMQ server."""
        if self.zmq_is_running:
            self.zmq_is_running = False

            if self.zmq_socket:
                self.zmq_socket.close()
                self.zmq_socket = None

            if self.zmq_context:
                self.zmq_context.term()
                self.zmq_context = None

            if self.zmq_thread and self.zmq_thread.is_alive():
                self.zmq_thread.join(timeout=5)
                if self.zmq_thread.is_alive():
                    logger.warning("ZMQ thread did not terminate cleanly")

            logger.info("ZMQ server shutdown complete")

    def shutdown(self):
        """Override shutdown to ensure ZMQ server is properly closed."""
        # First shutdown ZMQ server
        self._shutdown_zmq_server()

        # Call parent class shutdown
        super().shutdown()

    def check_health(self):
        """Check executor health."""
        return