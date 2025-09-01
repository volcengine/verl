# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import time
import uuid
import collections
import aiohttp
import traceback
from typing import List, Dict, Any, Callable, Awaitable

# Define the expected signature for the batch submission function
# It should be an async callable that takes:
# 1. A list of original request payloads (List[Any])
# 2. The aiohttp.ClientSession instance
# It should return:
# 1. A list of results (List[Any]) where the order *strictly* matches the input payloads order.
BatchSubmitFunc = Callable[[List[Any], aiohttp.ClientSession], Awaitable[List[Any]]]


class RequestProcessor:
    """
    Manages batch submission concurrently using an injected batch submission function.
    Requests are buffered and processed by concurrent sender workers.
    """
    def __init__(self, batch_size: int, batch_timeout_seconds: float, session: aiohttp.ClientSession, concurrency: int, batch_submit_func: BatchSubmitFunc):
        """
        Initializes the Request Processor with concurrent sending and a generic submission function.
        Must be called within an event loop context.

        Args:
            batch_size: Maximum items per batch.
            batch_timeout_seconds: Timeout for gathering items into a batch.
            session: The aiohttp.ClientSession to pass to the submission function.
            language: Submission language to pass to the submission function.
            concurrency: Maximum number of concurrent batches being sent to B.
            batch_submit_func: The async function to call for sending a batch.
                               Must match the BatchSubmitFunc signature.
        """
        if batch_size <= 0 or concurrency <= 0:
             raise ValueError("batch_size and concurrency must be positive")
        if batch_timeout_seconds <= 0:
             print("Warning: batch_timeout_seconds <= 0, batching will be strictly based on batch_size or queue availability.")

        self._batch_size = batch_size
        self._batch_timeout_seconds = batch_timeout_seconds
        self._session = session
        self._concurrency = concurrency
        self._batch_submit_func = batch_submit_func # Store the injected function

        self._submission_queue = asyncio.Queue()
        self._pending_requests: Dict[str, Dict[str, Any]] = {} # {request_id: {"future": Future, "payload": payload}}

        self._semaphore = asyncio.Semaphore(concurrency)

        self._sender_workers: List[asyncio.Task] = []
        self._running = False

        # --- Statistics ---
        self._reset_stats_internal() # Initialize stats
        # --- End Statistics ---

        print(f"[{time.monotonic():.4f}] RequestProcessor initialized with concurrency={self._concurrency} (async thread).")

    def _reset_stats_internal(self):
        """Helper to initialize or reset statistics."""
        self._stats = {
            "total_batch_submission_duration_seconds": 0.0,
            "num_batches_submitted": 0,
            "num_successful_batches": 0,
            "num_failed_batches": 0,
            "total_items_processed_in_batches": 0,
            "actual_batch_sizes": [] # Stores the size of each batch
        }

    async def send_request(self, request_payload: Any, timeout: float = None):
        """
        Adds a single request to the buffer and waits for its result.
        This call is awaitable and provides the synchronous-like pattern.
        """
        if not self._running:
             raise RuntimeError("RequestProcessor is not running. Call .start() first.")

        request_id = str(uuid.uuid4())

        future = asyncio.get_running_loop().create_future()
        self._pending_requests[request_id] = {
            "future": future,
            "payload": request_payload
        }

        await self._submission_queue.put(request_id)

        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            if request_id in self._pending_requests:
                 del self._pending_requests[request_id]
            print(f"[{time.monotonic():.4f}] Request {request_id[:6]}... timed out waiting for result.")
            raise
        except Exception as e:
            if request_id in self._pending_requests:
                 del self._pending_requests[request_id]
            print(f"[{time.monotonic():.4f}] Request {request_id[:6]}... encountered error while waiting: {e}")
            raise

    async def send_requests(self, request_payloads: List[Any], timeout: float = None) -> List[Any]:
        """
        Submits multiple request payloads concurrently and waits for all their results.
        Returns results or exceptions in the same order as input payloads.
        Uses send_request internally and gathers futures.
        """
        if not self._running:
             raise RuntimeError("RequestProcessor is not running. Call .start() first.")

        if not request_payloads:
            return []

        # print(f"[{time.monotonic():.4f}] RequestProcessor.send_requests submitting {len(request_payloads)} individual requests concurrently...")

        tasks = []
        for payload in request_payloads:
             tasks.append(asyncio.create_task(self.send_request(payload, timeout=timeout)))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # print(f"[{time.monotonic():.4f}] RequestProcessor.send_requests received all results.")

        return results

    async def start(self):
        """
        Starts the concurrent sender worker tasks. Must be called within loop.
        """
        if self._running:
            print(f"[{time.monotonic():.4f}] RequestProcessor is already running.")
            return
        self._running = True
        self._sender_workers = [asyncio.create_task(self._sender_worker()) for _ in range(self._concurrency)]
        print(f"[{time.monotonic():.4f}] RequestProcessor started {self._concurrency} sender workers.")

    async def stop(self):
        if not self._running:
            print(f"[{time.monotonic():.4f}] RequestProcessor is not running.")
            return

        print(f"[{time.monotonic():.4f}] Stopping RequestProcessor. Signaling workers...")
        self._running = False

        await self._submission_queue.join()
        print(f"[{time.monotonic():.4f}] Submission queue joined. All buffered items processed by sender workers.")

        # Wait for sender workers to finish their current batch and exit their loops
        for worker in self._sender_workers:
            worker.cancel()
        try:
            await asyncio.gather(*self._sender_workers, return_exceptions=True)
        except asyncio.CancelledError:
            print(f"[{time.monotonic():.4f}] Sender workers cancelled as expected.")
        except Exception as e:
            print(f"[{time.monotonic():.4f}] Error during sender workers shutdown: {e}")

        print(f"[{time.monotonic():.4f}] All sender workers stopped.")

        print(f"[{time.monotonic():.4f}] Waiting for {len(self._pending_requests)} pending results...")
        wait_tasks = [asyncio.create_task(req_info["future"])
                      for req_id, req_info in list(self._pending_requests.items())
                      if not req_info["future"].done()]

        if wait_tasks:
             stop_results_timeout = self._batch_timeout_seconds * 5
             print(f"[{time.monotonic():.4f}] Waiting for remaining results with timeout {stop_results_timeout:.2f}s...")
             try:
                await asyncio.wait_for(asyncio.gather(*wait_tasks, return_exceptions=True), timeout=stop_results_timeout)
                print(f"[{time.monotonic():.4f}] All pending results awaited or timed out during stop.")
             except asyncio.TimeoutError:
                print(f"[{time.monotonic():.4f}] Warning: Timeout waiting for all pending results during stop.")

        else:
             print(f"[{time.monotonic():.4f}] No pending results to await.")

        if self._pending_requests:
            print(f"[{time.monotonic():.4f}] Warning: Stopping with {len(self._pending_requests)} requests still pending (futures not completed/timed out)!")

        print(f"[{time.monotonic():.4f}] RequestProcessor stopped.")

    async def _sender_worker(self):
        """
        A single worker coroutine that continuously tries to send batches.
        Controls its own access to concurrent sending via the semaphore.
        Runs within the async thread's event loop.
        """
        print(f"[{time.monotonic():.4f}] Sender worker started.")
        batch_gathering_timeout = self._batch_timeout_seconds

        try:
            while self._running or not self._submission_queue.empty():
                 batch_item_ids = []

                 try:
                     first_item_id = await asyncio.wait_for(self._submission_queue.get(), timeout=batch_gathering_timeout)
                     self._submission_queue.task_done()
                     batch_item_ids.append(first_item_id)

                     while len(batch_item_ids) < self._batch_size:
                         try:
                             next_item_id = self._submission_queue.get_nowait()
                             self._submission_queue.task_done()
                             batch_item_ids.append(next_item_id)
                         except asyncio.QueueEmpty:
                             break

                 except asyncio.TimeoutError:
                      if not batch_item_ids:
                          continue
                      print(f"[{time.monotonic():.4f}] Worker: Timeout, but gathered {len(batch_item_ids)} items. Proceeding to send.")
                      pass

                 except Exception as e:
                     print(f"[{time.monotonic():.4f}] Worker encountered error getting items: {e}")
                     await asyncio.sleep(1.0)
                     continue

                 if batch_item_ids:
                     # Acquire semaphore permit before starting the potentially long-running batch submission
                     async with self._semaphore:
                         # Perform the actual batch sending using the injected function
                         await self._perform_send_batch(batch_item_ids)
                 else:
                     pass

        except asyncio.CancelledError:
            print(f"[{time.monotonic():.4f}] Sender worker received cancellation signal.")
        except Exception as e:
            print(f"[{time.monotonic():.4f}] Sender worker encountered major error: {e}")

        print(f"[{time.monotonic():.4f}] Sender worker finished.")


    async def _perform_send_batch(self, batch_item_ids: List[str]):
        """
        Internal method to execute the batch submission using the injected function and process results.
        Assumes this method is called within the context of an acquired semaphore permit.
        Runs within the async thread's event loop.
        """
        batch_info = [] # [{"request_id": id, "payload": payload}]
        payloads_for_server = [] # List of just payloads to pass to the injected function

        # Ensure the original requests are still pending before forming the batch data
        valid_item_ids_for_batch = [req_id for req_id in batch_item_ids if req_id in self._pending_requests]

        if not valid_item_ids_for_batch:
             # print(f"[{time.monotonic():.4f}] Batch contains no valid pending items after worker picked them up.")
             return # Nothing valid to send

        # Build the payload list for the injected function using only valid IDs
        for req_id in valid_item_ids_for_batch:
             req_info = self._pending_requests[req_id] # Should exist based on valid_item_ids_for_batch
             batch_info.append({"request_id": req_id, "payload": req_info["payload"]})
             payloads_for_server.append(req_info["payload"])

        # --- CALL THE INJECTED BATCH SUBMISSION FUNCTION ---
        # print(f"[{time.monotonic():.4f}] Submitting batch of {len(payloads_for_server)} items using injected function...")
        self._stats["num_batches_submitted"] += 1
        self._stats["actual_batch_sizes"].append(len(payloads_for_server))

        start_time = time.monotonic()
        try:
            # Call the function provided during initialization
            # It must return results in the same order as input payloads_for_server
            results_list = await self._batch_submit_func(payloads_for_server, self._session)

            submission_duration = time.monotonic() - start_time
            self._stats["total_batch_submission_duration_seconds"] += submission_duration
            self._stats["num_successful_batches"] += 1
            self._stats["total_items_processed_in_batches"] += len(payloads_for_server)


            # Process the results returned by the injected function
            # The order of results_list is assumed to match the order of payloads_for_server
            if len(results_list) != len(batch_info):
                 print(f"[{time.monotonic():.4f}] Warning: Injected function returned {len(results_list)} results, but batch had {len(batch_info)} items. Cannot reliably match results.")
                 match_count = min(len(results_list), len(batch_info))
            else:
                 match_count = len(batch_info)

            for i in range(match_count):
                 req_id = batch_info[i]["request_id"] # Get the original ID
                 result = results_list[i]          # Get the corresponding result

                 if req_id in self._pending_requests:
                      req_info = self._pending_requests[req_id]
                      future = req_info["future"]
                      if not future.done():
                           future.set_result(result)
                           del self._pending_requests[req_id]
                      else:
                           if req_id in self._pending_requests:
                                del self._pending_requests[req_id]
                 else:
                     print(f"[{time.monotonic():.4f}] Warning: Received result for unknown or already completed request ID {req_id[:6]}... Result: {result}")

        except Exception as e:
            submission_duration = time.monotonic() - start_time
            self._stats["total_batch_submission_duration_seconds"] += submission_duration # Still record time even on failure
            self._stats["num_failed_batches"] += 1
            # print error stack trace for debugging
            traceback.print_exc()
            print(f"[{time.monotonic():.4f}] Error calling or processing results from injected function for batch: {e}")
            # Handle failure of the injected function.
            # Items remain in _pending_requests, rely on timeout/stop cleanup.
            # To avoid silently failing, throw an exception directly to the caller
            for req_id in valid_item_ids_for_batch:
                if req_id in self._pending_requests:
                    req_info = self._pending_requests[req_id]
                    future = req_info["future"]
                    if not future.done():
                        future.set_exception(e)
                        del self._pending_requests[req_id]

    def get_stats(self) -> Dict[str, Any]:
        """Returns the collected performance statistics."""
        stats_copy = self._stats.copy()
        if stats_copy["num_successful_batches"] > 0:
            stats_copy["avg_successful_batch_submission_duration_seconds"] = \
                self._stats["total_batch_submission_duration_seconds"] / stats_copy["num_successful_batches"] \
                if self._stats["num_successful_batches"] > 0 else 0 # Avoid division by zero if only failures
        else:
            stats_copy["avg_successful_batch_submission_duration_seconds"] = 0

        if stats_copy["num_batches_submitted"] > 0: # Calculate overall average if any batch was submitted
            stats_copy["avg_overall_batch_submission_duration_seconds"] = \
                self._stats["total_batch_submission_duration_seconds"] / stats_copy["num_batches_submitted"]
        else:
            stats_copy["avg_overall_batch_submission_duration_seconds"] = 0

        if self._stats["actual_batch_sizes"]:
            stats_copy["avg_actual_batch_size"] = sum(self._stats["actual_batch_sizes"]) / len(self._stats["actual_batch_sizes"])
        else:
            stats_copy["avg_actual_batch_size"] = 0
        return stats_copy

    def print_stats(self):
        """Prints the collected performance statistics."""
        stats_to_print = self.get_stats()
        print(f"[{time.monotonic():.4f}] --- RequestProcessor Statistics ---")
        for key, value in stats_to_print.items():
            if key == "actual_batch_sizes":
                if value:  # Check if the list of batch sizes is not empty
                    batch_size_counts = collections.Counter(value)
                    # Format as a list of (batch_size, count) tuples, sorted by batch_size
                    formatted_batch_sizes = sorted(batch_size_counts.items())
                    print(f"  {key}: {formatted_batch_sizes}")
                else:
                    print(f"  {key}: []")  # Print an empty list if no batches were processed
            elif isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                # Handles other data types including other lists (if any)
                print(f"  {key}: {value}")
        print(f"[{time.monotonic():.4f}] --- End Statistics ---")

    def reset_stats(self):
        """Resets all collected performance statistics to their initial values."""
        print(f"[{time.monotonic():.4f}] Resetting RequestProcessor statistics.")
        self._reset_stats_internal()
