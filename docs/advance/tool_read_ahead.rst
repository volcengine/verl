Tool read-ahead
===============

Last updated: 10/12/2026.

When a tool returns the identifier of the next record, an agent can start reading
that record while the model generates its next turn. ``verl.tools.read_ahead.ReadAhead``
provides this opt-in overlap for immutable, read-only data. The model still chooses
the next call: the prepared result is delivered only when its requested key matches.
This does not change model inference or the backend request scheduler.

Use it for snapshot pagination or linked records whose next identifier is available
in the previous result. It is unsuitable for writes, mutable data, or a next query
that depends on reasoning the model has not performed yet. Even a read-only API can
have billing, rate limits, or server load: speculative reads must be acceptable to
the tool owner.

Using the helper
----------------

.. code-block:: python

   from verl.tools.read_ahead import ReadAhead

   reader = ReadAhead(fetch_record, lambda record: record.get("next_record_id"))
   try:
       # Call this only after the model actually requests this record.
       result = await reader.read(record_id)
       # Deliver result to the model; one successor may now be read concurrently.
   finally:
       # Close at trajectory end, after all demanded reads have stopped.
       await reader.aclose()

Keep the reader alive across model turns, with one reader per trajectory. Calls to
``read`` must be serial. The key must identify the entire operation, including any
arguments, authorization scope, or snapshot affecting its result. ``fetch`` must be
an async coroutine that supports cancellation, and ``successor`` must be a pure
callback using only the result just returned to the caller.

There is at most one pending result. A completed prefetch does not recursively read
its successor until the model consumes it. A different requested key cancels and
joins the unused task before fetching the actual key. A speculative failure retries
the demanded read; a demanded failure propagates. ``aclose`` retrieves unused errors
and cancels unused work, and is safe to call again.

This bounds speculation per trajectory, not across a worker or cluster. Apply the
tool service's capacity limits to ``fetch`` and measure wasted reads when predictions
miss. Cancelling an HTTP client request does not guarantee the remote server stops
work. Wrapping blocking I/O with ``asyncio.to_thread`` also does not make that I/O
cancellable.

ToolAgentLoop example
---------------------

The example at ``examples/tutorial/agent_loop_get_started/read_ahead.py`` uses
``BaseTool`` and a custom ``ToolAgentLoop``. Each ``run`` scopes a reader and an HTTP
session to its own context, serializes reads with a lock, and closes both in
``finally``. This matters because the standard loop calls ``BaseTool.release`` after
each individual tool call; ending the reader there would cancel the desired overlap.
No tasks or sessions are stored in serialized trajectory output.

The record service must expose a fixed snapshot as ``GET <records_url>/<id>.json``:

.. code-block:: json

   {"record_id": 123, "value": 17, "next_record_id": 456}

The terminal record has ``"next_record_id": null``. Deploy the service where all
rollout workers can reach it, and edit ``records_url`` in ``read_ahead_agent.yaml``.
It is a snapshot URL, not a mutable "latest" endpoint. The example needs ``aiohttp``,
which is already used by the rollout environment.

Add these overrides to an existing tool-agent training configuration:

.. code-block:: bash

   actor_rollout_ref.rollout.agent.default_agent_loop=read_ahead_agent \
   actor_rollout_ref.rollout.agent.agent_loop_config_path=examples/tutorial/agent_loop_get_started/read_ahead_agent.yaml \
   actor_rollout_ref.rollout.multi_turn.tool_config_path=examples/tutorial/agent_loop_get_started/read_ahead_tools.yaml

Keep the model's existing tool parser configuration. Dataset prompts should ask the
model to follow the returned ``next_record_id`` one call at a time. The example
starts with ``read_ahead: false``; set it to ``true`` in the agent YAML to enable
overlap. Both modes use the same HTTP fetch and return the same record schema.

Measured scope
--------------

An experiment on 2026-10-12 used the full MiMo-V2.6-Flash-RL model (revision
``5711b268169967567844e1e560e8a3966da959b1``), 8 MI355X GPUs, and the real verl
ToolAgentLoop with 64 trajectories. Tools read immutable linked records with injected
0.25/0.5/1/2 second delays. The same running server executed off/on/off phases, each
with a four-task quality check, three stable warmups, and three measured batches.
Model, prompts, native FCFS scheduling and sparse4096 graph configuration were fixed.
Startup and warmup were excluded. vLLM version was ``0.30.1rc1.dev143+g29468dde8``.

.. list-table:: Batch completion time, seconds
   :header-rows: 1

   * - Mode
     - Three measured batches
     - Mean ± sample standard deviation
   * - Off, before
     - 31.646806 / 31.437801 / 31.572598
     - 31.552402 ± 0.105956
   * - On
     - 26.126972 / 26.439359 / 26.370070
     - 26.312134 ± 0.164054
   * - Off, recheck
     - 31.309377 / 31.495038 / 31.507548
     - 31.437321 ± 0.110979

Completion time fell 16.6% against the first baseline and 16.3% against the recheck;
baseline drift was -0.36%. The last trajectory's exposed read wait fell from 24.518
to 18.914 seconds. Its physical reads still took 24.521 seconds, with 5.609 seconds
overlapped. Its model service calls took 6.927 seconds versus 6.429 seconds before:
the benefit was reduced exposed tool waiting, not faster model computation.

All 576 trajectories passed checks of ordered record IDs, returned values,
calculator inputs, and final answers. Each batch made 624 model requests, 496
physical reads, and 64 calculator calls. Each enabled batch reused 432 prefetches
without extra physical reads; no readers remained pending. Output token totals
were 18704/18711/18920, 18596/18820/18729, and 18879/18618/18780 respectively:
token-by-token equality is not claimed.

These measurements used the helper in a dedicated linked-record harness before
this contribution. The four helper method bodies are equivalent to the measured
implementation; the HTTP example above is new and has CPU integration coverage,
not a new GPU performance measurement. The original result archive is retained by
the contributor with SHA256
``3fc719bd956a37a6ff4ea1e55a9d7fecebacbae288d2faf849de788a45670297``.
There were no RL optimizer or weight updates. These results do not establish a
general RL speedup, benefits for real remote services, or an arbitrary prediction
hit rate. The deterministic chain had no shared tool executor capacity limit.

Related work
------------

- `Conveyor <https://arxiv.org/abs/2406.00059v2>`_: overlap tool execution with
  generation once required inputs are available. Here the overlap spans adjacent
  model turns, rather than a partially generated program.
- `PASTE <https://arxiv.org/abs/2603.18897v3>`_: bounded speculative tool work and
  matching actual calls before exposing results. This helper uses a deterministic
  successor, without its learned predictor or joint scheduling system.
- `LLMCompiler <https://arxiv.org/abs/2312.04511v3>`_: execute independent ready
  calls concurrently. Linked records retain their dependencies; this helper does
  not read through unseen future results.
