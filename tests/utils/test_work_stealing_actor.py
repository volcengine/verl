import asyncio
import os
from typing import Any

from verl.workers.rollout.chat_scheduler.utils import WorkStealingActor


def test_work_stealing_behavior():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def async_test():
        num_workers = 3
        global_queue = asyncio.Queue()
        local_queues = [asyncio.Queue() for _ in range(num_workers)]

        executed_tasks: list[tuple[int, Any]] = []

        async def test_handler(worker_id: int, task: Any):
            executed_tasks.append((worker_id, task))
            await asyncio.sleep(0.05)

        (await local_queues[0].put("local-0"),)
        (await global_queue.put("global-1"),)
        (await local_queues[2].put("local-2"),)
        (await local_queues[2].put("extra-2"),)

        actors = [WorkStealingActor(i, local_queues, global_queue, test_handler) for i in range(num_workers)]
        for actor in actors:
            asyncio.get_event_loop().create_task(actor.run())

        await asyncio.sleep(2)

        tasks_seen = [task for _, task in executed_tasks]
        assert sorted(tasks_seen) == sorted(["local-0", "global-1", "local-2", "extra-2"])

        assert len(set(tasks_seen)) == 4
        assert len(executed_tasks) == 4

        owner_map = {
            "local-0": 0,
            "global-1": -1,
            "local-2": 2,
            "extra-2": 2,
        }

        stealing_detected = any(owner_map[task] != -1 and owner_map[task] != wid for wid, task in executed_tasks)

        assert stealing_detected, "Expected at least one task to be stolen"

    loop.run_until_complete(async_test())
    loop.close()


def test_queue_group_pop_from_longest():
    os.environ["VERL_QUEUE_LOGGING_LEVEL"] = "DEBUG"
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    from verl.workers.rollout.chat_scheduler.utils import QueueGroup

    async def run_test():
        group = QueueGroup(3, [asyncio.Queue() for _ in range(3)])
        await group.push(0, "q0-a")
        await group.push(1, "q1-a")
        await group.push(1, "q1-b")
        await group.push(2, "q2-a")
        await group.push(2, "q2-b")
        await group.push(2, "q2-c")
        await group.push(2, "q2-d")

        results = []
        for _ in range(6):
            item = group.pop_from_longest()
            results.append(item)

        # Items from q2 should come out first
        assert results[:3] == ["q2-a", "q2-b", "q2-c"] or results[:3] == ["q2-a", "q2-b", "q1-a"]

    loop.run_until_complete(run_test())
    loop.close()
