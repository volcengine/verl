# Copyright 2025 Meituan Ltd. and/or its affiliates
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
import random
import time


class SimpleStreamingSystem:
    """简化的流式处理系统演示"""

    def __init__(self, max_concurrent_tasks: int = 4):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.data_queue = asyncio.Queue()
        self.result_queue = asyncio.Queue()
        self.consumer_count = 0

    # 数据流协程
    async def data_stream(self):
        # 添加初始数据
        # 准备测试数据
        test_data = [{"id": f"task_{i}", "content": f"数据_{i}"} for i in range(8)]
        await self.add_data_stream(test_data)

        # 模拟后续数据流
        await asyncio.sleep(3)
        print("\n添加第二批数据...")
        extra_data = [{"id": f"extra_{i}", "content": f"额外数据_{i}"} for i in range(5)]
        await self.add_data_stream(extra_data)

        # 发送结束信号
        await asyncio.sleep(1)
        await self.data_queue.put("DONE")
        print("发送结束信号")

    async def add_data_stream(self, data_list: list[dict]):
        """模拟数据流"""
        print("开始添加数据流...")

        for i, data_item in enumerate(data_list):
            await self.data_queue.put(data_item)
            print(f"数据 {data_item['id']} 进入待处理队列")

            # 模拟数据流的间隔
            if i < len(data_list) - 1:  # 最后一个不等待
                await asyncio.sleep(0.8)

        print("初始数据流添加完成")

    async def _process_data_async(self, data_item: dict):
        """异步处理单个数据项"""
        data_id = data_item["id"]
        content = data_item["content"]

        # 模拟不同的处理时间（1-3秒）
        processing_time = random.uniform(1, 3)

        print(f"    开始处理 {data_id}，预计耗时 {processing_time:.1f}s")

        # 异步等待处理完成
        await asyncio.sleep(processing_time)

        result = {
            "id": data_id,
            "processed_content": f"处理后的{content}",
            "processing_time": round(processing_time, 2),
            "completed_at": time.time(),
        }

        # 立即放入结果队列
        await self.result_queue.put(result)
        print(f"    {data_id} 处理完成！(耗时 {processing_time:.1f}s) -> 进入结果队列")

    async def _submit_worker(self):
        """流式提交工作协程"""
        active_tasks = set()

        print("流式提交器启动...")

        while True:
            # 获取待处理数据
            data_item = await self.data_queue.get()

            if data_item == "DONE":
                print("收到结束信号，等待剩余任务完成...")
                if active_tasks:
                    await asyncio.gather(*active_tasks, return_exceptions=True)
                break

            # 检查并发数限制
            while len(active_tasks) >= self.max_concurrent_tasks:
                print(f"达到最大并发数 {self.max_concurrent_tasks}，等待任务完成...")
                done_tasks, active_tasks = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)

                # 清理完成的任务
                for task in done_tasks:
                    try:
                        await task
                        print(f"task 完成 {task}")
                    except Exception as e:
                        print(f"任务执行失败: {e}")

            # 立即提交新任务
            task = asyncio.create_task(self._process_data_async(data_item), name=f"active {data_item}")
            active_tasks.add(task)

            print(f"提交任务 {data_item['id']}，当前并发数: {len(active_tasks)}")

    async def _consumer_worker(self):
        """结果消费协程"""
        print("消费者启动...")

        while True:
            try:
                # 从结果队列获取处理结果
                result = await asyncio.wait_for(self.result_queue.get(), timeout=2.0)

                self.consumer_count += 1

                print(
                    f"消费 #{self.consumer_count}: {result['id']} "
                    f"(处理时间 {result['processing_time']}s) - {result['processed_content']}"
                )

            except asyncio.TimeoutError:
                print("    消费者等待中...")
                await asyncio.sleep(0.5)

    async def run_demo(self):
        """运行演示"""
        print("=" * 60)
        print(f"最大并发数: {self.max_concurrent_tasks}")
        print("=" * 60)

        # 启动核心协程
        stream_task = asyncio.create_task(self.data_stream())
        submit_task = asyncio.create_task(self._submit_worker())
        consumer_task = asyncio.create_task(self._consumer_worker())

        try:
            # 等待数据流完成
            await stream_task
            print("数据流完成")

            # 等待处理完成
            await submit_task
            print("所有任务处理完成")

        finally:
            # 清理
            submit_task.cancel()
            consumer_task.cancel()
            await asyncio.gather(submit_task, consumer_task, return_exceptions=True)

        print(f"\n最终统计: 消费了 {self.consumer_count} 个结果")


async def main():
    """主函数"""
    system = SimpleStreamingSystem(max_concurrent_tasks=3)
    await system.run_demo()


if __name__ == "__main__":
    asyncio.run(main())
