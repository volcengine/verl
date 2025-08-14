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

# 测试使用 asyncio 的 MessageQueue
# 对比 @ray.remote(num_cpus, max_concurrency) 参数的实际效果

import asyncio
import random

# 导入修改后的 MessageQueue
import time
from dataclasses import dataclass

import ray
from omegaconf import DictConfig

from recipe.fully_async_policy.message_queue import MessageQueue, MessageQueueClient, QueueSample


@dataclass
class TestConfig:
    """测试配置"""

    async_training: dict


def create_test_config() -> DictConfig:
    """创建测试配置"""
    from omegaconf import OmegaConf

    config_dict = {"async_training": {"staleness_threshold": 3}}
    return OmegaConf.create(config_dict)


class AsyncMessageQueueTester:
    """异步消息队列测试器"""

    def __init__(self):
        self.config = create_test_config()

    async def test_basic_async_operations(self):
        """测试基本异步操作"""
        print("\n🧪 测试基本异步操作")
        print("=" * 50)

        # 创建MessageQueue Actor
        queue_actor = MessageQueue.remote(self.config, max_queue_size=100)
        client = MessageQueueClient(queue_actor)

        # 测试异步放入样本
        test_samples = [
            QueueSample(
                data={"task_id": f"task_{i}", "content": f"测试数据_{i}"},
                rollout_metadata={"timestamp": time.time(), "version": 1},
            )
            for i in range(10)
        ]

        # 异步并发放入样本
        put_tasks = []
        for i, sample in enumerate(test_samples):
            task = asyncio.create_task(client.put_sample(sample, param_version=1), name=f"put_task_{i}")
            put_tasks.append(task)

        # 等待所有放入任务完成
        put_results = await asyncio.gather(*put_tasks)
        successful_puts = sum(put_results)

        print(f"✅ 成功放入 {successful_puts}/{len(test_samples)} 个样本")

        # 异步获取统计信息
        stats = await client.get_statistics()
        print(f"📊 队列统计: {stats}")

        # 异步获取样本
        samples_batch, queue_size = await client.get_samples(min_batch_count=5)
        print(f"📦 获取了 {len(samples_batch)} 个样本，剩余队列大小: {queue_size}")

        # 清理
        await client.shutdown()

        return successful_puts

    async def test_concurrent_producers_consumers(self):
        """测试并发生产者和消费者"""
        print("\n🏭 测试并发生产者和消费者")
        print("=" * 50)

        # 创建 MessageQueue Actor
        queue_actor = MessageQueue.remote(self.config, max_queue_size=200)
        client = MessageQueueClient(queue_actor)

        # 生产者协程
        async def producer(producer_id: int, sample_count: int):
            """生产者协程"""
            produced = 0
            for i in range(sample_count):
                sample = QueueSample(
                    data={
                        "producer_id": producer_id,
                        "task_id": f"producer_{producer_id}_task_{i}",
                        "content": f"来自生产者{producer_id}的数据{i}",
                    },
                    rollout_metadata={"producer_timestamp": time.time(), "producer_id": producer_id},
                )

                success = await client.put_sample(sample, param_version=1)
                if success:
                    produced += 1

                # 模拟生产间隔
                await asyncio.sleep(random.uniform(0.01, 0.1))

            print(f"🏭 生产者{producer_id} 完成，成功生产 {produced} 个样本")
            return produced

        # 消费者协程
        async def consumer(consumer_id: int, target_count: int):
            """消费者协程"""
            consumed = 0
            start_time = time.time()

            while consumed < target_count:
                try:
                    # 尝试获取样本，设置超时
                    sample = await asyncio.wait_for(client.get_sample(), timeout=2.0)

                    if sample is not None:
                        consumed += 1

                        if consumed % 10 == 0:
                            print(f"🍽️  消费者{consumer_id} 已消费 {consumed} 个样本")
                    else:
                        print(f"⚠️ 消费者{consumer_id} 收到空样本，队列可能已关闭")
                        break

                except asyncio.TimeoutError:
                    print(f"⏰ 消费者{consumer_id} 超时，检查队列状态...")
                    stats = await client.get_statistics()
                    if stats["queue_size"] == 0:
                        print(f"📭 队列为空，消费者{consumer_id} 等待...")
                        await asyncio.sleep(0.5)
                    continue

                # 模拟处理时间
                await asyncio.sleep(random.uniform(0.02, 0.05))

            elapsed = time.time() - start_time
            print(f"🍽️  消费者{consumer_id} 完成，消费了 {consumed} 个样本，耗时 {elapsed:.2f}s")
            return consumed

        # 启动并发生产者和消费者
        num_producers = 3
        num_consumers = 2
        samples_per_producer = 20

        # 创建生产者任务
        producer_tasks = [
            asyncio.create_task(producer(i, samples_per_producer), name=f"producer_{i}") for i in range(num_producers)
        ]

        # 创建消费者任务
        total_expected_samples = num_producers * samples_per_producer
        samples_per_consumer = total_expected_samples // num_consumers

        consumer_tasks = [
            asyncio.create_task(
                consumer(i, samples_per_consumer + (5 if i == 0 else 0)),  # 第一个消费者多处理一些
                name=f"consumer_{i}",
            )
            for i in range(num_consumers)
        ]

        # 等待所有任务完成
        start_time = time.time()

        producer_results = await asyncio.gather(*producer_tasks, return_exceptions=True)
        consumer_results = await asyncio.gather(*consumer_tasks, return_exceptions=True)

        end_time = time.time()

        # 统计结果
        total_produced = sum(r for r in producer_results if isinstance(r, int))
        total_consumed = sum(r for r in consumer_results if isinstance(r, int))

        print("\n📈 并发测试结果:")
        print(f"   总生产样本: {total_produced}")
        print(f"   总消费样本: {total_consumed}")
        print(f"   总耗时: {end_time - start_time:.2f}s")
        print(f"   生产效率: {total_produced / (end_time - start_time):.2f} samples/s")
        print(f"   消费效率: {total_consumed / (end_time - start_time):.2f} samples/s")

        # 最终统计
        final_stats = await client.get_statistics()
        print(f"📊 最终队列统计: {final_stats}")

        # 清理
        await client.shutdown()

        return total_produced, total_consumed

    async def compare_resource_configurations(self):
        """对比不同资源配置的效果"""
        print("\n⚡ 对比不同资源配置的效果")
        print("=" * 50)

        # 测试配置列表
        configs = [
            {"name": "默认配置", "num_cpus": None, "max_concurrency": None, "decorator": ray.remote},
            {
                "name": "高CPU低并发",
                "num_cpus": 4,
                "max_concurrency": 5,
                "decorator": lambda: ray.remote(num_cpus=4, max_concurrency=5),
            },
            {
                "name": "低CPU高并发",
                "num_cpus": 1,
                "max_concurrency": 20,
                "decorator": lambda: ray.remote(num_cpus=1, max_concurrency=20),
            },
            {
                "name": "平衡配置",
                "num_cpus": 2,
                "max_concurrency": 10,
                "decorator": lambda: ray.remote(num_cpus=2, max_concurrency=10),
            },
        ]

        results = {}

        for config in configs:
            print(f"\n🧪 测试配置: {config['name']}")
            print(f"   num_cpus: {config['num_cpus']}")
            print(f"   max_concurrency: {config['max_concurrency']}")

            # 动态创建MessageQueue类
            if config["num_cpus"] is None:
                QueueClass = MessageQueue
            else:
                QueueClass = config["decorator"]()(MessageQueue)

            # 创建queue实例
            queue_actor = QueueClass.remote(self.config, max_queue_size=100)
            client = MessageQueueClient(queue_actor)

            # 执行性能测试
            start_time = time.time()

            # 并发放入大量样本
            sample_count = 50
            put_tasks = []

            for i in range(sample_count):
                sample = QueueSample(
                    data={
                        "task_id": f"perf_test_{i}",
                        "config": config["name"],
                        "data_size": random.randint(100, 1000),
                    },
                    rollout_metadata={"config_test": True},
                )

                task = asyncio.create_task(client.put_sample(sample, param_version=1))
                put_tasks.append(task)

                # 模拟流式到达
                if i % 10 == 0:
                    await asyncio.sleep(0.01)

            # 等待所有put完成
            put_results = await asyncio.gather(*put_tasks)
            put_time = time.time() - start_time

            # 获取所有样本
            get_start_time = time.time()
            all_samples = []

            while True:
                samples_batch, queue_size = await client.get_samples(min_batch_count=1)
                if not samples_batch:
                    break
                all_samples.extend(samples_batch)

                if queue_size == 0:
                    break

            get_time = time.time() - get_start_time
            total_time = time.time() - start_time

            successful_puts = sum(put_results)

            # 记录结果
            results[config["name"]] = {
                "successful_puts": successful_puts,
                "retrieved_samples": len(all_samples),
                "put_time": put_time,
                "get_time": get_time,
                "total_time": total_time,
                "put_throughput": successful_puts / put_time if put_time > 0 else 0,
                "get_throughput": len(all_samples) / get_time if get_time > 0 else 0,
                "total_throughput": (successful_puts + len(all_samples)) / total_time if total_time > 0 else 0,
            }

            print(f"   ✅ 放入: {successful_puts}/{sample_count}")
            print(f"   📦 获取: {len(all_samples)}")
            print(f"   ⏱️  放入耗时: {put_time:.3f}s")
            print(f"   ⏱️  获取耗时: {get_time:.3f}s")
            print(f"   🚀 放入吞吐量: {successful_puts / put_time:.2f} ops/s")

            # 清理
            await client.shutdown()

            # 间隔
            await asyncio.sleep(1)

        # 生成对比报告
        print("\n📊 资源配置对比报告")
        print("=" * 80)
        print(f"{'配置名称':<15} {'放入吞吐量':<12} {'获取吞吐量':<12} {'总吞吐量':<12} {'总耗时':<10}")
        print("-" * 80)

        best_config = ""
        best_throughput = 0

        for config_name, result in results.items():
            put_throughput = result["put_throughput"]
            get_throughput = result["get_throughput"]
            total_throughput = result["total_throughput"]
            total_time = result["total_time"]

            print(
                f"{config_name:<15} {put_throughput:<12.2f} {get_throughput:<12.2f} "
                f"{total_throughput:<12.2f} {total_time:<10.3f}s"
            )

            if total_throughput > best_throughput:
                best_throughput = total_throughput
                best_config = config_name

        print(f"\n🏆 最佳配置: {best_config} (总吞吐量: {best_throughput:.2f} ops/s)")

        return results


async def main():
    """主函数"""
    # 初始化Ray
    if not ray.is_initialized():
        ray.init(
            num_cpus=8,
            object_store_memory=1000000000,  # 1GB
            ignore_reinit_error=True,
        )

    print("🎯 异步MessageQueue测试")
    print(f"Ray集群资源: {ray.cluster_resources()}")

    tester = AsyncMessageQueueTester()

    try:
        # 基本异步操作测试
        await tester.test_basic_async_operations()

        # 并发生产者消费者测试
        await tester.test_concurrent_producers_consumers()

        # 资源配置对比测试
        await tester.compare_resource_configurations()

        print("\n✅ 所有测试完成!")

        # 总结
        print("\n📋 总结:")
        print("1. 使用 asyncio 后的优势:")
        print("   - 真正的异步等待，不阻塞事件循环")
        print("   - 更好的并发性能")
        print("   - 与Ray的异步接口完美集成")

        print("\n2. 资源配置建议:")
        print("   - num_cpus: 控制CPU资源分配，影响计算密集型任务")
        print("   - max_concurrency: 控制并发数，影响I/O密集型任务")
        print("   - 对于MessageQueue: 推荐 num_cpus=2, max_concurrency=20")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()

    finally:
        ray.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
