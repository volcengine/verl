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

import ray


# 配置1: 默认配置
class DefaultStreamingActor:
    """默认配置的流式处理Actor"""

    def __init__(self, actor_id: str):
        self.actor_id = actor_id
        self.processed_count = 0
        self.start_time = time.time()
        self.max_concurrent_tasks = 0
        self.current_tasks = 0

    async def process_data_async(self, data_item: dict) -> dict:
        """异步处理数据"""
        self.current_tasks += 1
        self.max_concurrent_tasks = max(self.max_concurrent_tasks, self.current_tasks)

        try:
            task_id = data_item["id"]
            processing_time = random.uniform(1, 3)

            print(f"[{self.actor_id}] 开始处理 {task_id} (当前并发: {self.current_tasks})")

            # CPU密集型任务模拟
            await asyncio.sleep(processing_time * 0.5)  # I/O部分

            # 模拟CPU计算
            total = 0
            for i in range(int(processing_time * 100000)):  # CPU密集计算
                total += i * 0.001

            await asyncio.sleep(processing_time * 0.5)  # 更多I/O

            self.processed_count += 1

            result = {
                "id": task_id,
                "actor_id": self.actor_id,
                "processing_time": processing_time,
                "processed_count": self.processed_count,
                "max_concurrent": self.max_concurrent_tasks,
                "compute_result": total,
                "completed_at": time.time(),
            }

            print(f"[{self.actor_id}] 完成处理 {task_id} (耗时: {processing_time:.1f}s)")
            return result

        finally:
            self.current_tasks -= 1

    def get_stats(self) -> dict:
        return {
            "actor_id": self.actor_id,
            "processed_count": self.processed_count,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "uptime": time.time() - self.start_time,
        }


# 配置2: 只设置 num_cpus
@ray.remote(num_cpus=4)
class HighCpuStreamingActor(DefaultStreamingActor):
    """高CPU配置的Actor"""

    pass


# 配置3: 只设置 max_concurrency
@ray.remote(max_concurrency=5)
class HighConcurrencyStreamingActor(DefaultStreamingActor):
    """高并发配置的Actor"""

    pass


# 配置4: 同时设置两者
@ray.remote(num_cpus=4, max_concurrency=8)
class OptimalStreamingActor(DefaultStreamingActor):
    """最优配置的Actor"""

    pass


# 配置5: 极端低配置
@ray.remote(num_cpus=1, max_concurrency=2)
class LowResourceStreamingActor(DefaultStreamingActor):
    """低资源配置的Actor"""

    pass


class RayStreamingSystemTest:
    """Ray流式处理系统测试"""

    def __init__(self):
        self.test_data = []
        self.results = {}

    def generate_test_data(self, count: int = 20) -> list[dict]:
        """生成测试数据"""
        return [
            {"id": f"task_{i:03d}", "content": f"测试数据_{i}", "priority": random.choice(["high", "normal", "low"])}
            for i in range(count)
        ]

    async def test_actor_configuration(self, actor_class, config_name: str, test_data: list[dict]) -> dict:
        """测试特定配置的Actor"""
        print(f"\n{'=' * 60}")
        print(f"测试配置: {config_name}")
        print(f"{'=' * 60}")

        # 创建Actor实例
        actor = actor_class.remote(config_name)

        start_time = time.time()

        # 并发提交所有任务
        print(f"提交 {len(test_data)} 个任务...")
        task_futures = []

        for i, data_item in enumerate(test_data):
            future = actor.process_data_async.remote(data_item)
            task_futures.append(future)

            # 模拟流式数据到达
            if i < len(test_data) - 1:
                await asyncio.sleep(0.1)  # 100ms间隔

        print("所有任务已提交，等待完成...")

        # 等待所有任务完成
        try:
            results = await asyncio.gather(*[asyncio.wrap_future(future.future()) for future in task_futures])
        except Exception as e:
            print(f"任务执行出错: {e}")
            results = []

        end_time = time.time()
        total_time = end_time - start_time

        # 获取Actor统计信息
        stats = ray.get(actor.get_stats.remote())

        # 计算性能指标
        performance_metrics = {
            "config_name": config_name,
            "total_tasks": len(test_data),
            "completed_tasks": len(results),
            "total_time": total_time,
            "throughput": len(results) / total_time if total_time > 0 else 0,
            "avg_processing_time": sum(r.get("processing_time", 0) for r in results) / len(results) if results else 0,
            "max_concurrent_tasks": stats["max_concurrent_tasks"],
            "actor_stats": stats,
            "success_rate": len(results) / len(test_data) if test_data else 0,
        }

        print(f"✅ 完成测试 {config_name}:")
        print(f"   总任务数: {performance_metrics['total_tasks']}")
        print(f"   完成任务数: {performance_metrics['completed_tasks']}")
        print(f"   总耗时: {performance_metrics['total_time']:.2f}s")
        print(f"   吞吐量: {performance_metrics['throughput']:.2f} tasks/s")
        print(f"   最大并发: {performance_metrics['max_concurrent_tasks']}")
        print(f"   成功率: {performance_metrics['success_rate'] * 100:.1f}%")

        return performance_metrics

    async def run_comprehensive_test(self):
        """运行综合测试"""
        print("🚀 开始Ray异步资源配置测试")
        print(f"Ray集群状态: {ray.cluster_resources()}")

        # 生成测试数据
        test_data = self.generate_test_data(15)  # 15个任务便于观察

        # 测试配置列表
        test_configs = [
            (DefaultStreamingActor, "默认配置 (无特殊设置)"),
            (HighCpuStreamingActor, "高CPU配置 (num_cpus=4)"),
            (HighConcurrencyStreamingActor, "高并发配置 (max_concurrency=5)"),
            (OptimalStreamingActor, "最优配置 (num_cpus=4, max_concurrency=8)"),
            (LowResourceStreamingActor, "低资源配置 (num_cpus=1, max_concurrency=2)"),
        ]

        results = {}

        # 逐个测试各种配置
        for actor_class, config_name in test_configs:
            try:
                result = await self.test_actor_configuration(actor_class, config_name, test_data)
                results[config_name] = result

                # 测试间隔
                await asyncio.sleep(2)

            except Exception as e:
                print(f"❌ 测试 {config_name} 失败: {e}")
                results[config_name] = {"error": str(e)}

        # 生成对比报告
        self.generate_comparison_report(results)

        return results

    def generate_comparison_report(self, results: dict):
        """生成对比报告"""
        print(f"\n{'=' * 80}")
        print("📊 配置对比报告")
        print(f"{'=' * 80}")

        # 表头
        print(f"{'配置名称':<25} {'吞吐量':<12} {'最大并发':<10} {'平均处理时间':<15} {'成功率':<10}")
        print("-" * 80)

        # 数据行
        best_throughput = 0
        best_config = ""

        for config_name, result in results.items():
            if "error" in result:
                print(f"{config_name:<25} {'错误':<12} {'':<10} {'':<15} {'':<10}")
                continue

            throughput = result.get("throughput", 0)
            max_concurrent = result.get("max_concurrent_tasks", 0)
            avg_time = result.get("avg_processing_time", 0)
            success_rate = result.get("success_rate", 0)

            print(
                f"{config_name:<25} {throughput:<12.2f} {max_concurrent:<10} "
                f"{avg_time:<15.2f} {success_rate * 100:<10.1f}%"
            )

            if throughput > best_throughput:
                best_throughput = throughput
                best_config = config_name

        print(f"\n🏆 最佳配置: {best_config} (吞吐量: {best_throughput:.2f} tasks/s)")

        # 详细分析
        print("\n📋 配置分析:")
        print("1. num_cpus 作用:")
        print("   - 资源预留: 确保Actor有足够计算资源")
        print("   - 节点选择: Ray选择有足够CPU的节点")
        print("   - 避免资源竞争: 防止过度调度")

        print("\n2. max_concurrency 作用:")
        print("   - 并发控制: 限制Actor内同时执行的任务数")
        print("   - 内存保护: 防止过多并发导致内存溢出")
        print("   - 性能调优: 平衡并发度和资源利用率")

        print("\n3. 建议配置:")
        print("   - CPU密集型任务: 设置较高的num_cpus，适中的max_concurrency")
        print("   - I/O密集型任务: 设置较低的num_cpus，较高的max_concurrency")
        print("   - 混合型任务: 平衡两个参数，根据实际测试调优")


async def run_resource_stress_test():
    """运行资源压力测试"""
    print(f"\n{'=' * 60}")
    print("🔥 资源压力测试")
    print(f"{'=' * 60}")

    # 创建多个不同配置的Actor
    actors = {
        "高并发低CPU": OptimalStreamingActor.remote("stress_test_1"),
        "低并发高CPU": ray.remote(num_cpus=8, max_concurrency=2)(DefaultStreamingActor).remote("stress_test_2"),
        "平衡配置": ray.remote(num_cpus=2, max_concurrency=4)(DefaultStreamingActor).remote("stress_test_3"),
    }

    # 大量并发任务
    heavy_workload = [{"id": f"heavy_{i}", "content": f"重载任务_{i}"} for i in range(50)]

    print("提交大量并发任务，观察资源使用...")

    all_futures = []
    for actor_name, actor in actors.items():
        print(f"向 {actor_name} 提交任务...")
        for task in heavy_workload[:15]:  # 每个Actor处理15个任务
            future = actor.process_data_async.remote(task)
            all_futures.append((actor_name, future))

    # 等待完成并记录时间
    start_time = time.time()
    results = []

    for actor_name, future in all_futures:
        try:
            result = await asyncio.wrap_future(future.future())
            results.append((actor_name, result))
        except Exception as e:
            print(f"{actor_name} 任务失败: {e}")

    end_time = time.time()

    print(f"压力测试完成，总耗时: {end_time - start_time:.2f}s")
    print(f"完成任务数: {len(results)}")

    # 按Actor分组统计
    actor_stats = {}
    for actor_name, result in results:
        if actor_name not in actor_stats:
            actor_stats[actor_name] = []
        actor_stats[actor_name].append(result)

    for actor_name, actor_results in actor_stats.items():
        avg_time = sum(r["processing_time"] for r in actor_results) / len(actor_results)
        print(f"{actor_name}: 完成 {len(actor_results)} 个任务, 平均耗时 {avg_time:.2f}s")


async def main():
    """主函数"""
    # 初始化Ray
    if not ray.is_initialized():
        ray.init(
            num_cpus=16,  # 设置足够的CPU资源
            object_store_memory=2000000000,  # 2GB
            ignore_reinit_error=True,
        )

    print("🎯 Ray异步资源配置测试")
    print(f"可用资源: {ray.cluster_resources()}")

    try:
        # 基础配置测试
        test_system = RayStreamingSystemTest()
        await test_system.run_comprehensive_test()

        # 压力测试
        await run_resource_stress_test()

        print("\n所有测试完成!")

    except Exception as e:
        print(f"测试执行失败: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # 清理资源
        ray.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
