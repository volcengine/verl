# Fully Async Policy 测试指南

本文档介绍如何测试完全异步PPO训练系统的各种功能和性能。

## 📋 测试概览

我们提供了多种类型的测试，涵盖从单元测试到端到端测试的完整测试套件：

### 测试类型
1. **单元测试** - 测试各个组件的独立功能
2. **集成测试** - 测试组件间的协作
3. **端到端测试** - 测试完整的训练流程
4. **性能基准测试** - 评估系统性能特征
5. **压力测试** - 测试系统在极限条件下的表现

## 🚀 快速开始

### 1. 端到端测试
最简单的方式是运行端到端测试，验证系统基本功能：

```bash
# 基本E2E测试
./run_e2e_test.sh

# 使用环境变量自定义配置
NUM_GPUS=4 MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct ./run_e2e_test.sh
```

### 2. 单元测试
运行组件级别的单元测试：

```bash
# 运行所有单元测试
cd unittest/
python test_fully_async_components.py

# 或者使用pytest（如果安装）
pytest test_components_pytest.py -v
```

### 3. 性能基准测试
评估系统性能特征：

```bash
# 运行完整的性能基准测试
./run_benchmark.sh

# 自定义GPU数量和策略
NUM_GPUS=8 ACTOR_STRATEGY=fsdp2 ./run_benchmark.sh
```

## 📊 测试脚本详解

### run_e2e_test.sh
- **目的**: 端到端功能验证
- **配置**: 最小化配置，快速验证基本功能
- **时长**: 约5-10分钟
- **用法**: `./run_e2e_test.sh`

**环境变量**:
- `NUM_GPUS`: GPU数量 (默认: 4)
- `MODEL_ID`: 使用的模型ID (默认: Qwen/Qwen2.5-0.5B-Instruct)
- `MODEL_PATH`: 模型存储路径

### run_benchmark.sh
- **目的**: 性能基准测试
- **配置**: 多种配置组合，评估性能影响
- **时长**: 约30-60分钟
- **用法**: `./run_benchmark.sh`

**测试覆盖**:
1. 不同新鲜度阈值的影响
2. 不同队列大小的性能表现
3. 生成间隔对吞吐量的影响
4. GPU资源分配的优化
5. 暂停/恢复功能测试

### test_fully_async_components.py
- **目的**: 单元和集成测试
- **配置**: 使用Mock对象的孤立测试
- **时长**: 约2-5分钟
- **用法**: `python unittest/test_fully_async_components.py`

**测试覆盖**:
- MessageQueue的基本功能
- 参数同步器的重试机制
- Rollouter的暂停/恢复
- 新鲜度指标计算
- 错误处理和超时机制

## 🔧 测试配置

### 最小化测试配置
用于快速验证功能：

```yaml
# 基本配置
data:
  train_batch_size: 4
  max_prompt_length: 512
  max_response_length: 1024

trainer:
  total_training_steps: 2
  n_gpus_per_node: 2

rollout:
  n_gpus_per_node: 2

async_training:
  staleness_threshold: 3
  max_queue_size: 100
```

### 性能测试配置
用于评估系统性能：

```yaml
# 性能配置
data:
  train_batch_size: 16
  max_prompt_length: 512
  max_response_length: 1024

trainer:
  total_training_steps: 10
  n_gpus_per_node: 6

rollout:
  n_gpus_per_node: 2

async_training:
  staleness_threshold: 3
  max_queue_size: 1000
  generation_timeout: 30.0
```

## 📈 测试结果分析

### 成功指标
测试成功应满足以下条件：

1. **功能正确性**:
   - 样本成功生成和消费
   - 参数同步正常工作
   - 暂停/恢复功能响应

2. **性能表现**:
   - 样本生成速率 > 目标吞吐量
   - 队列利用率在合理范围(50-80%)
   - 新鲜度指标符合预期

3. **稳定性**:
   - 无内存泄漏
   - 无死锁或竞争条件
   - 优雅处理错误情况

### 失败排查
常见问题及解决方案：

1. **Ray连接失败**:
   ```bash
   # 重新初始化Ray
   ray stop
   ray start --head
   ```

2. **GPU内存不足**:
   ```bash
   # 减少批大小或使用梯度检查点
   data.train_batch_size=2
   actor_rollout_ref.model.enable_gradient_checkpointing=True
   ```

3. **队列阻塞**:
   ```bash
   # 调整队列大小和新鲜度阈值
   async_training.max_queue_size=500
   async_training.staleness_threshold=5
   ```

## 🎯 特定功能测试

### 测试暂停/恢复功能
```python
# 在Python脚本中测试
import ray
from fully_async_rollouter import FullyAsyncRollouter

rollouter = FullyAsyncRollouter.remote(config, ...)

# 测试暂停
result = ray.get(rollouter.pause_rollout.remote())
assert result == True

# 测试恢复
result = ray.get(rollouter.resume_rollout.remote())
assert result == True
```

### 测试新鲜度控制
```python
# 测试样本过期机制
queue = MessageQueueClient.remote(max_staleness=3)

# 放入旧版本样本
queue.put_samples.remote(sample, param_version=1)

# 用新版本获取（应该被拒绝）
result = ray.get(queue.get_samples.remote(current_param_version=5))
assert result is None
```

### 测试参数同步
```python
# 测试同步重试机制
sync = ParameterSynchronizer.remote(config, actor_wg, rollout_wg)

# 测试成功同步
result = ray.get(sync.sync_weights.remote())
assert result == True
```

## 📝 测试报告

### 基准测试报告
运行`./run_benchmark.sh`后，会在`benchmark_results_*/`目录下生成：

- `performance_report.md` - 详细的性能报告
- `summary.txt` - 关键指标摘要
- `*.log` - 各项测试的详细日志

### 关键指标
需要关注的性能指标：

1. **吞吐量指标**:
   - 样本生成速率 (samples/second)
   - 训练步数完成速率 (steps/second)

2. **延迟指标**:
   - 样本平均年龄 (average sample age)
   - 参数同步延迟 (sync latency)

3. **资源利用率**:
   - GPU利用率 (GPU utilization)
   - 内存使用量 (memory usage)
   - 队列利用率 (queue utilization)

## 🔄 CI/CD 集成

### GitHub Actions 示例
```yaml
name: Fully Async Policy Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Setup Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest

    - name: Run unit tests
      run: |
        cd recipe/fully_async_policy/unittest/
        python test_fully_async_components.py

    - name: Run E2E test (if GPUs available)
      run: |
        if nvidia-smi; then
          cd recipe/fully_async_policy/
          ./run_e2e_test.sh
        fi
```

## 🛠️ 开发者测试

### 添加新测试
1. **单元测试**: 在`unittest/test_fully_async_components.py`中添加新的测试类
2. **集成测试**: 在相应的集成测试类中添加新方法
3. **性能测试**: 在`run_benchmark.sh`中添加新的基准测试场景

### 测试最佳实践
1. **隔离性**: 每个测试应该独立，不依赖其他测试
2. **可重现性**: 使用固定的随机种子和确定性配置
3. **清理**: 测试结束后清理资源，避免影响后续测试
4. **文档**: 为新测试添加清晰的文档说明

## ❓ 常见问题

**Q: 测试失败，提示Ray连接错误**
A: 确保Ray集群正常运行，或重新启动Ray

**Q: 内存不足错误**
A: 减少批大小或在测试配置中启用参数卸载

**Q: 测试运行时间过长**
A: 使用更小的模型或减少训练步数进行快速测试

**Q: 如何添加自定义测试？**
A: 参考现有测试模式，在对应的测试文件中添加新的测试方法

通过这套完整的测试系统，可以确保fully async policy系统的可靠性、性能和稳定性。

