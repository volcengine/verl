# 完全异步训练工作流 (Fully Async Training Workflow)

## 概述

本项目实现了基于现有 one step off policy 代码的完全异步训练工作流，将样本生成（Rollouter）和模型训练（Trainer）完全解耦，通过 MessageQueue 进行异步通信。

## 架构设计

### 核心组件

1. **MessageQueue**: 基于 ZeroMQ 的异步消息队列，作为 Ray Actor 存在
   - 管理生成的样本队列
   - 支持新鲜度控制，自动丢弃过期样本
   - 提供线程安全的生产者-消费者接口

2. **Rollouter**: 专门负责样本生成的组件
   - 持续循环生成训练样本
   - 支持暂停/恢复机制，用于参数更新
   - 实现新鲜度阈值控制，避免生成过多过期样本

3. **FullyAsyncTrainer**: 修改后的训练器
   - 从 MessageQueue 获取样本进行训练
   - 训练完成后通知 Rollouter 更新参数
   - 支持样本新鲜度监控和统计

4. **ParameterSynchronizer**: 参数同步模块
   - 基于 NCCL 实现高效的参数同步
   - 支持 Actor 到 Rollout 的参数传递

### 工作流程

```
┌─────────────┐    put_batch    ┌──────────────┐    get_batch    ┌─────────────┐
│  Rollouter  │ ──────────────► │ MessageQueue │ ──────────────► │   Trainer   │
│             │                 │              │                 │             │
│ - 生成样本   │                 │ - 队列管理    │                 │ - 模型训练   │
│ - 暂停/恢复  │                 │ - 新鲜度控制  │                 │ - 参数更新   │
│ - 新鲜度控制 │                 │ - 统计信息    │                 │ - 同步通知   │
└─────────────┘                 └──────────────┘                 └─────────────┘
       ▲                                                                 │
       │                        update_rollout_weights                   │
       └─────────────────────────────────────────────────────────────────┘
```

## 新鲜度控制机制

### 配置参数

- `freshness_threshold`: 新鲜度阈值，队列中超过此版本差异的样本会被丢弃
- `max_staleness_allowed`: 最大允许的新鲜度差异，Rollouter 会暂停生成
- `max_queue_size`: MessageQueue 的最大队列大小

### 控制逻辑

1. **样本丢弃**: 当样本的参数版本与当前 Trainer 版本差异超过 `freshness_threshold` 时，样本被丢弃
2. **生成暂停**: 当 Rollouter 的参数版本与 Trainer 版本差异超过 `max_staleness_allowed` 时，暂停生成
3. **队列管理**: 队列长度限制为 `freshness_threshold * batch_size`，避免内存溢出

## 性能优势

### 相比同步训练

- **GPU 利用率提升**: 生成和训练并行进行，减少 GPU 空闲时间
- **长尾样本优化**: 训练不需要等待最慢的样本生成完成
- **资源隔离**: 可以独立配置生成和训练的资源分配

### 相比 One Step Off Policy

- **更高的异步度**: 完全解耦生成和训练，支持多步异步
- **更灵活的控制**: 支持动态的新鲜度控制和队列管理
- **更好的监控**: 提供详细的统计信息和性能指标

## 使用方法

### 1. 安装依赖

```bash
pip install zmq filelock
```

### 2. 配置文件

使用 `config/fully_async_ppo_trainer.yaml` 配置文件，关键配置项：

```yaml
async_training:
  freshness_threshold: 3      # 新鲜度阈值
  max_staleness_allowed: 5    # 最大允许新鲜度差异
  max_queue_size: 1000        # 队列最大大小
  min_batch_count: 1          # 最小batch数量
  batch_timeout: 30.0         # 获取batch超时时间

actor_rollout_ref:
  rollout:
    mode: async               # 使用异步模式
    n_gpus: 4                # rollout专用GPU数量
    name: vllm               # 使用vLLM引擎
```

### 3. 启动训练

```bash
python -m recipe.one_step_off_policy.fully_async_main \
    data.train_files=~/data/train.parquet \
    data.val_files=~/data/val.parquet \
    actor_rollout_ref.model.path=Qwen/Qwen2-7B-Instruct \
    trainer.total_training_steps=1000
```

### 4. 监控训练

训练过程中会输出以下统计信息：

- `queue_size`: 当前队列大小
- `avg_sample_age`: 平均样本年龄（参数版本差异）
- `max_sample_age`: 最大样本年龄
- `param_version`: 当前参数版本
- `processed_samples`: 已处理样本数
- `dropped_samples`: 丢弃的过期样本数

## 性能调优建议

### 1. 资源分配

- **生成资源**: 根据模型大小和生成速度需求分配 GPU
- **训练资源**: 根据batch大小和训练复杂度分配 GPU
- **比例建议**: 生成:训练 = 1:2 到 1:3

### 2. 新鲜度控制

- **快速生成场景**: 降低 `freshness_threshold` (2-3)
- **慢速生成场景**: 提高 `freshness_threshold` (5-8)
- **队列大小**: 设置为 `freshness_threshold * batch_size * 2`

### 3. 网络优化

- **单节点**: MessageQueue 使用 IPC 协议
- **多节点**: MessageQueue 使用 TCP 协议，注意网络带宽

## 故障排除

### 常见问题

1. **队列为空**: 检查 Rollouter 是否正常运行，是否被新鲜度控制暂停
2. **内存溢出**: 减少 `max_queue_size` 或增加 `freshness_threshold`
3. **参数同步失败**: 检查 NCCL 配置和网络连接
4. **性能下降**: 调整资源分配比例，监控 GPU 利用率

### 调试模式

设置环境变量启用详细日志：

```bash
export VERL_LOGGING_LEVEL=DEBUG
export NCCL_DEBUG=INFO
```

## 与现有系统对比

| 特性 | 同步训练 | One Step Off | 完全异步 |
|------|----------|--------------|----------|
| 异步程度 | 无 | 一步 | 多步 |
| 资源利用率 | 低 | 中 | 高 |
| 实现复杂度 | 低 | 中 | 高 |
| 样本新鲜度 | 最新 | 一步延迟 | 可控延迟 |
| 内存使用 | 低 | 中 | 中-高 |

## 实验结果预期

基于现有 one step off policy 的实验结果，完全异步训练预期能够：

- **训练速度**: 相比同步训练提升 30-50%
- **GPU 利用率**: 提升至 85-95%
- **内存开销**: 增加 20-30%（主要用于队列缓存）
- **模型收敛**: 与同步训练基本一致（在合理的新鲜度控制下）

## 后续改进

1. **自适应新鲜度控制**: 根据训练进度动态调整新鲜度阈值
2. **多队列支持**: 支持不同优先级的样本队列
3. **分布式队列**: 支持跨节点的分布式消息队列
4. **更精细的资源调度**: 支持动态的资源分配和调整

