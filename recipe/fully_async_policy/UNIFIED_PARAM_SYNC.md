# 统一参数同步器使用指南 (Unified Parameter Synchronizer Guide)

本文档说明了新的统一参数同步器 `UnifiedParameterSynchronizer` 的使用方法。该类合并了原有的多个同步器类的功能，提供了更简洁和统一的接口。

## 🏗️ 类合并说明

### 原有类结构（已合并）
- `ParameterSynchronizer` - 基础参数同步器
- `ParameterSyncManager` - Ray Actor形式的参数同步管理器
- `AsyncParameterSynchronizer` - 异步参数同步器

### 新的统一类
- `UnifiedParameterSynchronizer` - 统一参数同步器，包含所有功能

## 🚀 使用方法

### 1. 异步训练模式（推荐）
```python
from recipe.fully_async_policy.param_sync import UnifiedParameterSynchronizer

# 创建异步模式的参数同步器
param_synchronizer = UnifiedParameterSynchronizer(
    config=config,
    trainer_actor=trainer_actor,
    rollouter_actor=rollouter_actor
)

# 同步参数到rollouter
success = param_synchronizer.sync_to_rollouter(new_version=1)
```

### 2. Ray Actor模式
```python
from recipe.fully_async_policy.param_sync import ParameterSyncManager

# 创建Ray remote参数同步管理器
sync_manager = ParameterSyncManager.remote(config)

# 注册workers
success = ray.get(sync_manager.register_workers.remote(actor_workers, rollout_workers))

# 执行同步
success = ray.get(sync_manager.sync_parameters.remote())
```

### 3. 传统模式
```python
from recipe.fully_async_policy.param_sync import UnifiedParameterSynchronizer

# 创建传统模式的参数同步器
synchronizer = UnifiedParameterSynchronizer(config)

# 初始化同步组
success = synchronizer.initialize_sync_group(actor_workers, rollout_workers)

# 同步权重
success = synchronizer.sync_weights(actor_workers, rollout_workers)
```

## 🔄 向后兼容性

为了确保现有代码的兼容性，提供了以下别名：

```python
# 这些别名指向 UnifiedParameterSynchronizer
ParameterSynchronizer = UnifiedParameterSynchronizer
AsyncParameterSynchronizer = UnifiedParameterSynchronizer

# Ray remote版本
ParameterSyncManager = ray.remote(UnifiedParameterSynchronizer)
```

现有代码无需修改即可使用新的统一同步器。

## ⚙️ 初始化参数

```python
def __init__(self, config, trainer_actor=None, rollouter_actor=None, as_ray_actor=False):
```

- `config`: 配置对象（必需）
- `trainer_actor`: trainer actor引用（用于async模式）
- `rollouter_actor`: rollouter actor引用（用于async模式）
- `as_ray_actor`: 是否作为Ray actor使用

## 📊 主要方法

### 异步模式
- `sync_to_rollouter(new_version)`: 同步参数到rollouter
- `get_current_version()`: 获取当前参数版本

### Ray Actor模式
- `register_workers(actor_workers, rollout_workers)`: 注册workers
- `sync_parameters()`: 执行参数同步

### 传统模式
- `initialize_sync_group(actor_workers, rollout_workers)`: 初始化同步组
- `sync_weights(actor_workers, rollout_workers)`: 同步权重

### 通用方法
- `get_statistics()`: 获取统计信息
- `get_weights_info()`: 获取权重信息
- `cleanup()`: 清理资源

## 📈 统计信息

```python
stats = synchronizer.get_statistics()
# 返回：
{
    "sync_count": 15,
    "sync_failures": 0,
    "last_sync_time": 1640995200.0,
    "sync_group_initialized": True,
    "current_param_version": 15,
    "current_version": 15,
    "is_ready": True  # 仅在Ray actor模式下
}
```

## 🎯 优势

1. **统一接口**: 一个类支持所有同步模式
2. **向后兼容**: 现有代码无需修改
3. **灵活配置**: 支持多种初始化方式
4. **完整功能**: 包含所有原有类的功能
5. **简化维护**: 减少代码重复，便于维护

## 🔧 配置示例

```yaml
async_training:
  max_sync_retries: 3
  sync_timeout: 30.0
  sync_retry_delay: 1.0
  sync_monitor_interval: 60.0
  staleness_threshold: 3
```

---

*统一参数同步器简化了参数同步的使用，同时保持了所有原有功能的完整性。*

