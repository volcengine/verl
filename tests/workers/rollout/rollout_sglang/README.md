# SGLang Rollout Tests

这个目录包含了专门针对 SGLang 后端的 rollout worker 测试。

## 📁 目录结构

```
tests/workers/rollout/rollout_sglang/
├── conftest.py                    # SGLang 专用的 pytest 配置和 fixtures
├── test_http_server_engine.py     # HTTP Server Engine Adapters 测试
├── run_tests.py                   # 测试运行脚本
└── README.md                      # 本文档
```

## 🎯 测试范围

### HTTP Server Engine Adapters
- `HttpServerEngineAdapter` - 同步 HTTP 适配器
- `AsyncHttpServerEngineAdapter` - 异步 HTTP 适配器
- `launch_server_process` - 服务器进程启动函数

### 测试覆盖的功能
- 服务器初始化和配置
- HTTP 请求处理（GET/POST）
- 异步操作支持
- 错误处理和重试机制
- 内存管理
- 分布式权重更新
- 路由器注册和注销
- 资源清理

## 🔧 测试环境配置

### SGLang 依赖
测试现在使用**真实的 SGLang 模块**进行集成测试，而不是 Mock 对象。

#### 安装要求
确保已安装 SGLang：
```bash
pip install sglang[all]
```

#### 环境变量
- `SGLANG_TEST_MODEL_PATH`: 测试用模型路径（默认：`/tmp/test_model`）

```bash
export SGLANG_TEST_MODEL_PATH="/path/to/your/test/model"
```

### 测试类型
- **集成测试**: 使用真实 SGLang 模块，标记为 `@pytest.mark.real_sglang`
- **单元测试**: 仅 Mock 外部依赖（HTTP 请求、进程管理），标记为 `@pytest.mark.mock_only`

## 🚀 运行测试

### 基本运行
```bash
# 进入测试目录
cd tests/workers/rollout/rollout_sglang

# 运行所有测试
python run_tests.py

# 或直接使用 pytest
python -m pytest
```

### 按测试类型运行
```bash
# 只运行 Mock 单元测试（不需要真实 SGLang 模型）
python run_tests.py -m "mock_only"

# 只运行真实 SGLang 集成测试
python run_tests.py -m "real_sglang"

# 排除慢速测试
python run_tests.py -m "not slow"
```

### 带选项运行
```bash
# 详细输出
python run_tests.py -v

# 带覆盖率报告
python run_tests.py -c

# 生成 HTML 覆盖率报告
python run_tests.py -c --html

# 并行运行测试（需要 pytest-xdist）
python run_tests.py -p

# 运行特定测试
python run_tests.py -k "test_init"

# 组合选项
python run_tests.py -v -c --html -x
```

### 直接使用 pytest
```bash
# 基本运行
pytest

# 详细输出
pytest -v -s

# 带覆盖率
pytest --cov=verl.workers.rollout.sglang_rollout --cov-report=term-missing

# 异步模式
pytest --asyncio-mode=auto

# 运行特定测试类
pytest test_http_server_engine.py::TestHttpServerEngineAdapter

# 运行特定测试方法
pytest test_http_server_engine.py::TestHttpServerEngineAdapter::test_init_with_router_registration
```

## 🔧 测试配置

### 真实 SGLang 集成
- **真实模块**: 测试使用真实的 `sglang` 模块和 `ServerArgs` 类
- **模型要求**: 某些测试可能需要真实的模型文件
- **环境配置**: 通过环境变量配置测试参数

### Fixtures
- `basic_adapter_kwargs` - 基本适配器参数
- `router_adapter_kwargs` - 带路由器配置的参数
- `non_master_adapter_kwargs` - 非主节点参数
- `real_adapter_kwargs` - 真实 SGLang 集成参数
- `sglang_test_model_path` - 测试模型路径
- `mock_launch_server_process` - Mock 服务器进程启动
- `mock_requests_*` - Mock HTTP 请求
- `mock_aiohttp_session` - Mock 异步 HTTP 会话

### 标记（Markers）
- `@pytest.mark.asyncio` - 异步测试
- `@pytest.mark.sglang` - SGLang 特定测试
- `@pytest.mark.integration` - 集成测试
- `@pytest.mark.slow` - 慢速测试
- `@pytest.mark.real_sglang` - 需要真实 SGLang 安装的测试
- `@pytest.mark.mock_only` - 仅使用 Mock 依赖的测试

## 📊 测试统计

- **总测试用例**: 50+
- **测试类**: 8 个主要测试类
- **覆盖的方法**: 所有公共方法
- **集成程度**: 真实 SGLang 模块 + Mock 外部依赖

## 🐛 故障排除

### 常见问题

1. **SGLang 导入错误**
   ```
   ModuleNotFoundError: No module named 'sglang'
   ```
   - 解决方案：安装 SGLang
   ```bash
   pip install sglang[all]
   ```

2. **模型路径错误**
   ```
   FileNotFoundError: Model path not found
   ```
   - 解决方案：设置正确的模型路径
   ```bash
   export SGLANG_TEST_MODEL_PATH="/path/to/valid/model"
   ```

3. **异步测试失败**
   ```
   RuntimeError: This event loop is already running
   ```
   - 确保使用 `pytest --asyncio-mode=auto`

4. **覆盖率报告问题**
   ```
   Coverage.py warning: No data was collected
   ```
   - 确保模块路径正确：`verl.workers.rollout.sglang_rollout`

### 调试测试
```bash
# 运行单个测试并查看详细输出
pytest test_http_server_engine.py::TestHttpServerEngineAdapter::test_init_with_router_registration -v -s

# 在测试失败时进入调试器
pytest test_http_server_engine.py --pdb

# 显示最慢的测试
pytest test_http_server_engine.py --durations=10

# 只运行快速的 Mock 测试
pytest -m "mock_only" -v
```

### 性能测试
```bash
# 运行所有集成测试（可能较慢）
pytest -m "real_sglang" -v

# 跳过慢速测试
pytest -m "not slow" -v
```

## 🔗 相关文档

- [主要 rollout 测试](../README_tests.md)
- [HTTP Server Engine 实现](../../../../verl/workers/rollout/sglang_rollout/http_server_engine.py)
- [SGLang 官方文档](https://github.com/sgl-project/sglang)

## 📝 贡献指南

### 添加新测试
1. 在相应的测试类中添加新方法
2. 使用描述性的测试方法名
3. 包含详细的文档字符串
4. 使用适当的 fixtures
5. 添加适当的测试标记：
   - `@pytest.mark.real_sglang` - 如果需要真实 SGLang
   - `@pytest.mark.mock_only` - 如果只需要 Mock
   - `@pytest.mark.slow` - 如果测试运行较慢

### 测试命名约定
- 测试方法以 `test_` 开头
- 使用描述性名称，如 `test_init_with_router_registration`
- 测试类以 `Test` 开头
- 边缘案例测试包含具体场景描述

### Mock 使用指南
- **选择性 Mock**: 只 Mock 外部依赖（HTTP 请求、进程管理等）
- **保留真实**: 使用真实的 SGLang 模块进行核心逻辑测试
- 优先使用现有的 fixtures
- 为新的外部依赖创建新的 fixtures
- 验证 Mock 对象的调用次数和参数 