# HTTP Server Engine Tests

这个目录包含了 HTTP Server Engine Adapters 的完整单元测试。

## 文件结构

- `test_http_server_engine.py` - 完整的测试套件，包含所有测试用例
- `run_tests.py` - 测试运行脚本
- `conftest.py` - pytest 配置和 fixtures
- `README_tests.md` - 本说明文档

## 测试覆盖范围

### 测试的类和功能

1. **launch_server_process 函数**
   - 成功启动服务器进程
   - 非主节点处理
   - 超时处理
   - 进程意外终止

2. **HttpServerEngineAdapter 类**
   - 初始化（带/不带路由器注册）
   - HTTP 请求处理（成功/失败/重试）
   - 权重更新方法
   - 文本生成
   - 缓存管理
   - 内存管理
   - 分布式权重更新
   - 生成控制（暂停/继续）
   - 关闭和清理

3. **AsyncHttpServerEngineAdapter 类**
   - 异步初始化
   - 会话管理
   - 异步 HTTP 请求
   - 异步内存管理
   - 上下文管理器支持
   - 异步清理

4. **边缘案例和错误处理**
   - 空参数和 None 值处理
   - 大负载处理
   - Unicode 和特殊字符
   - 格式错误的响应
   - 超时场景
   - 网络分区恢复
   - 资源清理
   - 数据类型处理

### 测试统计

- **总测试用例**: 67+
- **测试类**: 9 个主要测试类
- **覆盖的方法**: 所有公共方法
- **模拟依赖**: SGLang, requests, aiohttp

## 运行测试

### 基本运行

```bash
# 运行所有测试
python run_tests.py

# 或直接使用 pytest
python -m pytest test_http_server_engine.py
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
python run_tests.py -k "test_generate"

# 组合选项
python run_tests.py -v -c --html -k "HttpServerEngineAdapter"
```

### 直接使用 pytest

```bash
# 基本运行
pytest test_http_server_engine.py

# 详细输出
pytest test_http_server_engine.py -v -s

# 带覆盖率
pytest test_http_server_engine.py --cov=verl.workers.rollout.sglang_rollout.http_server_engine --cov-report=term-missing

# 异步模式
pytest test_http_server_engine.py --asyncio-mode=auto

# 运行特定测试类
pytest test_http_server_engine.py::TestHttpServerEngineAdapter

# 运行特定测试方法
pytest test_http_server_engine.py::TestHttpServerEngineAdapter::test_generate
```

## 依赖要求

```bash
# 基本测试依赖
pip install pytest pytest-asyncio

# 覆盖率报告
pip install pytest-cov

# 并行测试（可选）
pip install pytest-xdist

# HTTP 客户端库
pip install requests aiohttp

# 如果需要实际的 torch 支持
pip install torch
```

## 测试设计特点

### 独立性
- 使用 Mock 对象模拟所有外部依赖
- 不需要实际的 SGLang 安装
- 测试之间完全独立

### 异步支持
- 包含完整的异步测试
- 使用 pytest-asyncio 处理异步测试
- 测试异步上下文管理器

### 错误注入
- 模拟各种网络错误
- 测试重试机制
- 验证错误恢复

### 边缘案例
- 极端配置值
- 大数据负载
- Unicode 字符处理
- 并发请求

## 故障排除

### 常见问题

1. **导入错误**
   ```
   ModuleNotFoundError: No module named 'sglang'
   ```
   - 这是正常的，测试使用 Mock 对象，不需要实际安装 SGLang

2. **异步测试失败**
   ```
   RuntimeError: This event loop is already running
   ```
   - 确保使用 `pytest --asyncio-mode=auto`

3. **覆盖率报告问题**
   ```
   Coverage.py warning: No data was collected
   ```
   - 确保模块路径正确
   - 检查是否正确安装了 pytest-cov

### 调试测试

```bash
# 运行单个测试并查看详细输出
pytest test_http_server_engine.py::TestHttpServerEngineAdapter::test_generate -v -s

# 在测试失败时进入调试器
pytest test_http_server_engine.py --pdb

# 显示最慢的测试
pytest test_http_server_engine.py --durations=10
```

## 贡献指南

### 添加新测试

1. 在相应的测试类中添加新方法
2. 使用描述性的测试方法名
3. 包含详细的文档字符串
4. 使用适当的 Mock 对象
5. 验证所有断言

### 测试命名约定

- 测试方法以 `test_` 开头
- 使用描述性名称，如 `test_generate_with_unicode_prompt`
- 测试类以 `Test` 开头
- 边缘案例测试包含 `edge_case` 或具体场景描述

### Mock 使用指南

- 始终 Mock 外部依赖（requests, aiohttp, multiprocessing）
- 使用 `patch` 装饰器进行方法级 Mock
- 为 Mock 对象提供合理的返回值
- 验证 Mock 对象的调用次数和参数

## 性能基准

在标准开发机器上的典型运行时间：

- **所有测试**: ~30-60 秒
- **同步测试**: ~20-30 秒  
- **异步测试**: ~15-25 秒
- **边缘案例测试**: ~10-15 秒

## 持续集成

这些测试设计为在 CI/CD 环境中运行：

```yaml
# 示例 GitHub Actions 配置
- name: Run HTTP Server Engine Tests
  run: |
    cd tests/workers/rollout
    python run_tests.py -v -c
``` 