# Micro-batch 级 Profiler Schedule

Last updated: 10/16/2026.

> 状态：已实现首期代码，已通过 CPU 控制流和原生 Torch CPU trace 测试；GPU/NPU 分布式实机待验证。
>
> 首期接入 FSDP、AutoModel、TorchTitan、VeOmni。Megatron/MindSpeed 暂不支持
> micro-batch schedule，显式启用时会报错。本文末尾记录本次全部改动与验证范围。

## 背景

在长序列训练场景中，单个训练 step 的前向、反向计算量和算子数量都较大。如果对完整
step 进行 profiling，生成的 trace 或数据库文件可能过大，导致以下问题：

- profiling 文件写入时间和磁盘占用显著增加；
- 数据解析和可视化工具加载变慢，甚至无法打开；
- 实际排查通常只需要观察少量稳定迭代，而不需要采集完整训练过程。

改动前的实现支持在 Actor/Critic update 中按 mini-batch 推进 profiler schedule。它可以跳过
一部分 mini-batch，只采集指定范围，因此能够减小 profiling 文件。但在长序列场景下，
一个 mini-batch 内部还会拆分为多个用于梯度累积的 micro-batch。即使只采集一个
mini-batch，文件仍可能包含其全部 micro-batch，数据量依然较大。

本特性将 schedule 的采集粒度从 mini-batch 下沉到 micro-batch，使用户能够只采集
指定训练 step 中的少量 micro-batch。

训练层级如下：

```text
global training step
  -> update epoch
    -> mini-batch
      -> micro-batch（新的 profiler schedule 单位）
```

## 目标

- 通过 `global_profiler.steps` 选择需要 profiling 的训练 step。
- 在选中的训练 step 内，以 micro-batch 为单位推进 profiler schedule。
- 支持跳过启动阶段的若干训练 step，再采集稳定阶段的指定 micro-batch。
- Torch profiler 和 Ascend NPU profiler 使用一致的 schedule 语义。
- 未启用 schedule 时保持当前连续采集行为不变。
- 首期聚焦 Actor 和 Critic 的训练更新过程。

## 改动前的能力

改动前的实现已经具备两层选择能力：

1. `global_profiler.steps` 选择 global training step。
2. `schedule` 使用 `skip_first`、`wait`、`warmup`、`active` 和 `repeat` 选择该
   profiling 生命周期内的 mini-batch。

默认 `profile_continuous_steps: false`。在这种模式下，每个选中的 global step 都会单独
启动和停止 profiler，因此 schedule 会在每个 global step 开始时重置。由此可以通过
组合配置采集“指定训练 step 的指定 mini-batch”。

改动前的方式存在以下限制：

- schedule 使用相对计数，不是显式的 `(global_step, mini_batch_index)` 映射；
- 多个 global step 共用同一套 schedule，不能为每个 step 配置不同的 batch 索引；
- 当 `profile_continuous_steps: true` 时，profiler 不会在连续训练 step 之间重启，
  schedule 会跨 global step 连续计数；
- 当前 schedule 单位是 mini-batch，无法进一步选择其中的某个 micro-batch。

## 期望使用方式

下面的配置跳过训练启动阶段，只在 global step 10 启动 profiling，并采集 Actor update
中的一个 micro-batch：

```yaml
global_profiler:
  steps: [10]
  profile_continuous_steps: false

actor_rollout_ref:
  actor:
    profiler:
      enable: true
      tool: npu
      tool_config:
        npu:
          discrete: true
          schedule:
            skip_first: 3
            wait: 0
            warmup: 1
            active: 1
            repeat: 1
```

该配置的预期语义是：

- global step 1 到 9 不启动 profiler；
- global step 10 启动一个新的 profiler；
- 跳过该 profiling 范围内前 3 个训练 micro-batch；
- 使用第 4 个 micro-batch 进行 profiler warmup；
- 实际记录第 5 个 micro-batch；
- 完成一个 active 周期后不再记录其他 micro-batch。

第一个 active micro-batch 的零基索引为：

```text
skip_first + wait + warmup
```

如果不需要 profiler warmup，将 `warmup` 设为 `0` 即可直接选择对应索引。这里的
`warmup` 是 profiler schedule 的预热，不等同于跳过训练启动阶段；后者应由
`global_profiler.steps` 控制。

## 设计方案

`TrainingWorker` 持有 `DistProfiler`，各训练 Engine 持有实际的 micro-batch 循环。
Worker 通过 `BaseEngine.set_micro_batch_callback()` 注册一个通用的完成回调。
仅当训练 Worker 启用 Torch/NPU profiler 且 `schedule.active > 0` 时注册
`self.profiler.step`；其他情况保持默认空回调。Engine 不依赖 profiler：

```text
Trainer
  -> Worker.start_profile(global_step)
  -> TrainingWorker.train_mini_batch()
    -> Engine.train_batch()
      -> Engine.forward_backward_batch()
        -> 完成一个 micro-batch 的 forward + backward
        -> on_micro_batch_end()
          -> DistProfiler.step()
  -> Worker.stop_profile()
```

回调触发语义为：

- 每个训练 micro-batch 成功完成后调用一次；
- 在下一个 micro-batch 开始前调用；
- 所有参与训练的 rank 调用次数一致；
- 仅用于训练路径，即 `forward_only=False`；
- micro-batch 执行失败时不推进 schedule。

已删除 `TrainingWorker.train_mini_batch()` 中的 mini-batch 级 `self.profiler.step()`。
SFT 同样复用 TrainingWorker，因此也删除了其训练循环中的 `step_profile()`，
由 Engine 统一计数，避免重复推进。SFT 仍使用原有的 start/end profiling step 配置。

`BaseEngine.supports_micro_batch_callback` 默认是 `False`。显式循环后端声明支持；
VeOmni 从 FSDP 继承该标记。不支持回调的 Engine 在训练 Worker 显式启用 schedule 时
抛出 `NotImplementedError`，提示将 `schedule.active` 设为 `0` 使用连续采集。
校验不依赖 rank 过滤，避免只有部分 rank 报错；禁用 profiler、关闭 schedule、
forward-only Worker 不触发此限制。

### 采集范围与边界

- `discrete: true`：每个最外层被标记的调用新建 profiler。Actor 使用
  `update_actor` 范围；Critic 在启用 schedule 时使用整个 `train_mini_batch` 范围，
  使计数跨越其所有 mini-batch 和 update epoch。嵌套 `train_batch` 只添加标记，
  不新建 profiler、不重置 schedule。直接调用 `train_batch` 时，其自身构成采集范围。
- `discrete: false`：从 Worker `start_profile()` 到 `stop_profile()` 共用一个
  profiler。`profile_continuous_steps: false` 时每个选中的 global step 重建，
  设为 `true` 时可跨相邻选中 step 继续计数；discrete 模式仍按调用范围重建。
- 回调在 backward 和该 micro-batch 的上下文退出后执行。FSDP 的普通/scaler backward、
  TorchTitan 的 train context、VeOmni 的 activation offload/router replay 保持原有顺序。
- 不新增 barrier 或设备同步。沿用各 Engine 的
  `prepare_micro_batches(..., same_micro_num_in_dp=True)` 和现有数据分发规则，
  在正常完成训练时各 rank 经过相同数量的逻辑回调；实际 backend 是否执行由
  `DistProfiler` 的 enable/rank gate 决定。这不代表发生分布式故障时还能保证计数一致。
- schedule 定义的是相邻 `step()` 之间的时间窗口，不能保证 trace 只含 forward/backward。
  窗口中可能包含数据准备、mini-batch 间的 optimizer 等操作；
  non-discrete 模式还可能包含当前 active 窗口中的推理。推理本身不增加 micro-batch 计数。
  动态 batching 时索引指 Engine 拆分/重排后的实际 micro-batch 顺序。

## 后端接入

FSDP、AutoModel、TorchTitan 和 VeOmni 在 `forward_backward_batch()` 中都有显式的
micro-batch 循环，已在每个成功完成的训练 micro-batch 末尾接入回调。

Megatron 将 micro-batch 执行交给 pipeline schedule。Pipeline parallel 下多个
micro-batch 的 forward/backward 会发生重叠，因此不能简单地在每次 `forward_step()`
后调用 profiler step。需要找到所有 rank 都能观察到的稳定逻辑边界，并单独验证其
schedule 语义。

| Backend | 本次实现 | 说明 |
| --- | --- | --- |
| FSDP | 已接入 | 普通与 scaler backward 共用完成回调，覆盖 LM/Value head |
| AutoModel | 已接入 | 保留梯度累积与 final backward 处理 |
| TorchTitan | 已接入 | 在 train context 退出后通知；不扩展原有 PP 支持范围 |
| VeOmni | 已接入 | 保留 activation offload 上下文与 router replay 清理 |
| Megatron / MindSpeed | 暂不支持 schedule | DP、PP、VPP 均不声明支持；`active: 0` 连续采集仍可用 |

## 兼容性

- 默认配置使用 `active: 0`，已有任务继续在 profiler start/stop 范围内连续采集。
- 已经启用 schedule 的任务会发生语义变化：计数单位从 mini-batch 变为 micro-batch。
- Actor、Critic 和 `global_profiler.steps` 的配置路径保持不变。
- 不新增 `step_unit` 或索引配置项。已有 schedule 参数需按 micro-batch 数量重新设置。
- Torch discrete 模式现在也传入 schedule，并向内层 Worker 暴露同一个活动 profiler。
- Torch/NPU 在未配置 schedule 或 `active: 0` 时，训练完成回调不驱动 backend `step()`。
  Torch 不再在 `stop()` 前额外调用 `step()`，由原生 `stop()` 刷出最后一个窗口。
- 新增的 `train_mini_batch` 标记使用 `schedule_only=True`，未启用 schedule 时
  不增加采集范围，也不改变 Nsight、memory profiler 等工具的范围。
- Torch/NPU 的离散采集在训练异常时仍执行 stop/清理；异常向上传播，不重跑训练函数。
- 建议 micro-batch schedule 使用 `profile_continuous_steps: false`，保证每个选中
  global step 都有独立且从零开始的 schedule。

## 验证目标

- 验证每个成功完成的训练 micro-batch 恰好调用一次 `profiler.step()`。
- 验证原 mini-batch 层不再调用 `profiler.step()`。
- 验证指定 global step 之外不会启动 profiler。
- 验证 schedule 在非连续采集模式下按 global step 重置。
- 验证缺少 schedule 或 `active == 0` 时不会驱动 backend `step()`。
- 验证 forward-only、推理和 rollout 路径不会推进训练 micro-batch schedule。
- 验证各分布式 rank 的 step 次数一致。
- Megatron 需要分别验证 DP、PP 和 VPP 场景。

## 本次取舍

1. 直接将 schedule 单位替换为 micro-batch，不保留双粒度开关，以减少配置与分支。
2. 不新增 `micro_batches: [4]`，继续用现有 schedule 参数选取窗口。
3. 所有选中 global step 共用同一套 schedule，不增加逐 step 索引映射。
4. Megatron/MindSpeed 延后，避免把 pipeline forward 或某个 rank 的局部事件当成完整
   micro-batch 边界，也不回退到容易产生歧义的 mini-batch 计数。
5. 不增加逐 micro-batch 的索引标记；保留原生 profiler step 与现有阶段标记。

## 全部改动记录

路径均相对仓库根目录。

| 文件 | 改动 |
| --- | --- |
| `verl/workers/engine/base.py` | 新增能力标记、可注册/清除的完成回调和默认空操作；无须子类调用基类构造函数 |
| `verl/workers/engine_workers.py` | 按启用状态注册回调；校验不支持的后端；删除 mini-batch step；为 scheduled Critic update 增加外层采集范围；更新注释 |
| `verl/workers/engine/fsdp/transformer_impl.py` | 声明支持；成功的训练 micro-batch 末尾通知 |
| `verl/workers/engine/automodel/transformer_impl.py` | 声明支持；成功的训练 micro-batch 末尾通知 |
| `verl/workers/engine/torchtitan/transformer_impl.py` | 声明支持；成功的训练 micro-batch 末尾通知 |
| `verl/workers/engine/veomni/transformer_impl.py` | 在原循环内通知，保留 router replay 的 finally 清理 |
| `verl/trainer/sft_trainer.py` | 删除与共用 Engine 回调重复的训练 step 计数 |
| `verl/utils/profiler/torch_profile.py` | 未启用 schedule 时 step 为空操作；discrete 模式接入 schedule/共享句柄；嵌套范围复用；异常清理；停止时不补计数 |
| `verl/utils/profiler/mstx_profile.py` | 嵌套 discrete 范围复用；异常时结束标记并 stop/清理；更新计数单位注释 |
| `verl/utils/profiler/profile.py` | 更新 step 接口说明；增加 `schedule_only` 注解选项 |
| `verl/utils/profiler/config.py` | 说明 schedule 以训练 micro-batch 为单位，字段与默认值不变 |
| `verl/trainer/config/actor/actor.yaml` | 更新 Torch/NPU schedule 注释 |
| `verl/trainer/config/critic/critic.yaml` | 更新 Torch/NPU schedule 注释 |
| `verl/trainer/config/profiler/profiler.yaml` | 更新共用 Torch/NPU schedule 注释 |
| `verl/trainer/config/ref/ref.yaml` | 说明 reference inference 不推进训练 schedule |
| `tests/utils/test_torch_profile.py` | 更新关闭 schedule 时的预期；为 step 委派用例显式启用 schedule |
| `tests/workers/test_micro_batch_profiler_schedule_on_cpu.py` | 新增可隔离运行的控制流回归与可选原生 Torch CPU trace 测试 |
| `docs/perf/torch_profiling.md` | 更新使用说明、global step 示例、后端限制和 schedule 计数范围 |
| `docs/advance/micro_batch_profiler_schedule.md` | 将讨论稿更新为实施说明，记录取舍、全部文件改动和验证结果 |

## 本地验证结果与限制

本地为 macOS/CPU，使用工作区内由 uv 创建的 Python 3.12 环境，
PyTorch 2.14.1、TensorDict 0.15.0、pytest 9.1.1、Ruff 0.12.2。
Engine 隔离测试通过 AST 编译**实际源码中的类/方法**，替换框架导入、框架装饰器
与外部训练依赖，不复制训练循环实现。它验证控制流，不能替代真实 TensorDict、
autograd 和分布式集成测试；另有原生 Torch CPU profiler 测试验证采集窗口。

```bash
# 在仓库根目录执行；本次环境位于工作区上一级 .venv
../.venv/bin/python -m pytest -q \
  tests/utils/test_torch_profile.py \
  tests/workers/test_micro_batch_profiler_schedule_on_cpu.py
../.venv/bin/python tests/special_sanity/check_docs_time_info.py
../.venv/bin/python tests/special_sanity/validate_structure.py
git diff --check
```

- 最终结果：**30 passed, 42 subtests passed**，无跳过。
  其中现有 Torch profiler 测试 21 个、新增测试 9 个。
- 13 个新增/修改的 Python 文件均通过编译、Ruff check 和 Ruff format --check；
  `git diff --check`、文档日期和测试目录结构检查通过。
  四个 YAML 文件去除注释/空行后的配置内容与改动前一致，无需更新生成配置。
- 控制流用例覆盖：四后端成功/forward-only/forward 失败/backward 失败，
  FSDP scaler 分支、VeOmni replay 清理、回调清除与实例隔离、两个模拟 rank、
  跨 mini-batch/epoch 恰好计数、未支持后端的启用校验、Torch/NPU nested
  discrete 范围、异常后再次启动、缺省/关闭 schedule、stop 不额外推进、
  内层 Worker 未独立 start 时的 rank gate，以及关闭 schedule 时不增加采集范围。
- 原生 Torch CPU trace 用例使用文档中的 `skip_first: 3, warmup: 1, active: 1,
  repeat: 1`，检查两次采集都只记录 `micro_4`，覆盖 discrete/non-discrete；
  也验证 `active: 0` 时完整记录所有 micro-batch，stop 无须额外 step 即可触发 trace 回调。
  测试在缺少 PyTorch 的环境中会显式跳过，本次已实际执行。
- NPU 专用的现有测试未执行。NPU wrapper 已由隔离测试覆盖，但 NPU trace、
  真实多 rank/动态 batching、PPO Trainer 的
  `global_profiler.steps` 调度与 global step 重置仍需实机验证，不能由模拟测试证明。
- 测试工具和 Python 环境仅位于工作区 `.tools/`、`.venv/`，不属于仓库代码变更。
  自动审批曾拒绝停止安装进程并重新安装依赖的操作，理由是未明确授权该操作；
  最初启动的依赖安装随后自行完成，本次验证使用该环境，没有执行被拒绝的操作。

实机建议：分别用 Actor/Critic、静态/动态 micro-batch 跑 `steps: [10, 12]`，
确认其他 global step 不启动采集，两个选中 step 都从零计数且仅记录第 5 个
micro-batch；再用相邻 steps 验证 continuous 开关，检查所有参与 rank 的回调次数。
Megatron DP/PP/VPP 的 micro-batch 边界验证留待其支持实现时完成。
