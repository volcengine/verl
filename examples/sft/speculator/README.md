# Speculator Design Doc

投机解码（Speculator）相关实现的设计与职责划分如下。

## 设计目标

- 解耦：Trainer/Engine 不依赖具体 speculator 架构细节。
- 可扩展：新增 speculator 类型仅需新增 adapter + 模型实现。
- 兼容性：支持 no-padding（NestedTensor）数据路径。
- checkpoint 可控：speculator 权重/配置与 base model 分离保存。

## 模块与职责

### 1) 模型层（Speculator Models）
路径：`verl/models/speculator/`
- `speculator.py`: `ArcticLSTMSpeculator`、`LayerNormParameterized`、`create_speculator_from_config`。
- `mlp_speculator.py`: `MLPSpeculator` 与 `MLPSpeculatorConfig`。

职责：
- 只负责 `nn.Module` 的参数定义与前向逻辑（`reset_parameters()` / `forward()`）。
- 不处理分布式/FSDP、checkpoint、DTensor 细节。

### 2) 适配层（Speculator Adapters）
路径：`verl/trainer/speculators/`
- `interface.py`: `SpeculatorAdapter` 抽象接口 + 通用实现（checkpoint、公共处理逻辑）。
- `lstm_adapter.py`: `LSTMSpeculatorAdapter`（LSTM speculator 构建与 loss 计算）。
- `mlp_adapter.py`: `MLPSpeculatorAdapter`（MLP speculator 构建与 loss 计算）。

职责：
- 构建/挂载 speculator（`build_and_attach`）。
- 计算 speculator loss（`compute_speculator_loss`）。
- 输入对齐/裁剪与 NestedTensor 处理。
- 统一 checkpoint 保存/加载（通过基类默认实现）。

公共逻辑（基类提供）：
- `_maybe_pad_nested`: NestedTensor 转 padded Tensor。
- `_slice_speculator_inputs`: 隐状态与 input_ids 对齐切片。
- 默认 `save_checkpoint` / `load_checkpoint`（调用 FSDP checkpoint manager）。

Adapter 选择方式：
- `build_speculator_adapter` 读取 `model.speculator_adapter`。
- 支持字符串 FQN，或字典 `{fqn: ...}` / `{path: ..., name: ...}`。

LSTM/MLP 专用要点：
- 两个 adapter 都支持 `freeze_base_model`，为 true 时冻结 base model，仅训练 speculator。
- `LSTMSpeculatorAdapter.compute_speculator_logits` 处理 DTensor/local Tensor 转换，避免 placement 不一致。
- `_get_speculator_config_obj` 返回 `ArcticLSTMSpeculatorConfig` / `MLPSpeculatorConfig`，保存到 `speculator/config.json`。

### 3) Trainer 入口
路径：`verl/trainer/fsdp_sft_trainer.py`

职责：
- 根据配置构建 `SpeculatorAdapter`（`model.speculator` 或 `model.speculator_adapter` 任一存在即构建）。
- 构建模型、FSDP 包裹、并在 FSDP 后挂载 speculator。
- 决定优化器参数（可只训练 speculator）。
- 将 speculator loss 接入训练流程。

关键点：
- `freeze_base_model=True` 时，base loss 不参与梯度计算，仅优化 speculator。
- speculator 在 FSDP 包裹之后挂载，避免 DTensor/Tensor 混用。
- speculator checkpoint 由 adapter 统一保存/加载。

### 4) Engine 计算路径
路径：`verl/workers/engine/fsdp/transformer_impl.py`

职责：
- `FSDPEngineWithLMHeadAndSpeculator` 在 forward 中获取 hidden states。
- 调用 `SpeculatorAdapter.compute_speculator_loss` 计算 speculator loss。

关键点：
- 训练要求 `use_remove_padding=False`（否则抛出 NotImplementedError）。
- `prepare_model_inputs` 强制 `output_hidden_states=True` 以便获取隐状态。
- speculator 在 FSDP 包裹之后挂载。

### 5) Checkpoint 统一处理
路径：`verl/utils/checkpoint/fsdp_checkpoint_manager.py`

职责：
- 提供 `save_speculator_checkpoint` / `load_speculator_checkpoint`。
- 处理 FSDP1/FSDP2 差异（full state / sharded state）。
- speculator 参数保存在 `checkpoint/speculator/pytorch_model.bin`。
- config 通过 `SpeculatorConfigBase.save()` 写入 `checkpoint/speculator/config.json`。

SpeculatorAdapter 默认实现直接调用上述通用函数。

## 类图（简化）

```text
+----------------------------+         +------------------------------------+
|        FSDPSFTTrainer       |         |  FSDPEngineWithLMHeadAndSpeculator |
| (trainer/fsdp_sft_trainer)  |         | (engine/fsdp/transformer_impl.py) |
+-------------+---------------+         +------------------+-----------------+
              | uses                                 | uses
              v                                      v
        +-----+----------------------+
        |     SpeculatorAdapter     |
        | (trainer/speculators/...) |
        +-----+----------------------+
              ^                    ^
              | inherits           | inherits
   +----------+-----------+   +----+-------------------+
   | LSTMSpeculatorAdapter|   | MLPSpeculatorAdapter  |
   | (lstm_adapter.py)    |   | (mlp_adapter.py)      |
   +----------+-----------+   +----+-------------------+
              | uses               | uses
              v                    v
   +------------------------+   +------------------------+
   | ArcticLSTMSpeculator   |   | MLPSpeculator         |
   | (models/speculator/...)|   | (models/speculator/...)|
   +------------------------+   +------------------------+
```

## 数据流与对齐逻辑

- base model forward 得到 `hidden_states`。
- speculator 预测位置对齐规则：
  - `hidden_states[t]` 对齐预测 `input_ids[t+1]`。
  - 需要 `n_predict` 个右侧 token，因此裁掉末尾 `n_predict + 1` 的 hidden。
- `_slice_speculator_inputs` 返回 `hidden = hidden_states[:, :-(n_predict+1), :]` 与 `seq_ids = input_ids[:, 1:]`。
- adapter 组装 `spec_inds = concat(seq_ids, pad_ids)`，并调用 `speculator(hidden, spec_inds)`。
- `compute_speculator_loss` 按 head 计算交叉熵：第 `i` 个 head 使用 `start=i+2` 的目标片段。

## NestedTensor / no-padding

- no-padding 路径会生成 NestedTensor。
- NestedTensor 不支持常规切片，adapter 里通过 `_maybe_pad_nested` 转换为 padded Tensor。
- `compute_speculator_logits` 也会将 NestedTensor 转为 padded Tensor 再进行切片与拼接。


## 配置入口

- `model.speculator`: speculator 超参（`n_predict`, `inner_dim`, `emb_dim`, `proj_dim`, `method`, `tie_weights`, `scale_input` 等）。
- `model.speculator_adapter`: adapter 类路径（FQN）或 `{path, name}`。
- `model.freeze_base_model`: 冻结 base model 参数，仅训练 speculator。
- `model.use_remove_padding`: speculator 训练要求 false。

## 扩展新 Speculator 的流程

1. 在 `verl/models/speculator/` 实现新模型（仅 forward）。
2. 在 `verl/trainer/speculators/` 新建 adapter：实现构建与 loss 逻辑。
3. 配置 `model.speculator_adapter.fqn` 指向新 adapter。
4. 如需分片，在 FSDP wrap policy 中加入新模型类名。
