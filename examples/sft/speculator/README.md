# Speculator Design Doc

投机解码（Speculator）相关实现的设计与职责划分如下

## 设计目标

- 解耦：Trainer/Engine 不依赖具体 speculator 架构细节。
- 可扩展：新增 speculator 类型仅需新增 adapter + 模型实现。
- 兼容性：支持 FSDP/DTensor 与 no-padding（NestedTensor）数据路径。
- checkpoint 可控：speculator 权重/配置与 base model 分离保存。

## 模块与职责

### 1) 模型层（Speculator Models）
路径：`verl/models/speculator/`
- `speculator.py`: `ArcticLSTMSpeculator` 实现。
- `mlp_speculator.py`: `MLPSpeculator` 实现。

职责：
- 只负责 `nn.Module` 前向逻辑与参数定义。
- 不处理分布式/DTensor 细节，不处理 checkpoint 细节。

关键方法（与代码一致）：
- `reset_parameters()`
- `forward(state, inds)`

### 2) 适配层（Speculator Adapters）
路径：`verl/trainer/speculators/`
- `interface.py`: `SpeculatorAdapter` 抽象接口 + 默认实现（checkpoint、公共处理逻辑）。
- `lstm_adapter.py`: `LSTMSpeculatorAdapter`（LSTM 特化构建、config 生成）。
- `mlp_adapter.py`: `MLPSpeculatorAdapter`（MLP 特化构建）。

职责：
- 构建 speculator（`build_and_attach`）。
- 计算 speculator loss（`compute_speculator_loss`）。
- 输入对齐/裁剪与 NestedTensor 处理。
- 统一 checkpoint 保存/加载（通过基类默认实现）。

公共逻辑（已上收至基类）：
- `_maybe_pad_nested`: NestedTensor 转 padded Tensor。
- `_slice_speculator_inputs`: hidden 和 input_ids 的对齐切片逻辑。
- 默认 `save_checkpoint` / `load_checkpoint`。

LSTM 特化：
- `_get_speculator_config_obj`: 生成 `ArcticLSTMSpeculatorConfig` 保存到 `speculator/config.json`。

### 3) Trainer 入口
路径：`verl/trainer/fsdp_sft_trainer.py`

职责：
- 根据配置构建 `SpeculatorAdapter`（通过 `build_speculator_adapter`）。
- 构建模型与优化器参数（可只训练 speculator）。
- 将 speculator 训练逻辑接入训练流程。

关键点：
- Adapter 由 `model.speculator_adapter` 配置驱动。
- `model.freeze_base_model` 时优化器只包含 speculator 参数。

### 4) Engine 计算路径
路径：`verl/workers/engine/fsdp/transformer_impl.py`

职责：
- `FSDPEngineWithLMHeadAndSpeculator` 在 forward 中获取 hidden states。
- 调用 `SpeculatorAdapter.compute_speculator_loss` 计算 speculator loss。

关键点：
- speculator 训练要求 `use_remove_padding=False`。
- hidden states 由 base model forward 返回。

### 5) Checkpoint 统一处理
路径：`verl/utils/checkpoint/fsdp_checkpoint_manager.py`

职责：
- 提供通用 `save_speculator_checkpoint` / `load_speculator_checkpoint`。
- 处理 FSDP1/FSDP2 差异，兼容 shard/full state。

SpeculatorAdapter 默认实现直接调用上述通用函数。

## 数据流与对齐逻辑

- base model forward 得到 `hidden_states`。
- speculator 预测位置对齐规则：
  - `hidden_states[t]` 对齐预测 `input_ids[t+1]`。
  - 需要 `n_predict` 个右侧 token，因此裁掉末尾 `n_predict + 1` 的 hidden。
- `_slice_speculator_inputs` 统一完成上述裁剪与偏移。

## NestedTensor / no-padding

- no-padding 路径会生成 NestedTensor。
- NestedTensor 不支持常规切片，adapter 里通过 `_maybe_pad_nested` 转换为 padded Tensor。

## FSDP / DTensor 说明

- 当前设计倾向不将 speculator 包裹进 FSDP 以减少 DTensor/Tensor 混用问题。
- 若 speculator 被 FSDP 分片，则输入也必须转换为 DTensor（否则 embedding/linear 报错）。
- 包裹策略入口：`verl/utils/fsdp_utils.py`。

## 配置入口

- `model.speculator`: speculator 超参（n_predict, inner_dim, emb_dim, proj_dim, method 等）。
- `model.speculator_adapter`: adapter 类路径。
- `model.freeze_base_model`: 冻结 base model 参数。
- `model.use_remove_padding`: speculator 训练要求 false。

## 扩展新 Speculator 的流程

1. 在 `verl/models/speculator/` 实现新模型（仅 forward）。
2. 在 `verl/trainer/speculators/` 新建 adapter：实现构建与 loss 逻辑。
3. 配置 `model.speculator_adapter.fqn` 指向新 adapter。
4. 如需分片，在 FSDP wrap policy 中加入新模型类名。

