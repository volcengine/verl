# Quantization Aware Training

## Low Precision Inference

### 背景

RL 训练中 rollout（推理）通常占据大部分时间，而大规模 MoE 模型的 rollout 往往受限于显存容量和访存带宽。使用低精度（FP8 / MXFP8 / MXFP4 / NVFP4）推理可以：

- 降低权重显存占用，留出更多 KV cache，提高并发；
- 降低 decode 阶段的访存量，提高吞吐；
- 在 Hopper/Blackwell 上使用 FP8/FP4 Tensor Core 提升 prefill 算力。

代价是训练（高精度）与推理（低精度）之间的数值差异，即 training-inference mismatch，它会影响 importance sampling 和 off-policy 修正的准确性，严重时会导致训练不稳定。

### 模型权重格式

HuggingFace checkpoint 通常有两种精度格式：

1. **高精度**：权重以 BF16 存储，`config.json` 中没有 `quantization_config`。
2. **低精度**：`config.json` 中有 `quantization_config`，描述各层的量化方式。不同模型对 linear（attention、dense MLP、shared experts）和 MoE（routed experts）往往采用不同的格式。

| 模型 | 结构 | `quant_method` | Linear | MoE (routed experts) | 不量化的层 |
|---|---|---|---|---|---|
| [Qwen/Qwen3.8-2.4T-A95B](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B/blob/main/config.json) | 92 层，512 experts，top-10 | 无 | BF16 | BF16 | - |
| [zai-org/GLM-5.2](https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json) | 78 层，256 experts，top-8，前 3 层 dense | 无 | BF16 | BF16 | - |
| [zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3/blob/main/config.json) | 与 GLM-5.2 相同 | `fp8` | FP8 E4M3，128x128 block，FP32 scale（`weight_scale_inv`），activation 动态量化 | 同 Linear | router `mlp.gate`、`indexers_proj`、`embed_tokens`、`lm_head`、MTP `eh_proj`、各类 norm（`modules_to_not_convert`） |
| [moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json) | 93 层，896 experts，第 1 层 dense | `compressed-tensors` | BF16 | MXFP4 W4A16：FP4 E2M1，1x32 group，UE8M0 scale（`weight_packed` + `weight_scale`），activation 不量化 | `self_attn`、`shared_experts`、dense `mlp`、`lm_head`、`vision_tower`、`mm_projector`（`ignore` 正则） |
| [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/config.json) | 40 层，384 experts，top-6 | `fp8` | FP8 E4M3，32x32 block（`wo_a` 可能是 128x128），UE8M0 scale（`scale_fmt: ue8m0`） | FP4 E2M1，1x32 group，UE8M0 scale（`expert_dtype: fp4`），格式与 MXFP4 相同 | router `ffn.gate`、indexer `wk`/`weights_proj`、`compressor` |

说明：

- GLM-5.3 是标准的 HF FineGrainedFP8 格式（与 DeepSeek-V3 相同），vLLM 和 SGLang 的 `Fp8Config.from_config` 都能直接解析。
- Kimi-K3 使用 compressed-tensors 格式：`config_groups` 以 `Linear` 为 target，再用 `ignore` 排除非 expert 层，从而实现 "linear BF16 + MoE MXFP4"。
- DeepSeek-V4.1-Flash 的 `fp8` 配置带有 `scale_fmt`、`expert_dtype` 等私有字段，需要引擎提供专门的支持（如 vLLM 的 `deepseek_v4_fp8`），不属于通用格式。

### 训练与推理精度

- **rollout 侧**：低精度权重直接使用其自带的 `quantization_config` 推理；高精度权重既可以高精度推理，也可以通过 override `quantization_config` 的方式使用低精度推理，以获取更高的吞吐。
- **trainer 侧**：总是使用高精度训练。低精度权重会被反量化到高精度后再训练，如 DeepSeek-V4.1-Flash、Kimi-K3。为了减少 training-inference mismatch，低精度推理时训练侧可以开启 Quantization Aware Training（QAT）。

### 低精度推理方案：override quantization_config

rollout 统一采用 **override HF `quantization_config` + `load_format=dummy`** 的方式：

1. 引擎启动时用 `dummy` 加载，并把 `quantization_config` override 成目标量化格式。这样引擎会按"已量化 checkpoint"创建参数，例如 FP8 的 `weight` + `weight_scale_inv`、MXFP4 的 `weight_packed` + `weight_scale`。
2. 每次权重同步时，把 BF16 权重量化成 checkpoint 格式（量化可以在 trainer 侧或 rollout worker 侧完成），再交给引擎的 `load_weights`，由引擎完成融合、TP 切分以及 kernel 布局转换。

对于原生低精度的模型（如 GLM-5.3、Kimi-K3），不需要 override，权重同步时按其自带的 `quantization_config` 量化即可。

#### 为什么选择 compressed-tensors

目标是灵活表达 (linear, moe) x (fp8, mxfp8, mxfp4, nvfp4) 的任意组合。各候选格式的能力如下：

| 格式 | 按层混合 | 覆盖的精度 | 备注 |
|---|---|---|---|
| `fp8`（HF FineGrainedFP8） | 否 | FP8 block；SGLang 支持 `quant_method: mxfp8` | verl 现有方案，两个引擎都支持 |
| `modelopt_fp8` / `modelopt_fp4` | 否 | FP8 / NVFP4 | |
| `modelopt_mixed` | 是 | FP8 + NVFP4 | 不支持 MX 格式 |
| `quark` | 是 | MXFP4 + FP8 | 主要面向 AMD |
| **`compressed-tensors`** | **是**（`config_groups`） | **FP8 / MXFP8 / MXFP4 / NVFP4** | Kimi-K3 和 verl QAT 已在使用 |

compressed-tensors 的优势：

- 用 `config_groups` 为不同 target 指定不同方案。target 匹配时先按层名（支持 `re:` 正则）、再按类名，因此 `re:.*mlp\.experts\..*` 会优先于 `Linear` 命中 routed experts。
- 是开放的格式规范（`compressed_tensors.QuantizationArgs`），vLLM 和 SGLang 都支持。
- verl QAT 已经产出 compressed-tensors 格式，NVFP4 W4A4 所需的 `input_global_scale` 由 QAT observer 维护，可以直接复用。

#### 引擎支持矩阵（compressed-tensors）

| 精度 | GPU | vLLM Linear | vLLM MoE | SGLang Linear | SGLang MoE |
|---|---|---|---|---|---|
| FP8（block / channel） | Hopper / Blackwell | ✅ `W8A8Fp8` | ✅ `W8A8Fp8MoE` | ✅ `W8A8Fp8` | ✅ `W8A8Fp8MoE` |
| MXFP8 | Blackwell | ✅ `W8A8Mxfp8` | ✅ `W8A8Mxfp8MoE` | ❌（只能用 `Fp8Config(quant_method="mxfp8")`，整模型统一） | ❌（同左） |
| MXFP4 | Blackwell | ✅ `W4A4Mxfp4` | ✅ `W4A4Mxfp4MoE` | ❌ | ✅ `Mxfp4MoEMethod` |
| NVFP4 | Blackwell | ✅ W4A4 / W4A16 | ✅ W4A4 / W4A16 | ✅ 仅 W4A4 | ✅ W4A4 |

vLLM 可以覆盖全部组合；SGLang 只支持其中一部分，缺失部分需要 SGLang 上游补齐（可参考 vLLM 对应的 scheme 移植）。

硬件方面：MXFP8 / MXFP4 / NVFP4 只在 Blackwell（SM100）上支持。Hopper 上 vLLM 虽然会退化为 Marlin weight-only kernel（激活保持 bf16），但数值与 Blackwell 上的 W8A8 / W4A4 不同，verl 不支持这种用法。

#### 配置

```yaml
actor_rollout_ref:
  rollout:
    quantization:          # linear / moe / config_file 全为 null 表示不做低精度推理
      linear: FP8_BLOCK    # compressed-tensors preset，目前支持 FP8_BLOCK
      moe: FP8_BLOCK
      ignore: []           # 额外跳过的层，支持精确层名或 "re:<regex>"
      config_file: null    # 直接提供 compressed-tensors quantization_config JSON，与 linear/moe/ignore 互斥
```

只支持 compressed-tensors，不再支持 `quantization: fp8` 字符串和 torchao。router（`mlp.gate`）、`lm_head`、embedding、视觉塔和 MTP 默认加入 ignore，形状不能被 block 整除的层（如 Qwen3.5 `in_proj_a/b`）自动加入 ignore。

#### 配置翻译规则

verl 内部（`verl/utils/qat/compressed_tensors.py`）生成 override 用的 compressed-tensors `quantization_config`，trainer 侧量化器使用同一份配置：

1. **linear 和 moe 相同**：一个 `targets: ["Linear"]` 的 config group。
2. **linear 和 moe 不同**：MoE 用 `re:.*mlp\.experts\..*` 单独成组，其余 `Linear` 一组；某一项为 null 时对应层保持高精度。
3. **`config_file`**：JSON 可以是 `quantization_config` 本身，也可以是带 `quantization_config` 字段的 HF `config.json`，原样使用（不追加默认 ignore）。每个 config group 的数值配置必须与某个已支持的 preset 一致，否则报错；QAT mode 也据此校验。

   多精度组合（目前未支持）的配置示例：

   ```json
   {
     "quant_method": "compressed-tensors",
     "format": "mixed-precision",
     "ignore": ["re:.*mlp\\.gate$", "lm_head"],
     "config_groups": {
       "moe": {
         "targets": ["re:.*mlp\\.experts\\..*"],
         "format": "mxfp4-pack-quantized",
         "weights": {"type": "float", "num_bits": 4, "strategy": "group", "group_size": 32,
                     "symmetric": true, "dynamic": false, "scale_dtype": "torch.uint8"},
         "input_activations": null
       },
       "linear": {
         "targets": ["Linear"],
         "weights": {"type": "float", "num_bits": 8, "strategy": "group", "group_size": 32,
                     "symmetric": true, "dynamic": false, "scale_dtype": "torch.uint8"},
         "input_activations": {"type": "float", "num_bits": 8, "strategy": "group", "group_size": 32,
                               "symmetric": true, "dynamic": true, "scale_dtype": "torch.uint8"}
       }
     }
   }
   ```

   上例为 "linear MXFP8 + MoE MXFP4 (W4A16)"，MoE 部分与 Kimi-K3 自带的配置一致。各字段的具体取值需要与所用的 `compressed-tensors` 版本核对。

4. **引擎**：vLLM 和 SGLang（训练侧目前只有 Megatron）支持；TRT-LLM 设置 `rollout.quantization` 会直接报错。SGLang 目前只验证了 Hopper 上的 FP8 block。

#### 权重同步

- 量化器输出 **checkpoint 格式** 的张量：按 expert 拆分、不做 TP 切分，参数名与 checkpoint 一致，例如：
  - FP8：`weight` + `weight_scale_inv`；
  - compressed-tensors FP4：`weight_packed` + `weight_scale`，NVFP4 另有 `weight_global_scale`、`input_global_scale`。
- vLLM 侧**不使用** vLLM 自带的 layerwise reload（`initialize_layerwise_reload` / `finalize_layerwise_reload`）。它会把一个 layer 的所有权重都攒齐之后才开始处理，对大规模 MoE 模型来说，一个 MoE layer 的全部 expert 权重会同时驻留在显存里，容易 OOM。
- vLLM 侧沿用 `verl/utils/vllm/vllm_fp8_utils.py`（`Fp8LinearMethod` / `Fp8MoEMethod`）和 `verl/utils/vllm/vllm_fp4_utils.py`（`Mxfp4MoEMethod`）的 **stage → load → process** 做法，入口在 `verl/utils/vllm/vllm_quant_utils.py`。它要解决的问题是：kernel 后处理会把 weight/scale 改写成推理布局（如 DeepGEMM 把 block scale 打包成 int32 UE8M0，Marlin 对 FP4 权重做 repack，FlashInfer 交换 w13 的 gate/up 并做 tile shuffle），checkpoint 格式的张量无法直接写回。具体分三步：
  1. **记录 checkpoint 布局**：首次加载时，通过 patch `process_weights_after_loading` 记下各参数的 checkpoint shape/dtype（只记布局，不保留张量），或者像 MXFP4 那样直接按 `create_weights` 推导出来。
  2. **stage（权重同步前）**：把活跃参数暂存起来，为每个参数换上一个 checkpoint 布局的 staging buffer，并带上 `weight_loader` 等参数元数据，供 `load_weights` 写入。只要字节数相同，就直接 reinterpret 原有存储（例如 MXFP4 expert 权重在 DeepGEMM 和 Marlin 下都是这样），不额外分配显存；只有元素个数会变化的参数（主要是 block scale，大小约为权重的 1/16）才需要新分配。权重按 bucket 流式写入，不会把整层攒在一起。
  3. **process（全部 bucket 加载完之后）**：重新执行原有的 `process_weights_after_loading`，同时 patch `replace_parameter`，把后处理结果 `copy_` 回原来的活跃参数存储，CUDA graph 捕获的地址保持不变。这一步必须在 `replace_parameter` 调用时就写回，因为 MoE 的 `_setup_kernel` 末尾会用当时 layer 上的张量重建 `moe_quant_config` 和 `moe_kernel`。
- 新增 compressed-tensors scheme（MXFP8、MXFP4、NVFP4 的 linear 和 MoE）时，按同样的模式实现各自的 stage/process：从 `create_weights` 推导出 checkpoint 布局，拦截对应模块里的 `replace_parameter`，并对不能原地写回的后端（如参数不以 Parameter 形式存放的 Triton MXFP4 后端）显式报错。注意 compressed-tensors MXFP4 MoE 会在后处理中把 `w13_weight_packed` 改名为 `w13_weight`，stage 时需要恢复成 checkpoint 中的参数名。
- SGLang 侧沿用 trainer 侧量化后 `update_weights_from_tensor` 的方式，它只调用 `load_weights`，不会重新执行 `process_weights_after_loading`。Hopper 上 compressed-tensors FP8 block 的 linear 和 Triton MoE 后处理不改写 weight/scale 布局，因此可以直接写回，不需要 patch。Blackwell 上 DeepGEMM 会把 block scale 改写成 UE8M0、FlashInfer TRTLLM MoE 会交换 w13，届时需要像 vLLM 一样实现 stage/process。

#### 注意事项

1. **NVFP4 W4A4 需要静态的 `input_global_scale`**：使用 `dummy` 加载时没有校准数据。可以使用 W4A16（仅 vLLM 支持），或开启 QAT 由 observer 维护该 scale，或额外进行一次校准。MX 格式的 activation 为动态量化，不需要校准。
2. **数值格式必须逐 bit 对齐**：UE8M0 scale 的编码、FP4 nibble 的打包顺序、NVFP4 两级 scale 的约定，都必须与引擎期望的一致。建议为每种格式编写往返测试：verl 量化，引擎反量化，再与 BF16 对比。
3. **activation 精度由 scheme 决定**：例如 compressed-tensors 的 MXFP4 linear 是 W4A4，而 Kimi-K3 的 MoE 是 W4A16（`input_activations: null`）。配置中的 `activation` 字段需要准确映射到 `input_activations`。

## Quantization Aware Training

QAT 在训练时对权重（W4A4 时还包括 activation）做 fake quantization：前向使用"量化后再反量化"的值，反向通过 STE 直接回传梯度，使模型学会容忍低精度推理带来的误差，从而缩小 training-inference mismatch。权重同步时，trainer 把 BF16 权重真正量化成 checkpoint 格式，再发给 rollout。

QAT 由两部分组成：

1. **训练侧 fake quantization**：与训练后端强相关，需要把量化器插入后端实际使用的 Linear 层。
2. **导出格式**：权重同步时输出的张量名和 scale 约定，它决定 rollout 侧使用哪个 `quantization_config`。

这两部分是相互独立的。训练侧可以按后端选择不同的实现，导出格式则应该统一为 compressed-tensors，与上文的 low precision inference 方案保持一致。

### FSDP/VeOmni

- **fake quantization**：使用 verl 自己实现的 `QATLinear`（`verl/utils/qat/linear.py`），替换模型中的 `nn.Linear`。W4A4 模式下，activation 的 `input_global_scale` 由 observer 在训练中维护。
- **导出**：`verl/utils/qat/quantizer.py` 输出 compressed-tensors NVFP4 格式，即 `weight_packed`、`weight_scale`、`weight_global_scale`，W4A4 时还有 `input_global_scale`。
- **global scale 融合**：vLLM 会把 q/k/v、gate/up 融合成一个层，融合后的层只能有一个 global scale，因此导出时需要用 `fuse_global_scales` 对齐各分片的 global scale（取 min）。
- **rollout 配置**：`recipe/qat/config/nvfp4_w4a16.json`、`nvfp4_w4a4.json`（`quant_method: compressed-tensors`）。

### Megatron-LM

Megatron-LM + TransformerEngine + ModelOpt

- **fake quantization**：Megatron 中实际计算的是 TE 实现的层（带 TP 切分的 `ColumnParallelLinear` / `RowParallelLinear`，以及 expert 使用的 `TEGroupedLinear`），无法沿用 FSDP 替换 `nn.Linear` 的方式。Megatron-Bridge 官方的做法是使用 `modelopt_transformer_layer_spec` 构建模型，再调用 `mtq.quantize` 插入 ModelOpt 量化器（`verl/utils/modelopt/quantize.py`）。因此训练侧继续使用 ModelOpt。
- **导出（现状）**：`verl/utils/modelopt/qat_weight_exporter.py` 输出 ModelOpt checkpoint 格式，rollout 使用 `quant_method: modelopt`（`recipe/qat/config/nvfp4_w4a16_megatron.json`），vLLM 侧需要单独维护一套 patch（`verl/utils/modelopt/vllm_modelopt_patch.py`）。

### 两个后端的导出差异

两个后端导出的都是 NVFP4，量化方式完全相同：FP4 E2M1，每 16 个元素一个 FP8 E4M3 block scale，外加一个 FP32 global scale。区别只在参数名和 global scale 的表示方式：

| | FSDP/VeOmni（compressed-tensors） | Megatron（ModelOpt） |
|---|---|---|
| FP4 打包后的权重 | `weight_packed` | `weight` |
| block scale（FP8 E4M3，每 16 个元素一个） | `weight_scale` | `weight_scale` |
| weight global scale | `weight_global_scale` = `6 * 448 / amax` | `weight_scale_2` = `amax / (6 * 448)` |
| activation global scale（W4A4） | `input_global_scale` | `input_scale`（同为倒数关系） |
| `ignore_patterns` 语法 | `re:` 正则，匹配 HF 层名 | fnmatch，匹配 Megatron 模块名，并手写映射（`lm_head` 对应 `*output_layer*`，`*mlp.gate` 对应 `*router*`） |
| vLLM patch | `verl/utils/qat/vllm_patch.py` | `verl/utils/modelopt/vllm_modelopt_patch.py` |

可以看出，`weight_global_scale` 与 `weight_scale_2` 互为倒数，`input_global_scale` 与 `input_scale` 同理。

### 统一方案：训练侧保留 ModelOpt，导出改为 compressed-tensors

在 `QATWeightExporter._quantize_nvfp4` 中改为输出 compressed-tensors 格式：

```python
yield (name.replace(".weight", ".weight_packed"), quantized)
yield (_derive_scale_name(name, "weight_scale"), w_scale)
yield (_derive_scale_name(name, "weight_global_scale"), 1.0 / w_scale_2)
if input_scale is not None:
    yield (_derive_scale_name(name, "input_global_scale"), 1.0 / input_scale)
```

这样改之后：

- **rollout 配置统一**：两个后端共用 `nvfp4_w4a16.json` / `nvfp4_w4a4.json`，`nvfp4_w4a16_megatron.json` 可以删除。rollout 侧不需要关心训练后端，与上文 "override quantization_config" 的方案一致。
- **少维护一套 patch**：`vllm_modelopt_patch.py` 可以下线。`qat/vllm_patch.py` 也可以改写成与 `vllm_fp8_utils.py`、`vllm_fp4_utils.py` 一致的 stage/process 模式，统一维护。
- **global scale 天然一致**：Megatron 的 `linear_qkv`、`linear_fc1` 本身就是融合层，只有一个量化器和一个 amax，导出后 q/k/v、gate/up 的 global scale 自然相同，不需要额外的 `fuse_global_scales`。MoE 的 `w1`/`w3` 同理，这也满足 vLLM compressed-tensors NVFP4 MoE 要求 `w13` 两个 global scale 相同的约束。
- **ignore 统一用 HF 层名和 `re:` 语法**：Megatron 侧通过 bridge 的 `mapping_registry` 把 HF 层名翻译到 Megatron 模块名（可复用 `_iter_hf_to_megatron_matches`），不再需要手写映射。

### 待验证

1. **FP4 打包逐 bit 一致**：ModelOpt 的 `to_quantized_weight` 与 compressed-tensors 的 `compress_weight` 的 nibble 打包顺序、block scale 排布是否一致。可以对同一个 BF16 权重分别过两条导出路径，比较输出张量或 vLLM 反量化后的结果。
2. **activation scale 的倒数关系**：确认 `NVFP4QTensor.get_activation_scaling_factor` 的定义为 `amax / (6 * 448)`，这决定了 W4A4 时 `input_global_scale` 取倒数是否正确。Megatron 目前仅支持 W4A16（`build_quantize_config` 中写死），暂不阻塞。
3. **MX 格式 fake quantization**：若 Megatron QAT 要扩展到 MXFP8 / MXFP4，需要确认 ModelOpt 是否提供对应的量化配置。导出侧同样按 compressed-tensors 的 MX scheme 输出即可。
