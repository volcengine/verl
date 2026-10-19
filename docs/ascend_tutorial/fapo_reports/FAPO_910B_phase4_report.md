# FAPO × GenRM × Megatron × Ascend 910B 打通性阶段性实验报告

Last updated: 10/19/2026

> 阶段：Phase 4 — 100 步长跑打通性验证
> 作业 ID：711597（Succeeded，2026-08-13T02:22:08Z）
> 报告日期：2026-08-15

---

## 1. 总体结论

FAPO 算法在 Ascend 910B 单节点 8 卡上的 Megatron 训练路径已**完整跑通 100 步**，无 OOM、无崩溃，训练收敛指标正常。step 25 OOM 障碍通过关闭中途验证（`TEST_FREQ=-1`）成功跨过，根因经运行数据证实。

**核心指标速览**：

| 维度 | 结果 |
|------|------|
| 训练完成 | 100/100 步，`LONGRUN_EXIT=0` |
| step 25 OOM 点 | 成功跨过（修复路径验证生效） |
| NPU 内存稳态 | 52.86 / 54.86 GiB（60.96 GiB total，headroom ~6.8 GiB） |
| 稳态吞吐 | ~70 tokens/s（64–79 区间，峰值 78.65） |
| 训练收敛 | entropy 0.599 → 0.271 持续下降，pearson_corr ~0.999 |
| reward 稳态 | mean ~-0.32（含 flawed-positive 惩罚，符合 FAPO reward 侧设计） |

---

## 2. 机器与作业配置

### 2.1 集群与资源

| 配置项 | 值 |
|--------|-----|
| 集群 | wulan |
| 资源池 | infra910b（resourcePoolId=225，Ascend 910B，capacity 312 GPU） |
| 镜像 | `infra/math-training:verl` |
| 文件系统 | filesystemId=268（创建时自动绑定） |
| GPU | 8 × 910B |
| 节点数 | 1 |
| 优先级 | PREEMPTABLE（抢占式，`requeueOnPreemption=True`） |

### 2.2 设备切分（单节点 8 卡）

| 角色 | GPU 数 | 并行策略 |
|------|--------|----------|
| Actor / Ref（Megatron） | 6 | TP=2, PP=1, CP=1 → DP=3 |
| GRM（vLLM，colocate） | 2 | TP=1 |

- `REWARD_MODE=colocate`：GRM 与 actor 共节点，独立 2 卡资源池。
- GQA 约束：Qwen2.5-Math-7B 28 heads / 4 KV heads，TP 必须整除 4 且整除 ACTOR_GPUS，故 `TRAIN_TP=2`。

---

## 3. 训练配置

> 配置来源：`examples/ascend_extras/fapo_trainer/run_qwen2_5_math_7b_megatron.sh`，由 `run_longrun.sh` 以环境变量覆盖。

### 3.1 模型与数据

| 配置项 | 值 |
|--------|-----|
| Actor 模型 | Qwen2.5-Math-7B（Megatron mcore） |
| GRM 模型 | FAPO-GenRM-4B |
| 训练数据 | DAPO-Math-17k（×20 重复，347,960 行） |
| `TRAIN_MAX_SAMPLES` | 6400（=16 × 100 × 4，cap 在加载前生效） |
| `MAX_PROMPT_LENGTH` | 1024 |
| `MAX_RESPONSE_LENGTH` | 3072 |

### 3.2 训练超参

| 配置项 | 值 | 说明 |
|--------|-----|------|
| `TOTAL_TRAINING_STEPS` | 100 | — |
| `TRAIN_BATCH_SIZE` | 16 | — |
| `N_RESP_PER_PROMPT` | 8 | 每 prompt 8 条 rollout |
| `PPO_MINI_BATCH_SIZE` | 6 | 需满足 (mini_batch × n) % dp == 0 |
| `USE_DYNAMIC_BSZ` | True | 跳过静态整除检查 |
| `lr` | 1e-6 | — |
| `LR_WARMUP_STEPS` | 10 | — |
| `weight_decay` | 0.1 | — |
| `GRAD_CLIP` | 1.0 | Megatron 侧 |
| `ADV_ESTIMATOR` | grpo | — |
| `CLIP_RATIO_LOW / HIGH` | 0.2 / 0.28 | — |
| `CLIP_RATIO_C` | 10.0 | — |
| `LOSS_AGG_MODE` | token-mean | — |
| `entropy_coeff` | 0 | — |
| `kl_ctrl.kl_coef` | 0.0 | FAPO 不在 loss 侧加 KL |

### 3.3 内存与计算策略

| 配置项 | 值 | 说明 |
|--------|-----|------|
| `ROLLOUT_MEM_UTIL` | 0.45 | vLLM 启动时按 total 内存的比例检查 |
| `OPTIMIZER_OFFLOAD_FRACTION` | 1 | 全量 offload optimizer state |
| `param_offload / grad_offload / optimizer_offload` | True | Megatron 三段 offload |
| `RECOMPUTE` | selective | 仅 core_attn 重计算 |
| `GRM_MEM_FRACTION` | 0.90 | GRM vLLM util |
| `GRM_MAX_NUM_SEQS` | 256 | GenRM 单副本 batching 上限 |
| `GRM_MAX_MODEL_LEN` | 8192 | — |

### 3.4 检查点与验证（关键修复项）

| 配置项 | 值 | 说明 |
|--------|-----|------|
| `SAVE_FREQ` | 5 | 每 5 步保存 |
| `TEST_FREQ` | **-1** | **关闭中途验证**（修复 step 25 OOM） |
| `VAL_BEFORE_TRAIN` | False | 关闭训练前验证 |
| `resume_mode` | auto | 抢占后从最近 ckpt 恢复 |

**验证关闭的影响**：`test_freq<=0` 使 `ray_trainer.py:1690` 的 gate（含 `is_last_step`）永不触发，故全程无 validation pass。打通性 run 可接受；val metrics 可后续单独 val_only pass 获取。

---

## 4. OOM 根因与修复

### 4.1 现象

前序作业 709340 在 step 1–24 稳态运行（52.4 GiB / 60.96 GiB，~8.5 GiB headroom），但在 **step 25 OOM**，崩溃点在 actor update（`mlp.py:225` glu → matmul），OOM 时仅 32–324 MiB free。

### 4.2 根因

`TEST_FREQ=25` 在 step 25 触发 `_validate()`，`val_batch_size=None` 一次性对全部 2712 行 val 数据做 `do_sample=True` 生成。该 transient 峰值叠加 25 步累积的 NPU 内存碎片（40–44 GiB reserved），越过 60.96 GiB 上限。这是**碎片化症状而非真实容量上限**。

### 4.3 修复

`TEST_FREQ=-1` 关闭中途验证，移除该 transient 峰值。作业 711597 全程 100 步无 OOM，稳态内存 52.86 GiB，证实根因诊断正确。

---

## 5. 吞吐率分析

### 5.1 实测序列

日志窗口（`cctl pytorchjob logs` 截断，仅保留末段）捕获到 step 46–49 与 95–100 的完整指标：

| step | 吞吐 (tokens/s) | t_step (s) | 备注 |
|------|-----------------|------------|------|
| 46 | 65.77 | 248.8 | — |
| 47 | 65.77 | 243.4 | — |
| 48 | 64.48 | 229.8 | — |
| 49 | 74.15 | 214.7 | — |
| 95 | 27.78 | 516.9 | **SAVE_FREQ 命中**，ckpt 保存 ~296s 计入 step |
| 96 | 68.33 | 232.8 | — |
| 97 | 74.58 | 229.9 | — |
| 98 | **78.65** | 230.7 | 全程峰值 |
| 99 | 63.51 | 228.8 | — |
| 100 | 31.63 | 526.1 | **SAVE_FREQ 命中** + 末步 |

### 5.2 结论

- **稳态吞吐 ~70 tokens/s**（排除 ckpt 保存步），区间 64–79。
- **达不到 100 tokens/s**：稳态差约 30%，即便取峰值 78.65 仍差 21%。
- **ckpt 保存步的低值是测量假象**：`timing_s/save_checkpoint` ~296s 计入 `t_step` 分母，把吞吐压低到 27–32 tokens/s，非真实训练吞吐退化。评估稳态应排除 `SAVE_FREQ` 命中步。
- **单步时间主导吞吐**：非 ckpt 步 `t_step ≈ 230s`，其中 `gen`（rollout 生成）~127s + `update_actor`（反向+优化器）~86s 占大头。要逼近 100 tokens/s 需把单步压到 ~160s，瓶颈在 `gen` 与 `update_actor` 两段。

### 5.3 吞吐优化空间（未实施，仅分析）

| 段 | 占比 | 可调方向 |
|----|------|----------|
| `gen`（~127s） | ~55% | rollout 并发、`ROLLOUT_MEM_UTIL` 上调、chunked prefill 调优 |
| `update_actor`（~86s） | ~37% | optimizer offload 比例、recompute 粒度、HCCL 通信 overlap |
| `update_weights`（~6.5s） | ~3% | — |
| `old_log_prob`（~9s） | ~4% | — |

---

## 6. Reward 与训练收敛变化

### 6.1 指标序列（可观测窗口）

| step | entropy | loss | reward_mean | reward_max | reward_min | resp_len_mean | pearson |
|------|---------|------|-------------|------------|------------|---------------|---------|
| 46 | 0.2786 | 0.1432 | -0.3146 | 1.00 | -2.00 | 868.5 | 0.9991 |
| 47 | 0.2351 | 0.0634 | -0.4709 | 1.00 | -2.00 | 860.1 | — |
| 48 | 0.2917 | 0.1459 | -0.6462 | 1.00 | -2.00 | 803.5 | — |
| 49 | 0.2710 | 0.1607 | -0.6406 | 1.00 | -2.00 | 853.8 | — |
| 96 | 0.2656 | -0.0048 | -0.2919 | 1.00 | -1.36 | 854.8 | 0.9992 |
| 97 | 0.2042 | 0.0086 | -0.3281 | 1.00 | -2.00 | 926.8 | — |
| 98 | 0.2304 | 0.0304 | -0.5234 | 1.00 | -2.00 | 1004.7 | — |
| 99 | 0.2178 | -0.0574 | -0.1719 | 1.00 | -1.00 | 780.5 | — |
| 100 | 0.2711 | 0.0079 | -0.3470 | 1.00 | -1.41 | 891.8 | 0.9989 |

> 注：日志截断导致 step 3–45、50–94 指标不可得。step 3 `entropy=0.599` 来自前序子任务报告（已确认）。

### 6.2 收敛分析

- **entropy**：从 step 3 的 0.599 降至 step 46 的 0.279，至 step 100 的 0.271，**策略熵持续下降**，表明策略在逐步聚焦。step 95–100 区间 entropy 在 0.20–0.27 波动，趋于稳态。
- **loss**：step 46–49 在 0.06–0.16，step 96–100 收敛至 ±0.06 附近小幅震荡，**loss 趋稳**。
- **pearson_corr**：step 46/96/100 均 ~0.999，ref 与 rollout logits 高度一致，**rollout 与训练 policy 对齐良好**，无策略漂移。
- **reward_mean**：全程稳定在 -0.17 ~ -0.65 区间，step 100 为 -0.347。**负值符合 FAPO reward 侧设计**——DAPO reward manager 含 flawed-positive 惩罚（`reward_min` 触及 -2.0），整体 reward 均值为负是预期行为，非训练退化。
- **reward_max / reward_min**：`max` 恒为 1.0（正确回答上限），`min` 在 -1.0 ~ -2.0 之间（flawed-positive 惩罚下限）。
- **response_length**：稳态 780–1005 tokens，`clip_ratio` 0.008–0.016，`aborted_ratio` 0.0——**生成长度健康，几乎无截断**。

### 6.3 结论

训练收敛正常：策略熵下降、loss 趋稳、ref/rollout 一致性高、reward 行为符合 FAPO 算法预期。无发散、无 NaN、无策略崩溃迹象。

---

## 7. NPU 内存稳态

| 指标 | 值 (GiB) |
|------|----------|
| 总显存 | 60.96 |
| `max_memory_allocated` | 52.86 → 53.63（step 100） |
| `max_memory_reserved` | 54.13 → 54.86 |
| headroom | ~6.8 |
| CPU memory used | ~433.9 |

内存稳态无碎片化恶化趋势：step 46 与 step 100 的 `max_memory_allocated` 仅差 0.77 GiB，`reserved` 差 0.73 GiB。关闭 validation 后，全程无 transient 峰值冲击。

---

## 8. 待办与下一步

1. **PR 交付**（pending，需用户确认）：提交 FAPO 打通性交付 PR 到 `xldeng-chn/verl` fork。
2. **吞吐优化**（可选）：当前稳态 ~70 tokens/s，距 100 差 30%。主要瓶颈在 `gen`（rollout）与 `update_actor`（反向+优化器），需单独的优化迭代验证。
3. **validation 恢复**（可选）：当前为打通性 run 关闭了验证。若需 reward 曲线全量评估，可跑单独 val_only pass，或调小 `val_batch_size` + 分批验证以避免 OOM 峰值。
4. **早期指标补全**（可选）：日志截断导致 step 3–45、50–94 指标缺失。如需完整收敛曲线，可降低 `SAVE_FREQ` 日志量或启用持久化 metrics 后端。

---

## 附录 A：关键文件

| 文件 | 用途 |
|------|------|
| `examples/ascend_extras/fapo_trainer/run_qwen2_5_math_7b_megatron.sh` | launcher 配置（read only，未修改） |
| `/tmp/fapo_longrun/run_longrun.sh` → remote `/data/user/verl-fapo/run_longrun.sh` | Phase 4 长跑 entry script |
| `/tmp/fapo_longrun/extract_metrics.py` | 指标提取脚本 |
| `/tmp/fapo_longrun/metrics_table.txt` | 原始指标表 |

## 附录 B：运行时序（作业 711597）

| 时刻 (UTC) | 事件 |
|------------|------|
| 11:33:48 | 作业创建 |
| ~11:37 | pod 启动 |
| ~11:38 | config dump |
| ~11:44 | WorkerDict + NPU 替换 |
| ~11:45 | DDP setup（WORLD_SIZE=6） |
| ~11:50 | vLLM actor（pid=16723） |
| ~11:58 | vLLM GRM（pid=20599） |
| ~12:13 | step 3 稳态（初始化耗时 ~36 min） |
| 02:21:04 | step 100 checkpoint 保存完成 |
| 02:22:08 | 作业 Succeeded |
