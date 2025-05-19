**文档三：SGLang 用于 VERL 中的 DeepSeek-V3 Rollout**

**目标**: 分析如何将 SGLang 集成到 `verl` 中作为 DeepSeek-V3 的高效 rollout (推理/生成) 引擎，重点关注模型加载、与 `verl` worker 的交互、权重同步以及并行策略。

**关键信息来源 (@docs 参考)**：
*   `docs/workers/sglang_worker.rst` (SGLang 后端指南)
*   `docs/sglang_multiturn/multiturn.rst` (SGLang 多轮对话支持)
*   `docs/plans/dsv3/inference_engine.md` (推理引擎选择，提及 SGLang 对 DeepSeek V3 的优化如 MLA)
*   `docs/plans/support_deepseek_v3.md` (总体规划中涉及 SGLang 的部分)
*   `docs/examples/config.rst` (Rollout 相关配置，如 `actor_rollout_ref.rollout.name=sglang`)
*   `third_party/sglang_for_verl/docs/references/deepseek.md` (假定存在，含 SGLang 启动 DeepSeek V3 细节)
*   `docs/hybrid_flow.rst` (`verl` 的 HybridFlow 设计原则)

**3.1 SGLang DeepSeek-V3 推理能力与特性**

*   **原生支持与优化**:
    *   根据 `docs/plans/dsv3/inference_engine.md`，SGLang 对 DeepSeek V3 已有完整支持，并包含特定优化，如 MLA (Multi-Latency Attention) 和 DP attention。这些优化对于提升长序列推理性能至关重要。
    *   SGLang 的 RadixAttention 内核和对多种后端的支持（如 Triton）是其高性能的基础。
*   **模型加载与格式**:
    *   SGLang 通常通过其 `Runtime` 加载 HuggingFace 格式的模型 (包含 `config.json`, `pytorch_model.bin` 或 `.safetensors`, `tokenizer.json` 等)。
    *   `verl` 通过配置 `actor_rollout_ref.rollout.name=sglang` (或 `sglang_async` 用于多轮对话) 来启用 SGLang 引擎。
    *   SGLang 在加载 DeepSeek-V3 时，会依据其 `config.json` 中的 MoE (如 `num_experts`, `moe_router_topk`) 和 MLA (`multi_latent_attention_enable`) 等参数来正确实例化模型并应用优化。
*   **并行化支持**:
    *   SGLang 自身支持张量并行 (TP)。`verl` 配置文件中的 `actor_rollout_ref.rollout.tensor_model_parallel_size` 参数将传递给 SGLang Runtime 来配置其 TP 程度。
    *   `docs/workers/sglang_worker.rst` 中提到 `SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=True` 环境变量，可能用于解决多卡初始化时由于 Ray Actor 启动时间不一致等原因导致的 GPU 间可用显存不均衡问题。

**3.2 VERL 中 SGLang Worker (`SGLangRolloutWorker`) 的集成方案**

*   **Worker 实现**:
    *   `verl` 体系中应包含一个 `SGLangRolloutWorker` 类 (例如在 `verl/workers/rollout/sglang_rollout.py`)，该类封装 SGLang Runtime 的初始化、API 调用及与 `verl` 控制流的交互。
    *   此类继承自 `verl.single_controller.base.Worker`，以便融入 `verl` 的 Ray Actor 分布式架构。
*   **初始化 (`init_model`)**:
    *   Worker 的 `init_model` 方法负责根据 `verl` 的配置参数（如 `actor_rollout_ref.model.path` 指向 HF DeepSeek-V3 checkpoint，`actor_rollout_ref.rollout.tensor_model_parallel_size`，以及 SGLang 特有的如 `gpu_memory_utilization`）来实例化 `sglang.Runtime`。
*   **核心 API 接口**:
    *   `generate_sequences(self, prompts: DataProto) -> DataProto`:
        *   输入：包含待生成 prompts 的 `DataProto` 对象。
        *   调用：SGLang Runtime 的生成接口（如 `generate` 或 `stream_generate`）。
        *   参数处理：从 `verl` 的 rollout 配置中获取采样参数 (`temperature`, `top_p`, `top_k`, `max_new_tokens` 等) 并传递给 SGLang。
        *   输出：包含生成文本 (responses) 和相关元信息（如生成长度）的 `DataProto` 对象。
        *   多轮/Agentic RL (`docs/sglang_multiturn/multiturn.rst`)：如果启用，需要使用 `sglang_async` 引擎，并能够处理工具调用（通过 `verl` 定义的 `BaseTool` 子类）和维护对话状态。
    *   `compute_log_prob(self, data: DataProto) -> DataProto`:
        *   输入：包含 prompts 和已生成 responses 的 `DataProto` 对象。
        *   调用：SGLang Runtime 提供的获取序列对数概率的接口。SGLang 必须能够返回序列中每个 token 的精确 logprob，这对于 PPO/GRPO 的重要性采样和 KL 散度计算至关重要。
        *   输出：包含 `log_probs` (通常是每个 token 的对数概率序列) 的 `DataProto` 对象。
*   **Ray Actor 封装**: `SGLangRolloutWorker` 的实例将作为 Ray Actor 运行在指定的 GPU 上，由 `RayPPOTrainer` 进行调度和控制。

**3.3 关键环节：从 Pai-Megatron-Patch (训练) 到 SGLang (Rollout) 的权重同步**

*   **核心挑战**: Actor 模型在 Pai-Megatron-Patch (Megatron-Core) 后端训练，其权重以 Megatron 的分布式格式（分片）存储。Rollout 时，SGLang 需要加载 HuggingFace 格式的权重。这一转换和同步过程是 HybridFlow 设计中的关键。
*   **拟议机制 (`MegatronSGLangShardingManager` - 拟议)**:
    1.  **保存/导出 Megatron Checkpoint**: 在训练过程中，`MegatronPPOWorker` (运行 Pai-Megatron-Patch) 会定期保存 Actor 模型的 checkpoint。
    2.  **转换为 HuggingFace 格式**:
        *   需要一个健壮的工具或流程，将 Megatron 分片权重（特别是包含 MoE 专家权重的复杂结构）合并并转换为标准的 HuggingFace `pytorch_model.bin` (或 `.safetensors`) 格式。
        *   `verl` 的 `scripts/model_merger.py` (见 `docs/advance/checkpoint.rst`) 提供了 FSDP/Megatron checkpoint 到 HF 格式的转换能力。此脚本必须确保能够正确处理 DeepSeek-V3 (MoE) 的 Megatron checkpoint。专家权重的正确排列和合并是难点。
    3.  **通知与加载**:
        *   `RayPPOTrainer` 在 Actor 模型更新并完成权重转换后，需要通知相关的 `SGLangRolloutWorker` 实例有新的权重可用。
        *   `SGLangRolloutWorker` 接收到通知后，需要从指定路径加载新的 HuggingFace 格式权重。这可能涉及：
            *   **方案1 (重启Runtime)**：停止当前的 SGLang Runtime，然后使用新权重重新初始化一个新的 Runtime。这是较简单但可能有较大开销的方案。
            *   **方案2 (SGLang动态加载/热更新)**：如果 SGLang 支持更轻量级的模型权重热更新或版本切换机制，应优先采用，以减少中断和开销。
*   **`MegatronSGLangShardingManager` 的职责**:
    *   此类 (可类比 `MegatronVLLMShardingManager`) 应封装权重转换和加载的协调逻辑。
    *   它可以由 `RayPPOTrainer` 调用，负责触发 Megatron checkpoint 到 HF 格式的转换（可能调用 `scripts/model_merger.py`），并将新权重路径或加载指令传递给 `SGLangRolloutWorker`。
*   **性能与效率考量**:
    *   频繁的权重转换和 SGLang Runtime 重启可能成为显著的性能瓶颈。同步频率（例如，每 N 个训练迭代或仅在评估前同步）需要权衡。
    *   优化 `model_merger.py` 对 MoE 模型处理的效率。

**3.4 SGLang Rollout 的并行策略与资源管理**

*   **SGLang 内部 TP**: 通过 `actor_rollout_ref.rollout.tensor_model_parallel_size` 配置，SGLang Runtime 会在分配给它的 GPUs 之间进行张量并行。
*   **VERL 层面的数据并行**: `verl` 可以实例化多个独立的 `SGLangRolloutWorker` Ray Actor 实例，每个实例处理一部分 prompts，从而实现数据层面的并行，提高总体 rollout 吞吐量。
*   **资源池与分配**: `verl` 的 `ResourcePoolManager` 负责为这些 SGLang worker 实例分配 GPU 资源。这些资源可以与训练 GPU 池隔离，也可以在训练和推理间分时复用。
*   **GPU 显存**: `actor_rollout_ref.rollout.gpu_memory_utilization` 参数控制 SGLang 使用的显存比例。需要精细调整以避免 OOM，尤其是在与训练任务共享 GPU 时。SGLang 的 KV 缓存是主要的显存消耗。
*   **多节点 Rollout**: `docs/workers/sglang_worker.rst` 提到 SGLang 支持跨机推理。若 `verl` 需要在大规模集群（如512 H100）上进行分布式 rollout，Ray 和 SGLang 的网络配置必须支持高效的跨节点通信和模型加载（SGLang 可能需要在每个节点独立加载模型，或支持某种形式的分布式模型服务）。

**3.5 主要技术挑战与风险点**

*   **权重同步的效率、正确性和鲁棒性**:
    *   确保 `scripts/model_merger.py` 或类似工具能完美处理 Megatron MoE checkpoint 到 HF MoE checkpoint 的转换，特别是专家权重的复杂映射。
    *   最小化因权重更新导致的 SGLang 服务中断或延迟。
*   **SGLang `compute_log_prob` 功能的完备性**: SGLang 必须提供稳定、高效且精确的接口来获取生成序列中每个 token 的对数概率，这对 PPO/GRPO 的重要性采样和 KL 散度计算至关重要。
*   **SGLang 在大规模、长时间 RLHF 任务中的稳定性**: 作为一个持续演进的推理引擎，其在 `verl` 这种复杂和长时间运行的 RLHF 场景下的稳定性和对各种边缘情况（如长尾请求、突发流量）的处理能力需要得到验证。
*   **多轮对话与工具使用 (`sglang_async`) 的集成复杂度**:
    *   若 GRPO 训练场景涉及多轮交互或模型需调用外部工具（如代码执行沙箱、数学计算器等），`SGLangRolloutWorker` 需与 SGLang 的 Agentic RAG / 工具调用框架（基于 `docs/sglang_multiturn/multiturn.rst` 的描述）紧密集成。
    *   这包括正确处理异步工具调用、维护多轮对话状态，以及确保奖励函数能解析和利用工具调用的结果。
*   **模型逻辑一致性**: SGLang 加载和执行的 DeepSeek-V3 模型（源自 HF 格式）必须与 Pai-Megatron-Patch 用于训练的模型在核心逻辑和行为上保持等价，以确保策略梯度能够正确传递和应用。任何由不同后端实现差异导致的行为不一致都可能损害训练效果。