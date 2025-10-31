==========================================
Verl 大模型最佳实践（DAPO + Qwen3-235B）
==========================================

本文目的
--------

以 DAPO 算法训练 Qwen3-235B 为例，逐步拆解优化目标中的每个参数对应的配置以及最佳实践，帮助用户在自己的场景下，能够自行给出合理的配置。

.. note::

   1. 本文仅包含对复现 DAPO 算法中部分参数的解析，如要查看全部参数，请参阅 verl 源码中 config 部分，具体路径为：https://github.com/volcengine/verl/tree/main/verl/trainer/config
   2. 由于 PPO/GRPO 算法中使用了 KL 散度约束模型，本文中也加入该模型进行讲解，即文本中所用配置，可以视为加入了 KL 散度约束的 DAPO 模型。

优化目标
--------

DAPO 优化目标
~~~~~~~~~~~~~~

.. math::

   \begin{aligned}
   \mathcal{J}_{\mathrm{DAPO}}(\theta)= & \mathbb{E}_{(q, a) \sim \mathcal{D},\left\{o_i\right\}_{i=1}^G \sim \pi_{\theta_{\text {old }}}(\cdot \mid q)} \\
   & {\left[\frac{1}{\sum_{i=1}^G\left|o_i\right|} \sum_{i=1}^G \sum_{t=1}^{\left|o_i\right|} \min \left(r_{i, t}(\theta) \hat{A}_{i, t}, \operatorname{clip}\left(r_{i, t}(\theta), 1-\varepsilon_{\text {low }}, 1+\varepsilon_{\text {high }}\right) \hat{A}_{i, t}\right)\right] } \\
   \text { s.t. } \quad & 0<\mid\left\{o_i \mid \text { is\_equivalent }\left(a, o_i\right)\right\} \mid<G,
   \end{aligned}

.. math::

   r_{i, t}(\theta)=\frac{\pi_\theta\left(o_{i, t} \mid q, o_{i,<t}\right)}{\pi_{\theta_{\text {old }}}\left(o_{i, t} \mid q, o_{i,<t}\right)}, \quad \hat{A}_{i, t}=\frac{R_i-\operatorname{mean}\left(\left\{R_i\right\}_{i=1}^G\right)}{\operatorname{std}\left(\left\{R_i\right\}_{i=1}^G\right)}

GRPO 优化目标
~~~~~~~~~~~~~~

.. math::

   \begin{aligned}
   \mathcal{J}_{G R P O}(\theta) & =\mathbb{E}\left[q \sim P(Q),\left\{o_i\right\}_{i=1}^G \sim \pi_{\theta_{\text {old }}}(O \mid q)\right] \\
   & \frac{1}{G} \sum_{i=1}^G \frac{1}{\left|o_i\right|} \sum_{t=1}^{\left|o_i\right|}\left\{\min \left[\frac{\pi_\theta\left(o_{i, t} \mid q, o_{i,<t}\right)}{\pi_{\theta_{\text {old }}}\left(o_{i, t} \mid q, o_{i,<t}\right)} \hat{A}_{i, t}, \operatorname{clip}\left(\frac{\pi_\theta\left(o_{i, t} \mid q, o_{i,<t}\right)}{\pi_{\theta_{\text {old }}}\left(o_{i, t} \mid q, o_{i,<t}\right)}, 1-\varepsilon, 1+\varepsilon\right) \hat{A}_{i, t}\right]-\beta \mathbb{D}_{K L}\left[\pi_\theta \| \pi_{r e f}\right]\right\},
   \end{aligned}

参数详解
--------

.. list-table:: 参数详解
   :header-rows: 1
   :widths: 24 20 22 34

   * - 原符号及含义
     - 相关配置
     - 配置解读
     - 最佳实践

   * - :math:`(q,a)\sim D`
     
     ``D``：数据集
     
     ``q``：数据集中输入（question/prompt）
     
     ``a``：目标输出（通常为最终答案，不含推理过程）
     - ``data.train_files``
       
       ``data.val_files``
     - 训练集路径
       
       测试集路径
     - 需为 ``.parquet``，参考 ``examples/data_preprocess`` 转换脚本；转换前确保 ``data_source`` 已实现匹配的 reward function；可使用 HuggingFace ``BytedTsinghua-SIA/DAPO-Math-17k`` 数据集

   * - 
     - ``data.prompt_key``
     - 训练集中 prompt 对应的 key
     - 按需设置；如无特殊要求建议使用 ``prompt`` 便于理解

   * - 
     - ``data.max_prompt_length``
     - prompt 最大长度
     - 设置为数据集中最长 prompt；若长尾样本导致值过大，可下调并配合 ``data.truncation``

   * - 
     - ``data.truncation``
     - 输入过长时的处理方式（截断方向或报错）
     - 大多数场景设为 ``left``；训练时关注报表中的 ``clip_ratio``，若 ``prompt_length/clip_ratio`` 偏高且效果欠佳，可增大 ``data.max_prompt_length`` 或预处理数据；严格场景可设为 ``error`` 抛异常

   * - :math:`G`：每个 prompt 的生成数量
     - ``actor_rollout_ref.rollout.n``
     - 单个 prompt 的生成轨迹数量
     - 依据经验或论文设定：GRPO 常用 64，DAPO 常用 16

   * - :math:`\theta`：模型参数
     - ``actor_rollout_ref.model.path``（actor）
     - checkpoint 路径
     - 使用 HuggingFace 兼容格式

   * - 
     - ``actor_rollout_ref.actor.megatron.use_mbridge``
     - 是否启用 mbridge 格式转换
     - 使用 Megatron 训练时建议开启；需配合最新版 mbridge（详见 https://github.com/ISEEKYAN/mbridge）

   * - :math:`\pi`：采样策略
     - ``actor_rollout_ref.rollout.name``
     - rollout 后端
     - Verl 当前支持 ``vllm`` 与 ``sglang``；结合实际测试与官方文档选择并调参

   * - 
     - ``actor_rollout_ref.rollout.response_length``
       
       ``data.max_response_length``
     - rollout 可生成的最大长度（前者优先生效）
     - 长度越大显存占用越高、速度越慢但效果更好；依据显存与训练速度需求设定；监控训练 ``clip_ratio``，若超过 0.1 表示截断多需调整

   * - 
     - ``actor_rollout_ref.rollout.gpu_memory_utilization``
     - rollout 后端显存利用率
     - 在不过 OOM 情况下越大越好；开启 param/grad/optim offload 时可设 0.8-0.9

   * - 
     - ``actor_rollout_ref.rollout.tensor_model_parallel_size``
     - 推理引擎的 tensor 并行度
     - 满足 ``单卡显存 * gpu_memory_utilization * TP > 参数量 * 2``（bf16/fp16）；在满足约束后可逐步增大 TP 提升 KV cache 空间但通信开销增加（尤其 TP>8）；先取满足约束的最小 TP，随后观察速度再调

   * - 
     - ``actor_rollout_ref.rollout.temperature``
       
       ``top_p``
       
       ``top_k``
     - Rollout 阶段采样参数
     - 保持足够随机性，建议 ``temperature=1.0``、``top_p=1.0``、``top_k=-1``

   * - 
     - ``actor_rollout_ref.rollout.val_kwargs.temperature``
       
       ``top_p``
       
       ``top_k``
       
       ``do_sample``
       
       ``n``
     - 验证阶段采样参数
     - 思考模型需 ``temperature>0`` 防止重复；测试集样本少时增大 ``n`` 降低方差（如 AIME24 取 64）；实践参考：初期 ``temperature=1.0``、``top_p=0.7``、``top_k=-1``、``do_sample=True``、``n=1``，最终按测试集规模调大 ``n``

   * - 
     - ``+actor_rollout_ref.rollout.engine_kwargs.vllm.*``
       
       ``+actor_rollout_ref.rollout.engine_kwargs.sglang.*``
     - vllm/sglang 扩展配置
     - 通过 ``+`` 形式注入（示例：``+actor_rollout_ref.rollout.engine_kwargs.vllm.enable_expert_parallel=True``）；参考官方文档获取精确定义；部分配置暂不支持（如 ``pipeline_parallel_size``）；优化项需验证实际收益（TP=32 时 ``enable_expert_parallel=True`` 可能减慢 DeepSeek-V3 rollout）

   * - :math:`\pi_\theta`：参数为 :math:`\theta` 时在策略 :math:`\pi` 下的 actor 模型
     - 
     - 概念说明
     - 与 actor 相关设置见下方各项

   * - 
     - ``data.train_batch_size``
     - 训练阶段累计 batch size（一次 rollout 产出 ``train_batch_size * n`` 样本）
     - 与其他 batch size 紧密相关：单次 forward 输入 ``micro_batch_size * n`` 样本，多次 forward 累积 ``mini_batch_size * n`` 更新 actor，累计 ``train_batch_size * n`` 后更新 old 模型；增大值可减少 rollout 次数但易引入 off-policy 偏差

   * - 
     - ``actor_rollout_ref.actor.ppo_mini_batch_size``
     - Actor 每次迭代的 batch size
     - 类似常规深度学习的 batch size，按经验或公开报告调整

   * - 
     - ``actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu``
     - 每次 forward 时每组 GPU 处理的样本数
     - Megatron 下代表每组包含 TP*PP*CP 的 GPU；取值不超过 ``ppo_mini_batch_size`` 且在不 OOM 情况下尽量大

   * - 
     - ``actor_rollout_ref.actor.use_dynamic_bsz``
     - 是否启用动态 batch
     - 推荐开启以按样本长度自适应分配提升效率

   * - 
     - ``actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu``
     - 动态 batch 下单卡 log_prob 计算的最大 token 数
     - 设为不少于 ``max_prompt_length + max_response_length`` 的倍数以避免截断

   * - 
     - ``actor_rollout_ref.actor.megatron.pipeline_model_parallel_size``
       
       ``tensor_model_parallel_size``
       
       ``expert_model_parallel_size``
       
       ``expert_tensor_parallel_size``
       
       ``context_parallel_size``
     - Megatron 并行度参数（PP/TP/EP/ETP/CP）
     - 显存占用可粗略分为参数/梯度、优化器状态和激活：每个参数在 bf16/fp16 下约占 ``2 / TP`` 字节（若保留 FP32 master weight 或未启用 offload，还需额外 4-8 字节给 Adam），激活量与 ``micro_batch_size × 序列长度 × hidden_size`` 成正比，可通过 gradient checkpointing、动态 batch 或 offload 缓解；优先调高 TP 在单机内分摊模型权重，受限时再引入 PP；长序列配合 CP 扩展上下文；MoE 模型根据专家并行需求设置 EP/ETP（通常与 TP 对齐）；DP 决定总卡数，资源紧张时保持 DP 最小并结合 offload 方案，确保各项并行配置与硬件拓扑、通信开销匹配

   * - 
     - ``actor_rollout_ref.model.use_fused_kernels``
     - 是否启用自定义融合 kernel
     - Verl 对常用模型提供优化，建议开启以获得最佳性能

   * - :math:`\hat{A}_{i,t}`：Group 内第 :math:`i` 个样本在时刻 :math:`t` 的优势
     - ``algorithm.adv_estimator``
     - 优势估计函数
     - DAPO/GRPO 设为 ``grpo``

   * - :math:`R_i`：Group 内第 :math:`i` 个样本的 reward
     - ``reward_model.reward_manager``
     - reward 管理方案
     - DAPO 设为 ``dapo``；GRPO 设为 ``naive``

   * - :math:`D_{KL}`：KL 散度
     - ``algorithm.use_kl_in_reward``
     - reward 中是否使用 KL 约束
     - PPO 设 ``True``；GRPO 与 DAPO 设 ``False``

   * - 
     - ``actor_rollout_ref.actor.use_kl_loss``
     - 损失中是否加入 KL 约束
     - PPO 设 ``False``；GRPO 设 ``True``；DAPO 设 ``False``

   * - :math:`\beta`：KL 损失权重
     - ``actor_rollout_ref.actor.kl_loss_coef``
     - KL 损失系数
     - 可从 0.001 等经验值起步；增大可抑制 reward hacking 但会降低探索能力

   * - 
     - ``algorithm.kl_ctrl.kl_coef``
     - reward 中 KL 系数
     - 按实际需要设定

   * - :math:`\pi_{old}`：每 ``train_batch_size`` 更新的 old 模型
     - ``actor_rollout_ref.rollout.log_prob_use_dynamic_bsz``
     - old 模型计算 log_prob 时是否启用动态 batch
     - 建议开启

   * - :math:`\pi_{ref}`：用于计算 KL 的参考模型
     - ``actor_rollout_ref.ref.log_prob_use_dynamic_bsz``
     - ref 模型计算 log_prob 时是否启用动态 batch
     - 建议开启

   * - 
     - ``actor_rollout_ref.ref.megatron.pipeline_model_parallel_size``
       
       ``tensor_model_parallel_size``
       
       ``expert_model_parallel_size``
       
       ``expert_tensor_parallel_size``
       
       ``context_parallel_size``
     - Ref 模型并行度参数
     - 与 actor 设置保持一致

   * - 
     - ``actor_rollout_ref.ref.megatron.param_offload``
     - Ref 模型是否 offload 至 CPU
     - 虽无梯度与优化器状态，仍建议与 actor 配置保持一致

   * - :math:`o_i,\ \lvert o_i \rvert`：第 :math:`i` 个 prompt 的输出及其长度
     - ``actor_rollout_ref.actor.loss_agg_mode``
     - loss 聚合方式
     - 推荐 token 级聚合 ``token-mean``（符合 Dr.GRPO、DAPO 推荐）；复现原始 GRPO 时用 ``seq-mean-token-mean``

   * - :math:`\pi_\theta(o_{i,t} \mid q_i,o_{i,<t})`：给定 prompt 与前缀时生成 token 的概率
     - ``actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu``
       
       ``actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu``
     - 计算概率时的 batch size
     - rollout engine 先生成输出再拼接输入各模型；设置 batch size 需在显存与吞吐间权衡

   * - :math:`\epsilon_{low},\ \epsilon_{high}`：重要性采样裁剪阈值
     - ``actor_rollout_ref.actor.clip_ratio_low``
       
       ``actor_rollout_ref.actor.clip_ratio_high``
     - 裁剪上下界
     - 按 DAPO 建议设 ``clip_ratio_low=0.2``、``clip_ratio_high=0.28``

   * - vllm 部分推荐开启的优化参数
     - ``actor_rollout_ref.rollout.enable_chunked_prefill``
     - 是否启用分块预填充（vllm）
     - 建议设为 ``True`` 以提升 GPU 利用率；需与 ``max_num_batched_tokens`` 协同；仅对 vllm 生效

   * - 
     - ``actor_rollout_ref.rollout.max_num_batched_tokens``
     - 单个 batch 可处理的最大 token 数
     - 增大可提升利用率；可设为 ``max(8192, max_prompt_length + max_response_length, max_model_len)``；参考 vllm 文档 https://docs.vllm.ai/en/v0.4.2/models/performance.html

   * - 
     - ``actor_rollout_ref.rollout.enforce_eager``
     - 是否禁用 CUDA graph
     - 启用 CUDA graph 通常更快但额外占显存（不受 ``gpu_memory_utilization`` 控制）；仅 vllm 生效；显存不足时设为 ``True``

   * - 
     - ``actor_rollout_ref.rollout.cudagraph_capture_sizes``
     - CUDA graph 捕获的 batch size 列表
     - 默认 ``null``；显存不足时可设为 ``[1,2,4,8,16,32]``

   * - optimizer 相关参数
     - ``actor_rollout_ref.actor.optim.lr``
     - 学习率
     - 从 ``1e-5`` 或 ``1e-6`` 起调

   * - 
     - ``actor_rollout_ref.actor.optim.lr_warmup_steps``
     - 学习率 warmup 步数
     - 建议设置，如 10

   * - 
     - ``actor_rollout_ref.actor.optim.weight_decay``
     - 权重衰减系数
     - 可用经验值 ``0.1``

   * - 
     - ``actor_rollout_ref.actor.optim.clip_grad``
     - 梯度裁剪阈值
     - 建议设为 1

   * - 
     - ``+actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction``
     - 混合优化器在 CPU 中更新的比例
     - 节省显存；大模型（如 DeepSeek）建议开启并设为 1

   * - 
     - ``+actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d``
       
       ``+...use_precision_aware_optimizer``
       
       ``+...optimizer_cpu_offload``
     - 混合优化器辅助开关
     - 启用混合优化器时推荐同时开启

   * - megatron相关参数
     - ``actor_rollout_ref.actor.megatron.param_offload``
       
       ``optimizer_offload``
       
       ``grad_offload``
     - 参数/优化器/梯度是否 offload 至 CPU
     - 显存不足时建议开启

   * - 
     - ``+actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method``
       
       ``recompute_granularity``
       
       ``recompute_num_layers``
     - 重算（gradient checkpointing）配置
     - 减少显存占用但会增加计算；显存不足时启用（如 ``uniform``、``full``、``1``）

   * - 
     - ``+actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype``
       
       ``moe_shared_expert_overlap``
       
       ``moe_permute_fusion``
       
       ``moe_enable_deepep``
       
       ``moe_token_dispatcher_type``
     - MoE 相关设置
     - 按推荐值配置以获得稳定性能（示例：``fp32``、``False``、``True``、``True``、``flex``）

   * - 
     - ``+actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion``
     - 梯度累积融合优化
     - 开启以提升训练速度

   * - 
     - ``+actor_rollout_ref.actor.megatron.override_transformer_config.account_for_embedding_in_pipeline_split``
       
       ``account_for_loss_in_pipeline_split``
       
       ``num_layers_in_last_pipeline_stage``
     - Pipeline parallel 相关配置
     - 模型层数与 PP 不整除时使用；前两项将 embedding/loss 视作一层；``num_layers_in_last_pipeline_stage`` 用于手动指定首尾阶段层数（如需要可设置为 0 或 ``${LAST_LAYER}``）

   * - trainer 相关参数
     - ``trainer.logger``
     - 日志输出目标
     - 可设为 ``['console','wandb']``；火山引擎机器学习平台可设 ``['console','vemlp_wandb']``

   * - 
     - ``trainer.project_name``
       
       ``trainer.experiment_name``
     - 项目与实验名称
     - 结合需要分层命名，便于快速定位和对比实验

   * - 
     - ``trainer.n_gpus_per_node``
       
       ``trainer.nnodes``
     - 节点及单节点 GPU 数
     - 按可用资源配置

   * - 
     - ``trainer.test_freq``
       
       ``trainer.save_freq``
       
       ``trainer.total_epochs``
     - 测试频率、保存频率与总 epoch 数
     - 依据需求设定

   * - 
     - ``trainer.log_val_generations``
     - 日志中保存的验证样本数
     - 初期可设为 10，后续按需调整

   * - 
     - ``trainer.val_before_train``
     - 训练前是否先运行验证
     - 按需开启

