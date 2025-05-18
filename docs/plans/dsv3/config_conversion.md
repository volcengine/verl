# 配置轉換分析與風險

在將 DeepSeek V3 納入 `verl` 時，首先需要實現 `hf_to_mcore_config_dpskv3`。此函式負責將 HuggingFace 的配置轉換為 Megatron-Core 的格式。

**閱讀重點**
- `config_converter.py` 內既有的 V2 轉換流程。
- Pai-Megatron-Patch 提供的 DeepSeek V3 訓練配置。

**風險預測**
1. 轉換後的維度與 dtype 可能與官方模型不一致，造成初始化失敗。
2. 專家並行（MoE）相關配置若填寫錯誤，將導致推理分片不一致。
3. DeepSeek 版本更新可能新增字段，需保持對上游的關注。
