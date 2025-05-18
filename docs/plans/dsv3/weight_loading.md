# 權重加載分析與風險

DeepSeek V3 需要特殊的權重切分邏輯，目前 `dtensor_weight_loaders.py` 僅實現 V2。整合 V3 時應仿照 Pai-Megatron-Patch 或 vLLM 的實現。

**閱讀重點**
- vLLM 是否已內建 DeepSeek V3 loader。
- Pai-Megatron-Patch 中的權重切分方式與張量並行設定。

**風險預測**
1. 權重切分錯誤會造成模型啟動後輸出異常或直接崩潰。
2. FP8 權重可能需特定 CUDA 版本，需在文檔中提醒。
3. 若依賴外部工具轉換權重，需考慮版本兼容問題。
