# 在 verl 中集成 DeepSeek V3 的計劃

本文檔說明如何在不依賴大規模測試的前提下，將 DeepSeek V3 集成進 `verl`。核心思路是充分閱讀現有代碼和公開資料，並預測可能的風險。

## 1. 閱讀現有基礎設施

1. **了解當前模型支持情況**
   - 閱讀 `verl/models/mcore`，關注 LLAMA、Qwen2 以及 DeepSeek V2 的集成方式。
       - `config_converter.py` 中 `hf_to_mcore_config_dpskv3` 目前拋出 `NotImplementedError`【F:verl/models/mcore/config_converter.py†L186-L188】。
       - `registry.py` 中雖然列出了 `DEEPSEEK_V3`，但標註為「未測試」【F:verl/models/mcore/registry.py†L55-L68】。
       - 權重加載僅在 `verl/third_party/vllm/*/dtensor_weight_loaders.py` 中支持 DeepSeek V2【F:verl/third_party/vllm/vllm_v_0_6_3/dtensor_weight_loaders.py†L343-L357】。
2. **查閱 DeepSeek 相關文檔**
   - 閱讀 `third_party/sglang_for_verl/docs/references/deepseek.md`，了解如何使用 SGLang 啟動 DeepSeek V3。
   - 參考 `third_party/pai-megatron-patch` 中的示例腳本與配置。

## 2. 參考實現

1. **Pai-Megatron-Patch**
   - 研究 `third_party/pai-megatron-patch/examples/deepseek_v3`，找出配置差異（詞表大小、專家並行等），用於完善 `hf_to_mcore_config_dpskv3`。
2. **SGLang 集成**
   - 在 `third_party/sglang_latest/python/sglang` 中搜索 DeepSeek V3 相關實現，理清推理流程。

## 3. 實現步驟

1. **完善配置轉換**
   - 參考 Pai-Megatron-Patch，補全 `hf_to_mcore_config_dpskv3`。
   - 確保專家並行、MLA 等參數正確。
2. **擴展權重加載**
   - 在 `dtensor_weight_loaders.py` 中新增 `DeepseekV3ForCausalLM` 相關邏輯。
   - 如 vLLM 已支持 DeepSeek V3，則修改 `SUPPORTED_MOE_MODELS`。
3. **更新文檔和示例**
   - 整合 SGLang 深度集成的指南。
   - 標註硬件要求、FP8 注意事項等。
4. **測試策略**
   - 設計與 DeepSeek V2 類似的單元測試。
   - 透過小規模的前向傳播驗證配置和權重加載。

## 4. 討論話題

1. **模型配置默認值**
2. **權重轉換流程**
3. **推理引擎選擇（SGLang 與 vLLM）**
4. **性能與顯存風險**
5. **有限測試的缺陷與補救措施**

## 5. 風險預測

- **配置不匹配**：轉化邏輯可能導致形狀或 dtype 錯誤，應對照 Pai-Megatron-Patch 反覆核對。
- **權重加載錯誤**：若 dtensor 加載不全，模型可能無法啟動；需確認 vLLM 介面。
- **性能退化**：FP8 與 MLA 依賴特定 CUDA 版本，需在文檔中說明。
- **維護負擔**：DeepSeek V3 的更新可能改變字段，需持續跟進上游。

通過上述計劃，我們可以在最小化實際測試的同時推進 DeepSeek V3 的支持。
