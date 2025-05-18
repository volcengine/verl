# 推理引擎選擇與風險

SGLang 已經對 DeepSeek V3 提供完整支持，vLLM 也在逐步添加相關功能。決定在 `verl` 中採用哪個推理框架，會影響接口設計與性能表現。

**閱讀重點**
- SGLang 文檔中對 DeepSeek V3 的特別優化，如 MLA 與 DP attention。
- vLLM 官方是否提供相同功能，以及其 tensor parallel 支持程度。

**風險預測**
1. 如同時維護兩套推理流程，代碼複雜度會明顯上升。
2. SGLang 與 vLLM 的張量並行實現可能不兼容，需確認 API 層可以抽象。
3. 不同推理框架在 FP8 支持上差異較大，可能造成性能落差。
