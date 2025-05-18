# 性能優化風險預測

DeepSeek V3 引入 FP8 計算與 MLA 等新特性，對硬體及軟體依賴較高，需要充分評估性能瓶頸和潛在風險。

**閱讀重點**
- SGLang 在 benchmark/deepseek_v3 中提供的性能數據與調優方法。
- Pai-Megatron-Patch 的 FP8 內核及 MLA 相關實現。

**風險預測**
1. 不同 GPU 架構（如 H100、MI300）對 FP8 的支持度可能不同，影響推理速度。
2. MLA 若配置不當會導致顯存占用過高。
3. 性能優化依賴的 Triton 或 CUDA 版本更新頻繁，需要持續追蹤。
