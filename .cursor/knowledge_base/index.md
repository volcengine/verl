# AI Agent 协作知识库索引

本知识库旨在结构化地记录与 AI Agent（尤其是学者模式下）协作的最佳实践、原则、能力、局限性以及理想的协作模式。

## 核心知识领域 (可视化)

## 核心知识领域 (列表)

*   **[01 原则 (Principles)](./01_principles.md):** 指导协作与开发的基本原则。
*   **[02 AI 能力 (AI Capabilities)](./02_ai_capabilities.md):** AI Agent 的核心能力、潜力与独特优势。
*   **[03 AI 局限 (AI Limitations)](./03_ai_limitations.md):** AI Agent 的根本性局限与挑战。
*   **[04 协作模式 (Collaboration Model)](./04_collaboration_model.md):** 人机协作的最佳实践、角色定义与理想工作流。
*   **[05 项目上下文 (Project Context)](./05_project_context.md):** (可选) 特定项目相关的背景、决策记录与风险评估。
*   **[06 可验证任务 (Verifiable Tasks)](./06_verifiable_tasks.md):** (可选) 用于验证知识应用的任务模板或案例研究。
*   **[经典文献 (Classical Texts)](./classical_texts/):** 包含《孙子兵法》、《庖丁解牛》、《中庸》、《大学》等原文。
*   **[misc (杂项/外部洞见)](./misc/):** 如 Terence Tao 思想总结等。 

```mermaid
graph TD
    A["What: Initial Goal\n(Effective AI Collab for Complex Tasks\ne.g., fp8_deep)"] --> B{"Why Limited?\n(Current AI Limits, GPU Cost,\nInteraction Issues)"};
    B --> C["Insight: Deeper Understanding\nAI Limits (Value, Novelty, Common Sense)\nAI Potential (Reasoning, Scale, Forking)"];
    C --> D["Insight: Lessons from Tao\n(Rigor/Intuition Stages, Learning/Relearning,\nStrategy, 'Dumb Questions', 'Wastebasket')"];
    D --> E{"How to Proceed?\n(Strategy Formulation)"};
    E --> F["Decision 1 (How & Where):\nEmbrace 'Mental Environment'\n(KB+Retrieval > Raw Large Context)"];
    F --> G["Analysis (Why this How):\nKB Build Cost (`C_kb`)\nvs. Runtime Efficiency & Knowledge Scope"];
    G --> H["Deeper Insight:\nKB Construction is Itself Learning for AI"];
    H --> I{"How to Train/Validate Advanced Abilities?\n('Taste', 'Persistence', Deep Understanding)"};
    I --> J["Insight:\nDirect Subjective Validation is Noisy/Difficult"];
    J --> K["Decision 2 (How & Where):\nFocus on 'Replicating Classics' as Proxy Task\n(Leverages AI strengths, Objective Anchor,\nImplicitly trains desired qualities)"];
    K --> L["Where We Are:\nDefined KB Structure & Focused Training Paradigm"];

    %% Style and Links for clarity
    linkStyle default interpolate basis
    classDef insight fill:#f9f,stroke:#333,stroke-width:2px;
    classDef decision fill:#ccf,stroke:#333,stroke-width:2px;
    classDef question fill:#ff9,stroke:#333,stroke-width:1px,stroke-dasharray: 5 5;
    class C,D,G,H,J insight;
    class F,K decision;
    class B,E,I question;

    %% Explicit influence lines (dashed)
    C -.-> F;
    D -.-> K;
    J -.-> K;
``` 