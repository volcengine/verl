from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class LMStyle(Enum):
    OpenAIChat = "OpenAIChat"
    OpenAIReasonPreview = "OpenAIReasonPreview"
    OpenAIReason = "OpenAIReason"

    Claude = "Claude"  # Claude 1 and Claude 2
    Claude3 = "Claude3"
    Gemini = "Gemini"
    GeminiThinking = "GeminiThinking"

    MistralWeb = "MistralWeb"
    CohereCommand = "CohereCommand"
    DataBricks = "DataBricks"
    DeepSeekAPI = "DeepSeekAPI"

    GenericBase = "GenericBase"

    DeepSeekCodeInstruct = "DeepSeekCodeInstruct"
    CodeLLaMaInstruct = "CodeLLaMaInstruct"
    StarCoderInstruct = "StarCoderInstruct"
    CodeQwenInstruct = "CodeQwenInstruct"
    QwQ = "QwQ"

    LLaMa3 = "LLaMa3"

    DeepSeekR1 = "DeepSeekR1"


@dataclass
class LanguageModel:
    model_name: str
    model_repr: str
    model_style: LMStyle
    release_date: datetime | None  # XXX Should we use timezone.utc?
    link: str | None = None

    def __hash__(self) -> int:
        return hash(self.model_name)

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "model_repr": self.model_repr,
            "model_style": self.model_style.value,
            "release_date": int(self.release_date.timestamp() * 1000),
            "link": self.link,
        }


LanguageModelList: list[LanguageModel] = [
    ## LLama3 Base (8B and 70B)
    LanguageModel(
        "meta-llama/Meta-Llama-3-70B",
        "LLama3-70b-Base",
        LMStyle.GenericBase,
        datetime(2023, 1, 1),
        link="https://huggingface.co/meta-llama/Meta-Llama-3-70B",
    ),
    LanguageModel(
        "meta-llama/Meta-Llama-3-8B",
        "LLama3-8b-Base",
        LMStyle.GenericBase,
        datetime(2023, 1, 1),
        link="https://huggingface.co/meta-llama/Meta-Llama-3-8B",
    ),
    ## LLama3 Instruct (8B and 70B)
    LanguageModel(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "LLama3-8b-Ins",
        LMStyle.LLaMa3,
        datetime(2023, 1, 1),
        link="https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct",
    ),
    LanguageModel(
        "meta-llama/Meta-Llama-3-70B-Instruct",
        "LLama3-70b-Ins",
        LMStyle.LLaMa3,
        datetime(2023, 1, 1),
        link="https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct",
    ),
    ## LLama3.1 Base (8B, 70B, 405B)
    LanguageModel(
        "meta-llama/Meta-Llama-3.1-8B",
        "LLama3.1-8b-Base",
        LMStyle.GenericBase,
        datetime(2023, 1, 1),
        link="https://huggingface.co/meta-llama/Meta-Llama-3.1-8B",
    ),
    LanguageModel(
        "meta-llama/Meta-Llama-3.1-70B",
        "LLama3.1-70b-Base",
        LMStyle.GenericBase,
        datetime(2023, 1, 1),
        link="https://huggingface.co/meta-llama/Meta-Llama-3.1-70B",
    ),
    LanguageModel(
        "meta-llama/Meta-Llama-3.1-405B-FP8",
        "LLama3.1-405b-Base-FP8",
        LMStyle.GenericBase,
        datetime(2023, 1, 1),
        link="https://huggingface.co/meta-llama/Meta-Llama-3.1-405B-FP8",
    ),
    ## LLama3.1 Instruct (8B, 70B, 405B)
    LanguageModel(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "LLama3.1-8b-Ins",
        LMStyle.LLaMa3,
        datetime(2023, 1, 1),
        link="https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct",
    ),
    LanguageModel(
        "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "LLama3.1-70b-Ins",
        LMStyle.LLaMa3,
        datetime(2023, 1, 1),
        link="https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct",
    ),
    LanguageModel(
        "meta-llama/Meta-Llama-3.1-405B-Instruct-FP8",
        "LLama3.1-405b-Ins-FP8",
        LMStyle.LLaMa3,
        datetime(2023, 1, 1),
        link="https://huggingface.co/meta-llama/Meta-Llama-3.1-405B-Instruct-FP8",
    ),
    ## LLama3.3 Instruct (8B, 70B)
    LanguageModel(
        "meta-llama/Llama-3.3-70B-Instruct",
        "LLama3.3-70b-Ins",
        LMStyle.LLaMa3,
        datetime(2023, 1, 1),
        link="https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct",
    ),
    LanguageModel(
        "meta-llama/Llama-3.3-8B-Instruct",
        "LLama3.3-8b-Ins",
        LMStyle.LLaMa3,
        datetime(2023, 1, 1),
        link="https://huggingface.co/meta-llama/Llama-3.3-8B-Instruct",
    ),
    ## Deepseek-Coder Base (33B, 6.7B, 1.3B)
    LanguageModel(
        "deepseek-ai/deepseek-coder-33b-base",
        "DSCoder-33b-Base",
        LMStyle.GenericBase,
        datetime(2023, 1, 1),
        link="https://huggingface.co/deepseek-ai/deepseek-coder-33b-base",
    ),
    LanguageModel(
        "deepseek-ai/deepseek-coder-6.7b-base",
        "DSCoder-6.7b-Base",
        LMStyle.GenericBase,
        datetime(2023, 1, 1),
        link="https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-base",
    ),
    LanguageModel(
        "deepseek-ai/deepseek-coder-1.3b-base",
        "DSCoder-1.3b-Base",
        LMStyle.GenericBase,
        datetime(2023, 1, 1),
        link="https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-base",
    ),
    ## Deepseek-Coder Instruct (33B, 6.7B, 1.3B)
    LanguageModel(
        "deepseek-ai/deepseek-coder-33b-instruct",
        "DSCoder-33b-Ins",
        LMStyle.DeepSeekCodeInstruct,
        datetime(2023, 9, 1),
        link="https://huggingface.co/deepseek-ai/deepseek-coder-33b-instruct",
    ),
    LanguageModel(
        "deepseek-ai/deepseek-coder-6.7b-instruct",
        "DSCoder-6.7b-Ins",
        LMStyle.DeepSeekCodeInstruct,
        datetime(2023, 9, 1),
        link="https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct",
    ),
    LanguageModel(
        "deepseek-ai/deepseek-coder-1.3b-instruct",
        "DSCoder-1.3b-Ins",
        LMStyle.DeepSeekCodeInstruct,
        datetime(2023, 8, 1),
        link="https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-instruct",
    ),
    ##
    LanguageModel(
        "01-ai/Yi-Coder-9B-Chat",
        "Yi-Coder-9B-Chat",
        LMStyle.DeepSeekAPI,
        datetime(2023, 8, 1),
        link="https://huggingface.co/01-ai/Yi-Coder-9B-Chat",
    ),
    ## Deepseek-Chat Latest API (currently DeepSeek-V3)
    LanguageModel(
        "deepseek-r1-preview",
        "DeepSeek-R1-Preview",
        LMStyle.DeepSeekAPI,
        datetime(2024, 6, 30),
        link="https://api-docs.deepseek.com/news/news1120",
    ),
    LanguageModel(
        "deepseek-r1-lite-preview",
        "DeepSeek-R1-Lite-Preview",
        LMStyle.DeepSeekAPI,
        datetime(2024, 6, 30),
        link="https://api-docs.deepseek.com/news/news1120",
    ),
    LanguageModel(
        "deepseek-chat",
        "DeepSeek-V3",
        LMStyle.DeepSeekAPI,
        datetime(2024, 6, 30),
        link="https://huggingface.co/deepseek-ai/DeepSeek-V3",
    ),
    ## Deepseek-Coder Latest API (currently DeepSeekCoder-V2.5)
    LanguageModel(
        "deepseek-coder",
        "DeepSeekCoder-V2.5",
        LMStyle.DeepSeekAPI,
        datetime(2023, 8, 1),
        link="https://huggingface.co/deepseek-ai/DeepSeek-V2",
    ),
    ## OpenAI GPT-3.5-Turbo
    LanguageModel(
        "gpt-3.5-turbo-0301",
        "GPT-3.5-Turbo-0301",
        LMStyle.OpenAIChat,
        datetime(2021, 10, 1),
        link="https://openai.com/blog/new-models-and-developer-products-announced-at-devday",
    ),
    LanguageModel(
        "gpt-3.5-turbo-0125",
        "GPT-3.5-Turbo-0125",
        LMStyle.OpenAIChat,
        datetime(2021, 10, 1),
        link="https://openai.com/blog/new-embedding-models-and-api-updates#:~:text=Other%20new%20models%20and%20lower%20pricing",
    ),
    ## OpenAI GPT-4, GPT-4-Turbo
    LanguageModel(
        "gpt-4-0613",
        "GPT-4-0613",
        LMStyle.OpenAIChat,
        datetime(2021, 10, 1),
        link="https://platform.openai.com/docs/models/gpt-4-turbo-and-gpt-4",
    ),
    LanguageModel(
        "gpt-4-1106-preview",
        "GPT-4-Turbo-1106",
        LMStyle.OpenAIChat,
        datetime(2023, 4, 30),
        link="https://openai.com/blog/new-models-and-developer-products-announced-at-devday",
    ),
    LanguageModel(
        "gpt-4-turbo-2024-04-09",
        "GPT-4-Turbo-2024-04-09",
        LMStyle.OpenAIChat,
        datetime(2023, 4, 30),
        link="https://platform.openai.com/docs/models/gpt-4-turbo-and-gpt-4",
    ),
    ## OpenAI GPT-4O (and Mini)
    LanguageModel(
        "gpt-4o-2024-05-13",
        "GPT-4O-2024-05-13",
        LMStyle.OpenAIChat,
        datetime(2023, 4, 30),
        link="https://openai.com/index/spring-update",
    ),
    LanguageModel(
        "gpt-4o-2024-08-06",
        "GPT-4O-2024-08-06",
        LMStyle.OpenAIChat,
        datetime(2023, 4, 30),
        link="https://openai.com/index/spring-update",
    ),
    LanguageModel(
        "gpt-4o-mini-2024-07-18",
        "GPT-4O-mini-2024-07-18",
        LMStyle.OpenAIChat,
        datetime(2023, 4, 30),
        link="https://openai.com/index/spring-update",
    ),
    ## O1-Mini and O1-Preview
    LanguageModel(
        "o1-preview-2024-09-12",
        "O1-Preview-2024-09-12",
        LMStyle.OpenAIReasonPreview,
        datetime(2023, 4, 30),
        link="https://platform.openai.com/docs/guides/reasoning",
    ),
    LanguageModel(
        "o1-mini-2024-09-12",
        "O1-Mini-2024-09-12",
        LMStyle.OpenAIReasonPreview,
        datetime(2023, 4, 30),
        link="https://platform.openai.com/docs/guides/reasoning",
    ),
    ## O1 (reasoning models)
    LanguageModel(
        "o1-2024-12-17__low",
        "O1-2024-12-17 (Low)",
        LMStyle.OpenAIReason,
        datetime(2023, 4, 30),
        link="https://platform.openai.com/docs/api-reference/chat/create#chat-create-reasoning_effort",
    ),
    LanguageModel(
        "o1-2024-12-17__medium",
        "O1-2024-12-17 (Med)",
        LMStyle.OpenAIReason,
        datetime(2023, 4, 30),
        link="htthttps://platform.openai.com/docs/api-reference/chat/create#chat-create-reasoning_effort",
    ),
    LanguageModel(
        "o1-2024-12-17__high",
        "O1-2024-12-17 (High)",
        LMStyle.OpenAIReason,
        datetime(2023, 4, 30),
        link="https://platform.openai.com/docs/api-reference/chat/create#chat-create-reasoning_effort",
    ),
    ## O3-Mini
    LanguageModel(
        "o3-mini-2025-01-31__low",
        "O3-Mini-2025-01-31 (Low)",
        LMStyle.OpenAIReason,
        datetime(2023, 4, 30),
        link="https://platform.openai.com/docs/api-reference/chat/create#chat-create-reasoning_effort",
    ),
    LanguageModel(
        "o3-mini-2025-01-31__medium",
        "O3-Mini-2025-01-31 (Med)",
        LMStyle.OpenAIReason,
        datetime(2023, 4, 30),
        link="https://platform.openai.com/docs/api-reference/chat/create#chat-create-reasoning_effort",
    ),
    LanguageModel(
        "o3-mini-2025-01-31__high",
        "O3-Mini-2025-01-31 (High)",
        LMStyle.OpenAIReason,
        datetime(2023, 4, 30),
        link="https://platform.openai.com/docs/api-reference/chat/create#chat-create-reasoning_effort",
    ),
    ## Claude and Claude 2
    LanguageModel(
        "claude-instant-1",
        "Claude-Instant-1",
        LMStyle.Claude,
        datetime(2022, 12, 31),
        link="https://www.anthropic.com/index/introducing-claude",
    ),
    LanguageModel(
        "claude-2",
        "Claude-2",
        LMStyle.Claude,
        datetime(2022, 12, 31),
        link="https://www.anthropic.com/index/claude-2",
    ),
    ## Claude 3 and Claude 3.5
    LanguageModel(
        "claude-3-opus-20240229",
        "Claude-3-Opus",
        LMStyle.Claude3,
        datetime(2023, 9, 1),
        link="https://www.anthropic.com/index/claude-3",
    ),
    LanguageModel(
        "claude-3-sonnet-20240229",
        "Claude-3-Sonnet",
        LMStyle.Claude3,
        datetime(2023, 9, 1),
        link="https://www.anthropic.com/index/claude-3",
    ),
    LanguageModel(
        "claude-3-5-sonnet-20240620",
        "Claude-3.5-Sonnet-20240620",
        LMStyle.Claude3,
        datetime(2024, 3, 31),
        link="https://www.anthropic.com/news/claude-3-5-sonnet",
    ),
    LanguageModel(
        "claude-3-5-sonnet-20241022",
        "Claude-3.5-Sonnet-20241022",
        LMStyle.Claude3,
        datetime(2024, 3, 31),
        link="https://www.anthropic.com/news/claude-3-5-sonnet",
    ),
    LanguageModel(
        "claude-3-haiku-20240307",
        "Claude-3-Haiku",
        LMStyle.Claude3,
        datetime(2023, 4, 30),
        link="https://www.anthropic.com/index/claude-3",
    ),
    ## Gemini
    LanguageModel(
        "gemini-1.5-pro-002",
        "Gemini-Pro-1.5-002",
        LMStyle.Gemini,
        datetime(2023, 4, 30),
        link="https://blog.google/technology/ai/gemini-api-developers-cloud",
    ),
    LanguageModel(
        "gemini-1.5-flash-002",
        "Gemini-Flash-1.5-002",
        LMStyle.Gemini,
        datetime(2023, 4, 30),
        link="https://blog.google/technology/ai/gemini-api-developers-cloud",
    ),
    LanguageModel(
        "gemini-exp-1206",
        "Gemini-Exp-1206",
        LMStyle.Gemini,
        datetime(2023, 4, 30),
        link="https://ai.google.dev/gemini-api/docs/models/experimental-models",
    ),
    LanguageModel(
        "gemini-2.0-flash-thinking-exp-1219",
        "Gemini-Flash-2.0-Thinking-12-19",
        LMStyle.GeminiThinking,
        datetime(2023, 4, 30),
        link="https://ai.google.dev/gemini-api/docs/models/experimental-models",
    ),
    LanguageModel(
        "gemini-2.0-flash-thinking-exp-01-21",
        "Gemini-Flash-2.0-Thinking-01-21",
        LMStyle.GeminiThinking,
        datetime(2023, 4, 30),
        link="https://ai.google.dev/gemini-api/docs/models/experimental-models",
    ),
    LanguageModel(
        "gemini-2.0-flash-exp",
        "Gemini-Flash-2.0-Exp",
        LMStyle.Gemini,
        datetime(2023, 4, 30),
        link="https://ai.google.dev/gemini-api/docs/models/experimental-models",
    ),
    ## Generic Base Models
    LanguageModel(
        "bigcode/starcoder2-3b",
        "StarCoder2-3b",
        LMStyle.GenericBase,
        datetime(2023, 1, 1),
        link="https://huggingface.co/bigcode/starcoder2-7b-magicoder-instruct/tree/main",
    ),
    LanguageModel(
        "bigcode/starcoder2-7b",
        "StarCoder2-7b",
        LMStyle.GenericBase,
        datetime(2023, 1, 1),
        link="https://huggingface.co/bigcode/starcoder2-7b-magicoder-instruct/tree/main",
    ),
    LanguageModel(
        "bigcode/starcoder2-15b",
        "StarCoder2-15b",
        LMStyle.GenericBase,
        datetime(2023, 1, 1),
        link="https://huggingface.co/bigcode/starcoder2-7b-magicoder-instruct/tree/main",
    ),
    LanguageModel(
        "google/codegemma-7b",
        "CodeGemma-7b-Base",
        LMStyle.GenericBase,
        datetime(2023, 1, 1),
        link="https://huggingface.co/google/codegemma-7b",
    ),
    LanguageModel(
        "google/codegemma-2b",
        "CodeGemma-2b-Base",
        LMStyle.GenericBase,
        datetime(2023, 1, 1),
        link="https://huggingface.co/google/codegemma-2b",
    ),
    LanguageModel(
        "google/gemma-7b",
        "Gemma-7b-Base",
        LMStyle.GenericBase,
        datetime(2023, 1, 1),
        link="https://huggingface.co/google/gemma-7b",
    ),
    LanguageModel(
        "google/gemma-2b",
        "Gemma-2b-Base",
        LMStyle.GenericBase,
        datetime(2023, 1, 1),
        link="https://huggingface.co/google/gemma-2b",
    ),
    ## Mistral Web
    LanguageModel(
        "mistral-large-latest",
        "Mistral-Large",
        LMStyle.MistralWeb,
        datetime(2023, 1, 1),
        link="https://mistral.ai/news/mistral-large/",
    ),
    ## Mistral OSS
    LanguageModel(
        "open-mixtral-8x22b",
        "Mixtral-8x22B-Ins",
        LMStyle.MistralWeb,
        datetime(2023, 1, 1),
        link="https://mistral.ai/news/mixtral-8x22b/",
    ),
    LanguageModel(
        "open-mixtral-8x7b",
        "Mixtral-8x7B-Ins",
        LMStyle.MistralWeb,
        datetime(2023, 1, 1),
        link="https://mistral.ai/news/mixtral-8x7b/",
    ),
    LanguageModel(
        "open-mixtral-8x7b",
        "Mixtral-8x7B-Ins",
        LMStyle.MistralWeb,
        datetime(2023, 1, 1),
        link="https://mistral.ai/news/mixtral-8x7b/",
    ),
    LanguageModel(
        "codestral-latest",
        "Codestral-Latest",
        LMStyle.MistralWeb,
        datetime(2023, 1, 1),
        link="https://mistral.ai/news/codestral/",
    ),
    ## QwQ
    LanguageModel(
        "Qwen/QwQ-32B-Preview",
        "QwQ-32B-Preview",
        LMStyle.QwQ,
        datetime(2024, 6, 30),
        link="https://huggingface.co/Qwen/QwQ-32B-Preview",
    ),
    ## Qwen 2
    LanguageModel(
        "Qwen/Qwen2-72B-Instruct",
        "Qwen2-Ins-72B",
        LMStyle.CodeQwenInstruct,
        datetime(2023, 8, 30),
        link="https://huggingface.co/Qwen/Qwen2-72B-Instruct",
    ),
    ## Qwen 2.5
    LanguageModel(
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen2.5-Ins-7B",
        LMStyle.CodeQwenInstruct,
        datetime(2023, 8, 30),
        link="https://huggingface.co/Qwen/Qwen2.5-7B-Instruct",
    ),
    LanguageModel(
        "Qwen/Qwen2.5-32B-Instruct",
        "Qwen2.5-Ins-32B",
        LMStyle.CodeQwenInstruct,
        datetime(2023, 8, 30),
        link="https://huggingface.co/Qwen/Qwen2.5-32B-Instruct",
    ),
    LanguageModel(
        "Qwen/Qwen2.5-72B-Instruct",
        "Qwen2.5-Ins-72B",
        LMStyle.CodeQwenInstruct,
        datetime(2023, 8, 30),
        link="https://huggingface.co/Qwen/Qwen2.5-72B-Instruct",
    ),
    ## Qwen 2.5-Coder
    LanguageModel(
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "Qwen2.5-Coder-Ins-7B",
        LMStyle.CodeQwenInstruct,
        datetime(2024, 6, 30),
        link="https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct",
    ),
    LanguageModel(
        "Qwen/Qwen2.5-Coder-32B-Instruct",
        "Qwen2.5-Coder-Ins-32B",
        LMStyle.CodeQwenInstruct,
        datetime(2024, 6, 30),
        link="https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct",
    ),
    LanguageModel(
        "azerogpt",
        "AzeroGPT-64b",
        LMStyle.CodeQwenInstruct,
        datetime(2024, 3, 1),
        "https://azerogpt.soundai.com",
    ),
    LanguageModel(
        "Kimi-k1.5-IOI",
        "Kimi-k1.5-IOI",
        LMStyle.CodeQwenInstruct,
        datetime(2024, 8, 1),
        "https://kimi.moonshot.cn/",
    ),
    LanguageModel(
        "Qwen/QwQ-Max-Preview",
        "QwQ-Max-Preview",
        LMStyle.QwQ,
        datetime(2024, 6, 30),
        link="https://huggingface.co/Qwen/QwQ-Max-Preview",
    ),
    LanguageModel(
        "accounts/fireworks/models/deepseek-r1",
        "DeepSeek-R1",
        LMStyle.DeepSeekR1,
        datetime(2024, 6, 30),
        link="https://huggingface.co/deepseek-ai/DeepSeek-R1",
    ),
    ## DeepSeek R1 distilled of Qwen/Llama models
    LanguageModel(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "DeepSeek-R1-Distill-Qwen-1.5B",
        LMStyle.DeepSeekR1,
        datetime(2025, 1, 20),
        link="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    ),
    LanguageModel(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "DeepSeek-R1-Distill-Qwen-7B",
        LMStyle.DeepSeekR1,
        datetime(2025, 1, 20),
        link="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    ),
    LanguageModel(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "DeepSeek-R1-Distill-Qwen-14B",
        LMStyle.DeepSeekR1,
        datetime(2025, 1, 20),
        link="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    ),
    LanguageModel(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "DeepSeek-R1-Distill-Qwen-32B",
        LMStyle.DeepSeekR1,
        datetime(2025, 1, 20),
        link="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    ),
    LanguageModel(
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "DeepSeek-R1-Distill-Llama-8B",
        LMStyle.DeepSeekR1,
        datetime(2025, 1, 20),
        link="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    ),
    LanguageModel(
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "DeepSeek-R1-Distill-Llama-70B",
        LMStyle.DeepSeekR1,
        datetime(2025, 1, 20),
        link="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    ),
]

LanguageModelStore: dict[str, LanguageModel] = {
    lm.model_name: lm for lm in LanguageModelList
}

if __name__ == "__main__":
    print(list(LanguageModelStore.keys()))
