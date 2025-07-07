from verl.utils.reward_score.livecodebench.lcb_runner.lm_styles import LMStyle, LanguageModel


def build_runner(args, model: LanguageModel):
    if model.model_style == LMStyle.OpenAIChat:
        from verl.utils.reward_score.livecodebench.lcb_runner.runner.oai_runner import OpenAIRunner

        return OpenAIRunner(args, model)
    if model.model_style in [LMStyle.OpenAIReason, LMStyle.OpenAIReasonPreview]:
        from verl.utils.reward_score.livecodebench.lcb_runner.runner.oai_runner import OpenAIRunner

        return OpenAIRunner(args, model)
    if model.model_style in [LMStyle.Gemini, LMStyle.GeminiThinking]:
        from verl.utils.reward_score.livecodebench.lcb_runner.runner.gemini_runner import GeminiRunner

        return GeminiRunner(args, model)
    if model.model_style == LMStyle.Claude3:
        from verl.utils.reward_score.livecodebench.lcb_runner.runner.claude3_runner import Claude3Runner

        return Claude3Runner(args, model)
    if model.model_style == LMStyle.Claude:
        from verl.utils.reward_score.livecodebench.lcb_runner.runner.claude_runner import ClaudeRunner

        return ClaudeRunner(args, model)
    if model.model_style == LMStyle.MistralWeb:
        from verl.utils.reward_score.livecodebench.lcb_runner.runner.mistral_runner import MistralRunner

        return MistralRunner(args, model)
    if model.model_style == LMStyle.CohereCommand:
        from verl.utils.reward_score.livecodebench.lcb_runner.runner.cohere_runner import CohereRunner

        return CohereRunner(args, model)
    if model.model_style == LMStyle.DeepSeekAPI:
        from verl.utils.reward_score.livecodebench.lcb_runner.runner.deepseek_runner import DeepSeekRunner

        return DeepSeekRunner(args, model)
    if model.model_style == LMStyle.DeepSeekAPI:
        from verl.utils.reward_score.livecodebench.lcb_runner.runner.deepseek_runner import DeepSeekRunner

        return DeepSeekRunner(args, model)
    if "/fireworks/" in model.model_name:
        from verl.utils.reward_score.livecodebench.lcb_runner.runner.fireworks_runner import FireWorksRunner

        return FireWorksRunner(args, model)
    elif model.model_style in []:
        raise NotImplementedError(
            f"Runner for language model style {model.model_style} not implemented yet"
        )
    else:
        from verl.utils.reward_score.livecodebench.lcb_runner.runner.vllm_runner import VLLMRunner

        return VLLMRunner(args, model)
