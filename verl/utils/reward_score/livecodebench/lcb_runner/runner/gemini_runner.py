import os
from time import sleep

try:
    from google import genai
    from google.genai.types import GenerateContentConfigDict, ThinkingConfig
except ImportError as e:
    pass

from verl.utils.reward_score.livecodebench.lcb_runner.runner.base_runner import BaseRunner
from verl.utils.reward_score.livecodebench.lcb_runner.lm_styles import LMStyle


class GeminiRunner(BaseRunner):
    client = genai.Client(
        api_key=os.getenv("GOOGLE_API_KEY"), http_options={"api_version": "v1alpha"}
    )
    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE",
        },
    ]

    def __init__(self, args, model):
        super().__init__(args, model)
        self.args = args
        self.model = model
        if self.model.model_style == LMStyle.GeminiThinking:
            self.generation_config = GenerateContentConfigDict(
                # candidate_count=args.n,
                # temperature=0.7,
                # top_p=0.95,
                # top_k=64,
                # max_output_tokens=65536,
                safety_settings=GeminiRunner.safety_settings,
                thinking_config=ThinkingConfig(include_thoughts=True),
            )
            print("GeminiThinking model")
        else:
            self.generation_config = GenerateContentConfigDict(
                max_output_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                safety_settings=GeminiRunner.safety_settings,
                candidate_count=args.n,
            )

    def _run_single(self, prompt: str) -> list[str]:

        try:
            outputs = self.client.models.generate_content(
                model=self.model.model_name,
                contents=prompt,
                config=self.generation_config,
            ).candidates
        except Exception as e:
            print("Exception: ", repr(e))
            print("Sleeping for 30 seconds...")
            print("Consider reducing the number of parallel processes.")
            return self._run_single(prompt)

        new_outputs = []
        for output in outputs:
            try:
                texts = [part.text for part in output.content.parts]
                texts = [
                    "## Part " + str(i) + "\n" + text for i, text in enumerate(texts)
                ]
                text = "\n\n\n".join(texts)
                if text == "":
                    print("Empty text for output")
                    print(output.__dict__)
                new_outputs.append(text)
            except Exception as e:
                print("Cannot extract text exception: ", repr(e))
                print(output.__dict__)
                new_outputs.append("")
        outputs = new_outputs

        return outputs
