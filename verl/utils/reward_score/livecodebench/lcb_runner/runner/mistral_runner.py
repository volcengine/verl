import os
from time import sleep

try:
    from mistralai.client import MistralClient
except ImportError as e:
    pass

from verl.utils.reward_score.livecodebench.lcb_runner.runner.base_runner import BaseRunner


class MistralRunner(BaseRunner):
    client = MistralClient(
        api_key=os.environ["MISTRAL_API_KEY"],
    )

    def __init__(self, args, model):
        super().__init__(args, model)
        self.client_kwargs: dict[str | str] = {
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "top_p": args.top_p,
        }

    def _run_single(self, prompt: list[dict[str, str]]) -> list[str]:

        def __run_single(counter):
            try:
                response = self.client.chat(
                    messages=prompt,
                    **self.client_kwargs,
                )
                content = response.choices[0].message.content
                return content
            except Exception as e:
                print("Exception: ", repr(e), "Sleeping for 20 seconds...")
                sleep(20 * (11 - counter))
                counter = counter - 1
                if counter == 0:
                    print(f"Failed to run model for {prompt}!")
                    print("Exception: ", repr(e))
                    raise e
                return __run_single(counter)

        outputs = []
        try:
            for _ in range(self.args.n):
                outputs.append(__run_single(10))
        except Exception as e:
            raise e

        return outputs
