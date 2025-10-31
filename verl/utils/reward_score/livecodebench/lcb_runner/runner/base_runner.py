import os
import json
from abc import ABC, abstractmethod

from tqdm import tqdm

from verl.utils.reward_score.livecodebench.lcb_runner.lm_styles import LanguageModel
from verl.utils.reward_score.livecodebench.lcb_runner.utils.path_utils import get_cache_path
from verl.utils.reward_score.livecodebench.lcb_runner.utils.multiprocess import run_tasks_in_parallel
from verl.utils.reward_score.livecodebench.lcb_runner.runner.scenario_router import Scenario


class BaseRunner(ABC):
    def __init__(self, args, model: LanguageModel):
        self.args = args
        self.model = model
        self.client_kwargs: dict[str | str] = {}

        if self.args.use_cache:
            self.cache_path = get_cache_path(model.model_repr, args)
            if os.path.exists(self.cache_path):
                with open(self.cache_path) as f:
                    self.cache: dict = json.load(f)
            else:
                self.cache = {}
        else:
            self.cache_path = None
            self.cache = None

    def save_cache(self):
        if self.args.use_cache:
            with open(self.cache_path, "w") as f:
                json.dump(self.cache, f, indent=4)

    # @abstractmethod
    def _run_single(self, prompt: str | list[dict[str, str]]) -> list[str]:
        pass

    @staticmethod
    def run_single(combined_args) -> list[str]:
        """
        Run the model for a single prompt and return the output
        Static method to be used in multiprocessing
        Calls the _run_single method with the combined arguments
        """
        prompt: str | list[dict[str, str]]
        cache: dict[str, str]
        call_method: callable
        prompt, cache, args, call_method = combined_args

        if isinstance(prompt, list):
            prompt_cache = json.dumps(prompt)
        elif isinstance(prompt, tuple):
            prompt_cache = prompt[0] + json.dumps(prompt[1])
        else:
            prompt_cache = prompt      

        if cache is not None and prompt_cache in cache:
            if len(cache[prompt_cache]) == args.n:
                return cache[prompt_cache]

        result = call_method(prompt)
        assert len(result) == args.n

        return result

    def run_batch(self, prompts: list[str | list[dict[str, str]]]) -> list[list[str]]:
        outputs = []
        arguments = [
            (
                prompt,
                self.cache,  ## pass the cache as argument for cache check
                self.args,  ## pass the args as argument for cache check
                self._run_single,  ## pass the _run_single method as argument because of multiprocessing
            )
            for prompt in prompts
        ]
        if self.args.multiprocess > 1:
            parallel_outputs = run_tasks_in_parallel(
                self.run_single,
                arguments,
                self.args.multiprocess,
                use_progress_bar=True,
            )
            for output in parallel_outputs:
                if output.is_success():
                    outputs.append(output.result)
                else:
                    print("Failed to run the model for some prompts")
                    print(output.status)
                    print(output.exception_tb)
                    outputs.extend([""] * self.args.n)
        else:
            outputs = [self.run_single(argument) for argument in tqdm(arguments)]

        if self.args.use_cache:
            for prompt, output in zip(prompts, outputs):
                if isinstance(prompt, list):
                    prompt_cache = json.dumps(prompt)
                elif isinstance(prompt, tuple):
                    prompt_cache = prompt[0] + json.dumps(prompt[1])
                else:
                    prompt_cache = prompt
                self.cache[prompt_cache] = output  ## save the output to cache

        return outputs

    def prompts_to_outputs(
        self, prompts: list[str | list[dict[str, str]]]
    ) -> list[list[str]]:
        if self.args.use_cache:
            outputs = []
            batch_size = self.args.cache_batch_size
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i : i + batch_size]
                batch_outputs = self.run_batch(batch)
                outputs.extend(batch_outputs)
                self.save_cache()
        else:
            outputs = self.run_batch(prompts)
        return outputs

    def run_main_repair(self, benchmark: list, format_prompt: callable) -> list[list[str]]:
        assert self.args.n == 1
        with open(
            f"output/{self.model.model_repr}/{Scenario.codegeneration}_{self.args.codegen_n}_{self.args.temperature}_eval_all.json"
        ) as f:
            check_metadata_list = json.load(f)

        outputs = [
            [None for _ in range(self.args.codegen_n)]
            for _ in range(len(benchmark))
        ]
        prompts = []
        prompt_index_to_question_idx = {}
        prompt_index_to_code_idx = {}
        count = 0

        for problem_idx, problem in enumerate(benchmark):
            for check_metadata_idx, check_metadata in enumerate(check_metadata_list):
                if problem.question_id == check_metadata['question_id']:
                    count += 1 
                    question_content = check_metadata["question_content"]
                    code_list = check_metadata["code_list"]
                    output_list = check_metadata["output_list"]
                    graded_list = check_metadata["graded_list"]
                    metadata = check_metadata["metadata"]
                    for code_idx in range(len(code_list)):
                        prompt = format_prompt(
                            question_content,
                            self.model.model_style,
                            code_list[code_idx],
                            graded_list[code_idx],
                            metadata[code_idx],
                        )
                        if prompt == "":
                            outputs[problem_idx][code_idx] = output_list[code_idx]
                            continue
                        prompts.append(prompt)
                        prompt_index_to_question_idx[len(prompts) - 1] = problem_idx
                        prompt_index_to_code_idx[len(prompts) - 1] = code_idx

        assert len(benchmark)==count, f"{len(benchmark)=}!={count=}"

        prompt_outputs = self.prompts_to_outputs(prompts)
        for prompt_idx, output in enumerate(prompt_outputs):
            question_idx = prompt_index_to_question_idx[prompt_idx]
            code_idx = prompt_index_to_code_idx[prompt_idx]
            outputs[question_idx][code_idx] = output

        return outputs

    def run_main(self, benchmark: list, format_prompt: callable) -> list[list[str]]:
        if self.args.scenario == Scenario.selfrepair:
            return self.run_main_repair(benchmark, format_prompt)

        prompts = [
            format_prompt(problem, self.model.model_style) for problem in benchmark
        ]
        outputs = self.prompts_to_outputs(prompts)
        return outputs
