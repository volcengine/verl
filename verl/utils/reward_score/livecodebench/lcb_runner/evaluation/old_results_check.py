import numpy as np
import json
from verl.utils.reward_score.livecodebench.lcb_runner.benchmarks import load_generation_dataset, CodeGenerationProblem
from verl.utils.reward_score.livecodebench.lcb_runner.evaluation import codegen_metrics


dataset = load_generation_dataset()

dataset = sorted(dataset, key=lambda x: x.question_id)


def check_model(model_key):
    path = f"/home/naman/Repos/LiveCodeBench/run_models_outputs/{model_key}/chat_0.2_checked.json"
    with open(path) as f:
        old_results = json.load(f)
    old_results = sorted(old_results, key=lambda x: x["question_id"])
    assert old_results[0]["question_id"] == dataset[0].question_id

    def debug(idx):
        codegen_metrics(
            [dataset[idx].get_evaluation_sample()],
            [old_results[idx]["code_list"][:1]],
            debug=True,
        )

    def run(idx):
        return codegen_metrics(
            [dataset[idx].get_evaluation_sample()],
            [old_results[idx]["code_list"]],
        )

    debug(380)
    exit()
    # debug(196)
    # debug(352)

    metrics = codegen_metrics(
        [d.get_evaluation_sample() for d in dataset],
        [r["code_list"] for r in old_results],
        num_process_evaluate=12,
    )
    old_pass1 = np.mean([np.mean(r["pass1_list"]) for r in old_results])

    print(old_pass1)
    print(metrics[0]["pass@1"])

    for idx in range(400):
        old_pass1 = np.mean(old_results[idx]["pass1_list"])
        new_pass1 = metrics[0]["detail"]["pass@1"][idx]
        if not abs(old_pass1 - new_pass1) < 1e-4:
            print(idx, old_pass1, new_pass1)


# model_key = "GPT-4-Turbo-1106"
# check_model(model_key)

model_key = "Claude-3-Opus"
check_model(model_key)

model_key = "GPT-4-0613"
check_model(model_key)

model_key = "Mistral-Large"
check_model(model_key)

model_key = "Claude-3-Sonnet"
check_model(model_key)

model_key = "GPT-3.5-Turbo-0301"
check_model(model_key)

model_key = "Gemini-Pro"
check_model(model_key)
