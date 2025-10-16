import os
import json

from verl.utils.reward_score.livecodebench.lcb_runner.runner.parser import get_args
from verl.utils.reward_score.livecodebench.lcb_runner.utils.scenarios import Scenario
from verl.utils.reward_score.livecodebench.lcb_runner.utils.path_utils import get_output_path
from verl.utils.reward_score.livecodebench.lcb_runner.evaluation import extract_instance_results
from verl.utils.reward_score.livecodebench.lcb_runner.runner.scenario_router import (
    build_prompt_benchmark,
    sort_and_extract_save_results,
    get_metrics,
)


def main():
    args = get_args()

    benchmark, _ = build_prompt_benchmark(args)

    with open(args.custom_output_file, "r") as f:
        custom_outputs = json.load(f)
        from datetime import datetime
        if args.start_date is not None:
            start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
            custom_outputs = [custom_output for custom_output in custom_outputs if datetime.strptime(custom_output["contest_date"],  "%Y-%m-%dT%H:%M:%S") >= start_date]
        if args.end_date is not None:
            end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
            custom_outputs = [custom_output for custom_output in custom_outputs if datetime.strptime(custom_output["contest_date"], "%Y-%m-%dT%H:%M:%S") <= end_date]

        assert isinstance(custom_outputs, list)
        assert len(custom_outputs) == len(benchmark), f"{len(custom_outputs)} != {len(benchmark)}"
        if isinstance(custom_outputs[0], list):
            ## custom outputs must list[list[str]]
            ## list of extracted outputs per question
            ## sorted by the benchmark question_id, test_id, id depending on the scenario

            assert all(
                isinstance(custom_output, list) for custom_output in custom_outputs
            )
        elif isinstance(custom_outputs[0], dict):
            ## custom outputs must list[dict[str, Any]]
            ## list of extracted outputs per question
            ## for codegeneration and selfrepair scenario -- `code_list` and `question_id` are required
            ## for testoutputprediction -- `pred_list`, `question_id`, `test_id` are required 
            ## for codeexecution -- `pred_list`, `id` are required 
            ## code_list/pred_list is a list of extracted answers (code or assertions) for a question

            assert all(
                isinstance(custom_output, dict) for custom_output in custom_outputs
            )
            if args.scenario in [Scenario.codegeneration, Scenario.selfrepair]:
                custom_outputs = [
                    custom_output["code_list"]
                    for custom_output in sorted(
                        custom_outputs, key=lambda x: str(x["question_id"])
                    )
                ]
            elif args.scenario == Scenario.testoutputprediction:
                custom_outputs = [
                    custom_output['pred_list']
                    for custom_output in sorted(
                        custom_outputs, key=lambda x: (str(x["question_id"]), str(x['test_id']))
                    )
                ]
            elif args.scenario == Scenario.codeexecution:
                custom_outputs = [
                    custom_output['pred_list']
                    for custom_output in sorted(
                        custom_outputs, key=lambda x: int(x.id.split("_")[1])
                    )
                ]

    save_results = [
        instance.insert_output(custom_output, custom_output)
        for instance, custom_output in zip(benchmark, custom_outputs)
    ]

    save_results, combined_results = sort_and_extract_save_results(
        args.scenario, save_results
    )

    metrics = get_metrics(args.scenario, args, benchmark, combined_results)
    graded = extract_instance_results(metrics[1])

    if args.scenario == Scenario.codegeneration:
        metadatas = metrics[2]
        save_eval_results = [
            instance.insert_output_evaluation(
                outputs_list, extracted_list, graded_list, metadata=meta
            )
            for instance, (outputs_list, extracted_list), graded_list, meta in zip(
                benchmark, combined_results, graded, metadatas
            )
        ]
    else:
        save_eval_results = [
            instance.insert_output_evaluation(
                outputs_list, extracted_list, graded_list
            )
            for instance, (outputs_list, extracted_list), graded_list in zip(
                benchmark, combined_results, graded
            )
        ]
    

    if args.custom_output_save_name is None:
        output_path = args.custom_output_file[:-5] + f"_{args.scenario.value}_output.json"
    else:
        output_path = get_output_path(args.custom_output_save_name, args)

    with open(output_path, "w") as f:
        json.dump(save_results, f, indent=4)


    with open(output_path.replace(".json", "_eval.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    with open(output_path.replace(".json", "_eval_all.json"), "w") as f:
        json.dump(save_eval_results, f, indent=4)

if __name__ == "__main__":
    main()
