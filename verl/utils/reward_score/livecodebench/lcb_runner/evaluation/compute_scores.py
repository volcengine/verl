import json
import argparse
import numpy as np
from datetime import datetime

from verl.utils.reward_score.livecodebench.lcb_runner.lm_styles import LanguageModelStore
from verl.utils.reward_score.livecodebench.lcb_runner.evaluation.pass_k_utils import (
    estimate_pass_at_k,
    compute_metrics_from_results,
)
from verl.utils.reward_score.livecodebench.lcb_runner.utils.scenarios import Scenario
from verl.utils.reward_score.livecodebench.lcb_runner.utils.path_utils import get_eval_all_output_path


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-3.5-turbo-0301",
        help="Name of the model to use matching `lm_styles.py`",
    )
    parser.add_argument(
        "--scenario",
        type=Scenario,
        default=Scenario.codegeneration,
        help="Type of scenario to run",
    )
    parser.add_argument(
        "--n", type=int, default=10, help="Number of samples to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.2, help="Temperature for sampling"
    )

    parser.add_argument(
        "--eval_all_file",
        type=str,
        default=None,
        help="Alternative way to provide the evaluation file",
    )

    parser.add_argument(
        "--start_date",
        type=str,
        default=None,
        help="Start date for the contest to filter the evaluation file (format - YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default=None,
        help="End date for the contest to filter the evaluation file (format - YYYY-MM-DD)",
    )

    parser.add_argument(
        "--platform",
        type=str,
        default=None,
        help="Platform to filter the evaluation file",
    )

    args = parser.parse_args()

    if args.eval_all_file is None:
        model = LanguageModelStore[args.model]
        args.eval_all_file = get_eval_all_output_path(model, args)

    return args


def compute_scores(args):
    with open(args.eval_all_file, "r") as f:
        results = json.load(f)

    for res in results:
        res["contest_date"] = datetime.fromisoformat(res["contest_date"])

    if args.start_date is not None:
        args.start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        results = [
            result for result in results if args.start_date <= result["contest_date"]
        ]

    if args.end_date is not None:
        args.end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
        results = [
            result for result in results if result["contest_date"] <= args.end_date
        ]

    if args.platform is not None:
        results = [result for result in results if result["platform"] == args.platform]

    print(len(results))
    totals = [len(x["graded_list"]) for x in results]
    corrects = [sum(x["graded_list"]) for x in results]

    easy_totals = [len(x["graded_list"]) for x in results if x["difficulty"] == "easy"]
    med_totals = [len(x["graded_list"]) for x in results if x["difficulty"] == "medium"]
    hard_totals = [len(x["graded_list"]) for x in results if x["difficulty"] == "hard"]
    easy_corrects = [sum(x["graded_list"]) for x in results if x["difficulty"] == "easy"]
    med_corrects = [sum(x["graded_list"]) for x in results if x["difficulty"] == "medium"]
    hard_corrects = [sum(x["graded_list"]) for x in results if x["difficulty"] == "hard"]
    for k in [1, 5, 10, 25, 50, 100, 150, 200]:
        print(
            f"Pass@{k} = ",
            estimate_pass_at_k(totals, corrects, k).mean(),
            # np.array(
            #     [estimate_pass_at_k(t, c, k) for t, c in zip(totals, corrects)]
            # ).mean(),
        )
        print(
            f"Easy Pass@{k} = ",
            estimate_pass_at_k(easy_totals, easy_corrects, k).mean(),
        )
        print(
            f"Medium Pass@{k} = ",
            estimate_pass_at_k(med_totals, med_corrects, k).mean(),
        )
        print(
            f"Hard Pass@{k} = ",
            estimate_pass_at_k(hard_totals, hard_corrects, k).mean(),
        )


    pass_1_list = [result["pass@1"] for result in results]
    print(f"Pass@1: {sum(pass_1_list) / len(pass_1_list)}")

    easy_pass_1_list = [
        result["pass@1"]
        for result in results
        if "difficulty" in result and result["difficulty"] == "easy"
    ]
    if len(easy_pass_1_list) > 0:
        print(f"Easy Pass@1: {sum(easy_pass_1_list) / len(easy_pass_1_list)}")

    medium_pass_1_list = [
        result["pass@1"]
        for result in results
        if "difficulty" in result and result["difficulty"] == "medium"
    ]
    if len(medium_pass_1_list) > 0:
        print(f"Medium Pass@1: {sum(medium_pass_1_list) / len(medium_pass_1_list)}")

    hard_pass_1_list = [
        result["pass@1"]
        for result in results
        if "difficulty" in result and result["difficulty"] == "hard"
    ]
    if len(hard_pass_1_list) > 0:
        print(f"Hard Pass@1: {sum(hard_pass_1_list) / len(hard_pass_1_list)}")


if __name__ == "__main__":
    compute_scores(get_parser())
