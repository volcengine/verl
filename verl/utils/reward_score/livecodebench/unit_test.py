from verl.utils.reward_score.livecodebench.lcb_runner.benchmarks.code_generation import load_code_generation_dataset
from verl.utils.reward_score.livecodebench.lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics
from verl.utils.reward_score.livecodebench.lcb_runner.evaluation.pass_k_utils import extract_instance_results
from verl.utils.reward_score.livecodebench.lcb_runner.runner.scenario_router import Scenario, sort_and_extract_save_results
from verl.utils.reward_score.livecodebench.lcb_runner.utils.extraction_utils import extract_code


def prepare_unit_test_data(release_version="release_latest", start_date="2024-08-01", end_date="2025-02-01"):
    benchmark = load_code_generation_dataset(release_version, start_date, end_date)
    benchmark = sorted(benchmark, key=lambda x: x.question_id)
    id_to_question = {x.question_id: x for x in benchmark}
    return id_to_question


def run_unit_test(benchmark, combined_results, num_process_evaluate=12, timeout=6):
    eval_samples = [instance.get_evaluation_sample() for instance in benchmark]
    generations = [extracted for _, extracted in combined_results]
    metrics = codegen_metrics(
        eval_samples,
        generations,
        num_process_evaluate=num_process_evaluate,
        timeout=timeout,
    )
    graded = extract_instance_results(metrics[1])
    return graded[0][0]


def lcb_compute_score(custom_outputs, benchmark):
    # Extract the code from the outputs
    for custom_output in custom_outputs:
        custom_output["code_list"] = [
            extract_code(output, None) for output in custom_output["output_list"]
        ]

    # Assert the format of the outputs
    assert isinstance(custom_outputs, list)
    assert len(custom_outputs) == len(benchmark), (
        f"{len(custom_outputs)} != {len(benchmark)}"
    )
    if isinstance(custom_outputs[0], list):
        assert all(isinstance(custom_output, list) for custom_output in custom_outputs)
    elif isinstance(custom_outputs[0], dict):
        assert all(isinstance(custom_output, dict) for custom_output in custom_outputs)
        custom_outputs = [
            custom_output["code_list"]
            for custom_output in sorted(
                custom_outputs, key=lambda x: str(x["question_id"])
            )
        ]

    # Insert the outputs into the benchmark
    save_results = [
        instance.insert_output(custom_output, custom_output)
        for instance, custom_output in zip(benchmark, custom_outputs)
    ]
    save_results, combined_results = sort_and_extract_save_results(
        Scenario.codegeneration, save_results
    )

    # Run unit test
    graded = run_unit_test(benchmark, combined_results)
    return graded
