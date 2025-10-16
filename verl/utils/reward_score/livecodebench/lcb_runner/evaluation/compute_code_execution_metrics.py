import numpy as np
from concurrent.futures import ProcessPoolExecutor
import tqdm

from verl.utils.reward_score.livecodebench.lcb_runner.evaluation.utils_execute import BASE_IMPORTS, check_correctness

def evaluate_score(args) -> list[bool]:
    gs, (c, i, o) = args

    execution_results = []
    for g in gs:
        if i in g:
            pass
        else:
            code_to_execute = f"{BASE_IMPORTS}\n{c}\nassert {o} == {g}"
            execution_results.append(check_correctness(code_to_execute, 3))
    if len(execution_results) == 0:
        execution_results = [False] * len(gs)
    return execution_results

def pass_at_k(n, c, k):
    if n - c < k: return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

def code_execution_metrics(
    samples,
    generations,
):
    # execute the code
    references = [(doc["code"], doc["input"], doc["output"]) for doc in samples]
    with ProcessPoolExecutor() as executor:
        args_list = zip(generations, references)
        results = executor.map(evaluate_score, args_list)
    all_results = list(results)

    # serial version
    # all_results = []
    # for i in range(len(generations)):
    #     generation = generations[i]
    #     result = evaluate_score([generation, references[i]])
    #     all_results.append(result)

    # compute pass@1
    pass_at_1s = []
    for execution_result in all_results:
        c, n = execution_result.count(True), len(execution_result)
        pass_at_1s.append(pass_at_k(n, c, 1))
    metrics = {"pass@1": sum(pass_at_1s) / len(pass_at_1s) * 100}

    results = {}
    for i, r in enumerate(all_results):
        r_new = []
        for _r in r:
            r_new.append([_r])
        results[i] = r_new
    return [metrics, results]
