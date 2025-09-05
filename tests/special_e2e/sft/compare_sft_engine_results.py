import os
import json
import torch

def get_result(file):
    file = os.path.expanduser(file)
    result = []
    with open(file, "r") as f:
        lines = f.readlines()
        for line in lines:
            result.append(json.loads(line))
    return result


def compare_results(golden_results, other_result):
    golden_loss = golden_results[0]['data']['train/loss']
    golden_grad_norm = golden_results[0]['data']['train/grad_norm']

    loss = other_result[0]['data']['train/loss']
    grad_norm = other_result[0]['data']['train/grad_norm']

    torch.testing.assert_close(golden_loss, loss, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(golden_grad_norm, grad_norm, atol=1e-4, rtol=1e-2)


if __name__ == "__main__":
    golden_results = get_result("~/verl/test/log/golden.jsonl")

    # get all other results
    other_results = {}
    # walk through all files in ~/verl/test/log
    for file in os.listdir(os.path.expanduser("~/verl/test/log/verl_sft_test")):
        if file.endswith(".jsonl"):
            other_results[file] = get_result(os.path.join(os.path.expanduser("~/verl/test/log/verl_sft_test"), file))

    # # compare results
    for file, other_result in other_results.items():
        print(f'compare results {file}')
        compare_results(golden_results, other_result)

    print("All results are close to golden results")