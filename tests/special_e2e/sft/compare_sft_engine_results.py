# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os

import torch


def get_result(file):
    file = os.path.expanduser(file)
    result = []
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
            result.append(json.loads(line))
    return result


def compare_results(golden_results, other_result, loss_only):
    # result[-1] is val loss, check last training loss/grad_norm is more strict
    golden_loss = golden_results[-2]["data"]["train/loss"]
    golden_grad_norm = golden_results[-2]["data"]["train/grad_norm"]

    loss = other_result[-2]["data"]["train/loss"]
    grad_norm = other_result[-2]["data"]["train/grad_norm"]

    torch.testing.assert_close(golden_loss, loss, atol=1e-2, rtol=1e-2)
    if not loss_only:
        torch.testing.assert_close(golden_grad_norm, grad_norm, atol=1e-4, rtol=1e-2)


def show_results(golden_results, other_results):
    print(f"{'File':<30} {'Loss':<15} {'Grad Norm':<15}")
    print("=" * 60)

    for i in range(len(golden_results) - 1):
        golden_loss = golden_results[i]["data"]["train/loss"]
        golden_grad_norm = golden_results[i]["data"]["train/grad_norm"]
        print(f"{'golden.jsonl':<30} {golden_loss:<15.6f} {golden_grad_norm:<15.6f}")

        for file, result in other_results.items():
            loss = result[i]["data"]["train/loss"]
            grad_norm = result[i]["data"]["train/grad_norm"]
            print(f"{file:<30} {loss:<15.6f} {grad_norm:<15.6f}")


def main(sub_dir, method, loss_only):
    golden_results = get_result("~/verl/test/log/golden.jsonl")

    # get all other results
    other_results = {}
    # walk through all files in ~/verl/test/log
    for file in os.listdir(os.path.expanduser(f"~/verl/test/log/{sub_dir}")):
        if file.endswith(".jsonl"):
            other_results[file] = get_result(os.path.join(os.path.expanduser(f"~/verl/test/log/{sub_dir}"), file))

    if method == "show":
        show_results(golden_results, other_results)
    elif method == "compare":
        # compare results
        for file, other_result in other_results.items():
            print(f"compare results {file}")
            compare_results(golden_results, other_result, loss_only)
        print("All results are close to golden results")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare or show SFT engine results")
    parser.add_argument("--sub_dir", type=str, default="verl_sft_test", help="Subdirectory under ~/verl/test/log/")
    parser.add_argument("--loss_only", default=False, action="store_true", help="only test loss")
    parser.add_argument(
        "--method",
        type=str,
        choices=["compare", "show"],
        default="compare",
        help="Method to use: 'compare' to compare results, 'show' to display all values",
    )

    args = parser.parse_args()
    main(args.sub_dir, args.method, args.loss_only)
