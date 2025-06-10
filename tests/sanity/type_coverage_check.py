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

import ast
import subprocess
import sys
import linecache
import argparse
from pathlib import Path

def get_changed_files():
    result = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=AM", "origin/main...HEAD"],
        stdout=subprocess.PIPE,
        text=True
    )
    return [Path(f) for f in result.stdout.splitlines() if f.endswith(".py")]

def get_changed_lines(file_path):
    result = subprocess.run(
        ["git", "diff", "-U0", "origin/main...HEAD", "--", str(file_path)],
        stdout=subprocess.PIPE,
        text=True,
    )
    lines = []
    for line in result.stdout.splitlines():
        if line.startswith("@@"):
            for part in line.split():
                if part.startswith("+") and "," in part:
                    start, count = map(int, part[1:].split(","))
                    lines.extend(range(start, start + count))
                elif part.startswith("+") and "," not in part:
                    lines.append(int(part[1:]))
    return set(lines)

def has_type_annotations(node):
    if isinstance(node, ast.FunctionDef):
        has_ann = all(
            arg.annotation is not None for arg in node.args.args
            if arg.arg != "self"
        ) and node.returns is not None
        return has_ann
    elif isinstance(node, ast.AnnAssign):
        return node.annotation is not None
    elif isinstance(node, ast.Assign):
        return False
    return True

def check_file(file_path, changed_lines):
    with open(file_path) as f:
        source = f.read()
    tree = ast.parse(source, filename=str(file_path))

    annotated, total = 0, 0
    failures = []

    for node in ast.walk(tree):
        if hasattr(node, "lineno") and node.lineno in changed_lines:
            if isinstance(node, (ast.FunctionDef, ast.Assign, ast.AnnAssign)):
                total += 1
                if has_type_annotations(node):
                    annotated += 1
                else:
                    source_line = linecache.getline(str(file_path), node.lineno).strip()
                    failures.append((file_path, node.lineno, source_line))

    return annotated, total, failures

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.2,
                        help="Minimum ratio of annotated lines required (0.0–1.0)")
    args = parser.parse_args()

    total_changed = 0
    total_annotated = 0
    all_failures = []

    for fpath in get_changed_files():
        changed_lines = get_changed_lines(fpath)
        annotated, total, failures = check_file(fpath, changed_lines)
        total_annotated += annotated
        total_changed += total
        all_failures.extend(failures)

    ratio = (total_annotated / total_changed) if total_changed else 1.0

    print(f"🔍 Type coverage on changed lines: {total_annotated}/{total_changed} = {ratio:.2%}")
    if ratio < args.threshold:
        print(f"\n❌ Type coverage below threshold ({args.threshold:.0%}). Missing annotations:\n")
        for fname, lineno, line in all_failures:
            print(f"{fname}:{lineno}: {line}")
        sys.exit(1)
    else:
        print("✅ Type annotation coverage acceptable.")

if __name__ == "__main__":
    main()
