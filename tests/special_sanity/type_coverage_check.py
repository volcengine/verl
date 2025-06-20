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
import ast
import linecache
import subprocess
from pathlib import Path
from typing import List, Set, Tuple


def get_changed_files() -> List[Path]:
    result = subprocess.run(["git", "diff", "--name-only", "--diff-filter=AM", "origin/main...HEAD"], stdout=subprocess.PIPE, text=True)
    return [Path(f) for f in result.stdout.splitlines() if f.endswith(".py")]


def get_changed_lines(file_path: Path) -> Set[int]:
    result = subprocess.run(
        ["git", "diff", "-U0", "origin/main...HEAD", "--", str(file_path)],
        stdout=subprocess.PIPE,
        text=True,
    )
    lines: Set[int] = set()
    for line in result.stdout.splitlines():
        if line.startswith("@@"):
            for part in line.split():
                try:
                    if part.startswith("+") and "," in part:
                        start, count = map(int, part[1:].split(","))
                        lines.update(range(start, start + count))
                    elif part.startswith("+") and "," not in part:
                        lines.add(int(part[1:]))
                except Exception:
                    # (vermouth1992) There are many edge cases here because + can be in the changed program
                    pass
    return lines


CHECK_SUCCESS = 0
CHECK_WARNING = 1
CHECK_FAILURE = -1


def has_type_annotations(node: ast.AST) -> int:
    if isinstance(node, ast.FunctionDef):
        has_ann = all(arg.annotation is not None for arg in node.args.args if arg.arg != "self") and node.returns is not None
        return has_ann
    elif isinstance(node, ast.Assign):
        if not isinstance(node.value, ast.Constant):
            return CHECK_WARNING
    return CHECK_SUCCESS


def check_file(file_path: Path, changed_lines: Set[int]) -> Tuple[int, int, List[Tuple[Path, int, str]], List[Tuple[Path, int, str]]]:
    with open(file_path) as f:
        source: str = f.read()
    tree = ast.parse(source, filename=str(file_path))

    annotated = 0
    total = 0
    warning_lines: List[Tuple[Path, int, str]] = []
    failure_lines: List[Tuple[Path, int, str]] = []

    for node in ast.walk(tree):
        if hasattr(node, "lineno") and node.lineno in changed_lines:
            if isinstance(node, (ast.FunctionDef, ast.Assign, ast.AnnAssign)):
                total += 1
                result = has_type_annotations(node)
                if result == CHECK_SUCCESS or result == CHECK_WARNING:
                    annotated += 1
                    if result == CHECK_WARNING:
                        warning_lines.append((file_path, node.lineno, linecache.getline(str(file_path), node.lineno).strip()))
                else:
                    source_line = linecache.getline(str(file_path), node.lineno).strip()
                    failure_lines.append((file_path, node.lineno, source_line))

    return annotated, total, warning_lines, failure_lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.2, help="Minimum ratio of annotated lines required (0.0 - 1.0)")
    args = parser.parse_args()

    total_changed = 0
    total_annotated = 0
    all_warnings: List[Tuple[Path, int, str]] = []
    all_failures: List[Tuple[Path, int, str]] = []

    for fpath in get_changed_files():
        changed_lines = get_changed_lines(fpath)
        annotated, total, warning_lines, failure_lines = check_file(fpath, changed_lines)
        total_annotated += annotated
        total_changed += total
        all_warnings.extend(warning_lines)
        all_failures.extend(failure_lines)

    ratio = (total_annotated / total_changed) if total_changed else 1.0

    print(f"🔍 Type coverage on changed lines: {total_annotated}/{total_changed} = {ratio:.2%}")

    if all_warnings:
        print("\n⚠️ Suggest Improve: Lines missing type annotations:\n")
        for fname, lineno, line in all_failures:
            print(f"{fname}:{lineno}: {line}")

    if all_failures:
        print("\n⚠️ [ERROR] Lines missing type annotations:\n")
        for fname, lineno, line in all_failures:
            print(f"{fname}:{lineno}: {line}")

    if ratio < args.threshold:
        raise Exception(f"\n❌ Type coverage below threshold ({args.threshold:.0%}).")
    else:
        print("\n✅ Type annotation coverage acceptable.")


if __name__ == "__main__":
    main()
