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
from pathlib import Path
import linecache

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
            # Parse diff hunk header like: @@ -10,0 +11,2 @@
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
            if arg.arg != "self"  # ignore self
        ) and node.returns is not None
        return has_ann
    elif isinstance(node, ast.AnnAssign):
        return node.annotation is not None
    elif isinstance(node, ast.Assign):
        return False  # plain assignment has no type info
    return True

def check_file(file_path, changed_lines):
    with open(file_path) as f:
        source = f.read()
    tree = ast.parse(source, filename=str(file_path))

    failures = []

    for node in ast.walk(tree):
        if hasattr(node, "lineno") and node.lineno in changed_lines:
            if isinstance(node, (ast.FunctionDef, ast.Assign, ast.AnnAssign)):
                if not has_type_annotations(node):
                    source_line = linecache.getline(str(file_path), node.lineno).strip()
                    failures.append((file_path, node.lineno, source_line))

    return failures

def main():
    failed = []
    for fpath in get_changed_files():
        changed_lines = get_changed_lines(fpath)
        failed += check_file(fpath, changed_lines)

    if failed:
        print("❌ Missing type annotations on changed lines:\n")
        for fname, lineno, line in failed:
            print(f"{fname}:{lineno}: {line}")
        sys.exit(1)
    else:
        print("✅ All changed lines have type annotations.")

if __name__ == "__main__":
    main()
