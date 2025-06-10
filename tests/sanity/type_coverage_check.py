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

import subprocess
import sys
import re
from typing import List, Tuple, Optional

TYPE_HINT_REGEX = re.compile(r"(->|:\s*[\w\[\], ]+)")

def get_changed_python_lines() -> List[Tuple[str, str]]:
    """
    Get the list of added or modified lines in Python files from the PR diff.
    Compares current HEAD against the merge base with 'origin/main'.
    """
    # Determine base branch (main) for the current PR
    base_result = subprocess.run(
        ["git", "merge-base", "HEAD", "origin/main"],
        capture_output=True,
        text=True,
        check=True
    )
    base_commit = base_result.stdout.strip()

    # Get the diff from base commit to HEAD
    diff_result = subprocess.run(
        ["git", "diff", "--unified=0", base_commit, "HEAD"],
        capture_output=True,
        text=True,
        check=True
    )

    diff_lines = diff_result.stdout.splitlines()
    changed_lines: List[Tuple[str, str]] = []
    current_file: Optional[str] = None

    for line in diff_lines:
        if line.startswith("+++ b/") and line.endswith(".py"):
            current_file = line[6:]
        elif line.startswith("@@") and current_file:
            continue
        elif line.startswith("+") and not line.startswith("+++") and current_file:
            changed_lines.append((current_file, line[1:].strip()))

    return changed_lines

def compute_annotation_ratio(changed_lines: List[Tuple[str, str]]) -> Tuple[int, int]:
    total = 0
    annotated = 0
    for _, line in changed_lines:
        if line and not line.startswith("#"):
            total += 1
            if TYPE_HINT_REGEX.search(line):
                annotated += 1
    return annotated, total

def main() -> None:
    changed_lines = get_changed_python_lines()
    print(f"Changed lines:\n{changed_lines}", flush=True)
    annotated, total = compute_annotation_ratio(changed_lines)

    threshold = 0.3  # 30%
    print(f"📊 Type-annotated lines: {annotated} / {total}")

    if total == 0:
        print("ℹ️ No Python lines changed in this PR.")
        sys.exit(0)

    ratio = annotated / total
    if ratio >= threshold:
        print("✅ Type annotation threshold met.")
        sys.exit(0)
    else:
        print(f"❌ Type annotation threshold not met. Required: {threshold:.0%}, Found: {ratio:.0%}")
        sys.exit(1)

if __name__ == "__main__":
    main()
