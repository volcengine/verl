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

# Basic regex for detecting Python type annotations (both return and argument types)
TYPE_HINT_REGEX = re.compile(r"(->|:\s*[\w\[\], ]+)")

def get_changed_lines() -> List[Tuple[str, str]]:
    """
    Returns a list of (filename, line) tuples for added lines in Python files changed in the PR.
    """
    result: subprocess.CompletedProcess[str] = subprocess.run(
        ["git", "diff", "origin/main...", "--unified=0"],
        capture_output=True,
        text=True,
        check=False
    )
    diff: List[str] = result.stdout.splitlines()

    changed_lines: List[Tuple[str, str]] = []
    current_file: Optional[str] = None

    for line in diff:
        if line.startswith("+++ b/") and line.endswith(".py"):
            current_file = line[6:]
        elif line.startswith("@@") and current_file:
            continue
        elif line.startswith("+") and not line.startswith("+++") and current_file:
            changed_lines.append((current_file, line[1:].strip()))
    return changed_lines

def compute_annotation_ratio(changed_lines: List[Tuple[str, str]]) -> Tuple[int, int]:
    """
    Calculates the number of annotated lines and total changed lines (non-comment) in the PR diff.
    """
    total: int = 0
    annotated: int = 0

    for _, line in changed_lines:
        if line and not line.startswith("#"):
            total += 1
            if TYPE_HINT_REGEX.search(line):
                annotated += 1

    return annotated, total

def main() -> None:
    changed_lines: List[Tuple[str, str]] = get_changed_lines()
    annotated, total = compute_annotation_ratio(changed_lines)

    threshold: float = 0.3  # e.g., 30%
    print(f"Type-annotated lines: {annotated} / {total}")

    if total == 0 or annotated / total >= threshold:
        print("✅ Type annotation threshold met.")
        sys.exit(0)
    else:
        print("❌ Type annotation threshold not met.")
        sys.exit(1)

if __name__ == "__main__":
    main()
