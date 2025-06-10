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

# Detect inline type hints
TYPE_HINT_REGEX = re.compile(r"(->|:\s*[\w\[\]., ]+)")

def get_changed_python_lines() -> List[str]:
    base_result = subprocess.run(
        ["git", "merge-base", "HEAD", "origin/main"],
        capture_output=True,
        text=True,
        check=True
    )
    base_commit = base_result.stdout.strip()

    diff_result = subprocess.run(
        ["git", "diff", "--unified=0", base_commit, "HEAD"],
        capture_output=True,
        text=True,
        check=True
    )

    diff_lines = diff_result.stdout.splitlines()
    added_lines: List[str] = []
    for line in diff_lines:
        if line.startswith("+++ b/") and line.endswith(".py"):
            continue
        elif line.startswith("@@"):
            continue
        elif line.startswith("+") and not line.startswith("+++"):
            added_lines.append(line[1:].rstrip())

    return added_lines

def is_logical_line_start(line: str) -> bool:
    """Is this the start of a logical block (assignment, def, class)?"""
    line = line.strip()
    return (
        line.startswith("def ")
        or line.startswith("class ")
        or ("=" in line and not line.startswith(("=", ")", "]", "}")))
    )

def compute_annotation_ratio(added_lines: List[str]) -> Tuple[int, int]:
    relevant = 0
    annotated = 0
    tracking_multiline = False
    open_parens = 0

    for line in added_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if not tracking_multiline:
            if is_logical_line_start(line):
                relevant += 1
                if TYPE_HINT_REGEX.search(line):
                    annotated += 1
                else:
                    print(f"Missing annotation: {line}")
                # Start tracking if line ends with open structure
                open_parens = (
                    line.count("(") + line.count("[") + line.count("{")
                    - line.count(")") - line.count("]") - line.count("}")
                )
                tracking_multiline = open_parens > 0
        else:
            open_parens += (
                line.count("(") + line.count("[") + line.count("{")
                - line.count(")") - line.count("]") - line.count("}")
            )
            tracking_multiline = open_parens > 0

    return annotated, relevant

def main() -> None:
    try:
        added_lines = get_changed_python_lines()
    except subprocess.CalledProcessError:
        print("❌ Cannot compute diff with origin/main. Make sure CI uses `fetch-depth: 0`.")
        sys.exit(1)

    annotated, total = compute_annotation_ratio(added_lines)

    threshold = 0.5
    print(f"🔍 Relevant lines: {total}, Annotated: {annotated}")

    if total == 0:
        print("ℹ️ No type-relevant lines changed.")
        sys.exit(0)

    ratio = annotated / total
    if ratio >= threshold:
        print("✅ Type annotation threshold met.")
        sys.exit(0)
    else:
        print(f"❌ Threshold not met. Required: {threshold:.0%}, Found: {ratio:.0%}")
        sys.exit(1)

if __name__ == "__main__":
    main()
