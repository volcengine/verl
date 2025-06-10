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

# Detect inline type annotations (func args, return types, var annotations)
TYPE_HINT_REGEX = re.compile(r"(->|:\s*[\w\[\]., ]+)")

def get_changed_python_lines() -> List[Tuple[str, str]]:
    base_result = subprocess.run(
        ["git", "merge-base", "HEAD", "origin/main"],
        capture_output=True,
        text=True,
        check=True,
    )
    base_commit = base_result.stdout.strip()

    diff_result = subprocess.run(
        ["git", "diff", "--unified=0", base_commit, "HEAD"],
        capture_output=True,
        text=True,
        check=True,
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
            changed_lines.append((current_file, line[1:].rstrip()))

    return changed_lines

def is_type_check_relevant(line: str) -> bool:
    """
    Only count lines that:
    - start a function or class definition
    - or assign a new variable
    - and are the start of the construct (not continuations)
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return False

    # Start of function or class
    if stripped.startswith("def ") or stripped.startswith("class "):
        return True

    # New variable assignment (top-level, not continuation)
    if "=" in line and not line.lstrip().startswith(("=", ")", "]", "}")):
        # Ignore lines that are purely part of argument passing
        if stripped.endswith(("(", "[", "{")):
            return True
        if not line.startswith(" "):  # avoid indented continuation lines
            return True
        if line.count("=") == 1 and not any(c in line for c in "([{") and line.index("=") < 30:
            return True

    return False

def has_type_annotation(line: str) -> bool:
    """True if line has any type hint."""
    return bool(TYPE_HINT_REGEX.search(line))

def compute_annotation_ratio(changed_lines: List[Tuple[str, str]]) -> Tuple[int, int]:
    total = 0
    annotated = 0
    for _, line in changed_lines:
        if is_type_check_relevant(line):
            total += 1
            if has_type_annotation(line):
                annotated += 1
            else:
                print(f"Missing annotation: {line}")
    return annotated, total

def main() -> None:
    try:
        changed_lines = get_changed_python_lines()
    except subprocess.CalledProcessError:
        print("❌ Cannot compute diff with origin/main. Ensure CI sets `fetch-depth: 0`.")
        sys.exit(1)

    annotated, total = compute_annotation_ratio(changed_lines)

    threshold = 0.5
    print(f"🔍 Relevant lines: {total}, Annotated: {annotated}", flush=True)

    if total == 0:
        print("ℹ️ No relevant lines to check.")
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
