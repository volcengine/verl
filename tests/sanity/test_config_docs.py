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

import re
from pathlib import Path

def validate_yaml_format(yaml_lines):
    errors = []
    i = 0
    while i < len(yaml_lines):
        line = yaml_lines[i]

        # Skip empty lines
        if line.strip() == "":
            i += 1
            continue

        # Skip top-level comments (not field doc comments)
        if line.strip().startswith("#"):
            i += 1
            continue

        # Match key lines (e.g. "tokenizer: null" or "use_shm: False")
        key_match = re.match(r'^(\s*)([a-zA-Z0-9_]+):', line)
        if key_match:
            indent = key_match.group(1)

            # Look at the line above
            if i == 0 or not yaml_lines[i - 1].strip().startswith("#"):
                errors.append(f"Missing comment above line {i+1}: {line.strip()}")

            # Look at the line after the field (should be blank unless it's a nested object)
            if i + 1 < len(yaml_lines):
                next_line = yaml_lines[i + 1]
                if next_line.strip() and not next_line.strip().startswith("#"):
                    if not next_line.startswith(indent + "  "):  # nested field is okay
                        errors.append(f"Missing blank line after line {i+1}: {line.strip()}")

            # Check for inline comments
            if "#" in line and not line.strip().startswith("#"):
                errors.append(f"Inline comment found on line {i+1}: {line.strip()}")

        i += 1

    return errors


def test_trainer_config_doc():
    yaml_path = Path("verl/trainer/config/ppo_trainer.yaml")  # path to your YAML file
    with open(yaml_path, "r") as f:
        lines = f.readlines()

    validation_errors = validate_yaml_format(lines)
    if validation_errors:
        print("YAML documentation format check failed:")
        print("Please read the top block of `verl/trainer/config/ppo_trainer.yaml` to see format rules:\n")
        for err in validation_errors:
            print(" -", err)
    else:
        print("YAML format check passed ✅")
