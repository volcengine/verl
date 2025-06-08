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

import os
import re
import sys

# Get PR title from environment
pr_title = os.environ.get("PR_TITLE", "").strip()

# Define rules
allowed_modules = ['fsdp', 'megatron', 'sglang', 'vllm', 'rollout', 'trainer']
allowed_modules += ['tests', 'training_utils', 'recipe', 'hardware', 'deployment']
allowed_modules += ['ray', 'worker', 'single_controller', 'misc']
allowed_modules += ['perf', 'model', 'algo', 'env', 'tool', 'ckpt']
allowed_types = ['feat', 'fix', 'doc', 'refactor', 'chore']

# Build dynamic regex pattern
types_pattern = '|'.join(re.escape(t) for t in allowed_types)
pattern = re.compile(rf'^\[([a-z]+)\]\s+({types_pattern}):\s+.+$', re.IGNORECASE)
match = pattern.match(pr_title)

if not match:
    print(f"❌ Invalid PR title: '{pr_title}'")
    print("Expected format: [module] type: description")
    print(f"Allowed modules: {', '.join(allowed_modules)}")
    print(f"Allowed types: {', '.join(allowed_types)}")
    sys.exit(1)

module, change_type = match.group(1).lower(), match.group(2).lower()

if module not in allowed_modules:
    print(f"❌ Invalid module: '{module}'")
    print(f"Must be one of: {', '.join(allowed_modules)}")
    sys.exit(1)

print(f"✅ PR title is valid: {pr_title}")
