# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
from argparse import ArgumentParser
from pathlib import Path

license_head_bytedance = "Copyright 2024 Bytedance Ltd. and/or its affiliates"
license_head_bytedance_25 = "Copyright 2025 Bytedance Ltd. and/or its affiliates"
# Add custom license headers below
license_head_prime = "Copyright 2024 PRIME team and/or its affiliates"
license_head_individual = "Copyright 2025 Individual Contributor:"
license_headers = [license_head_bytedance, license_head_bytedance_25, license_head_prime, license_head_individual]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--directory", "-d", required=True, type=str)
    args = parser.parse_args()
    directory_in_str = args.directory

    pathlist = Path(directory_in_str).glob("**/*.py")
    for path in pathlist:
        # because path is object not string
        path_in_str = str(path.absolute())
        print(path_in_str)
        with open(path_in_str, encoding="utf-8") as f:
            file_content = f.read()

            has_license = False
            for lh in license_headers:
                if lh in file_content:
                    has_license = True
                    break
            assert has_license, f"file {path_in_str} does not contain license"
