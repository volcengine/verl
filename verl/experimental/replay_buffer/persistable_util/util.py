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


def delete_files(*file_names):
    for file_name in file_names:
        try:
            os.remove(file_name)
        except OSError:
            pass


def to_bytes(value):
    """Convert an integer or string to bytes."""
    assert isinstance(value, int) or isinstance(value, str), "replay buffer key must be an int or a string."

    if isinstance(value, int):
        byte_length = (value.bit_length() + 7) // 8 or 1  # Calculate byte length
        return value.to_bytes(byte_length, byteorder="big", signed=True)
    elif isinstance(value, str):
        return value.encode("utf-8")

    return None
