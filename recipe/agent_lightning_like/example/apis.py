# Copyright 2025 Individual Contributor: linxxx3 (linxxx3@gmail.com)
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

from dataclasses import dataclass

from pydantic import BaseModel


@dataclass
class UserContext:
    """extra info for gsm8k tool"""

    trace_id: str
    max_turns: int = 10
    ground_truth: str | None = None


class ChatRequest(BaseModel):
    context: UserContext
    system_prompt: str | None = None
    prompt: str
    max_tokens: int = 1024
    temperature: float = 0.6
    top_p: float = 0.8
    extra_headers: dict = {}
    extra_body: dict = {}


class ChatResponse(BaseModel):
    context: UserContext
    response: str
    error: str | None = None
