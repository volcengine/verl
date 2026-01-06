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

from openai.types.chat.chat_completion import ChatCompletion, ChatCompletionMessage, Choice


class InternalChatCmplMessage(ChatCompletionMessage):
    """Extend ChatCompletionMessage to support reasoning content."""

    reasoning_content: str | None = None


class InternalChoice(Choice):
    message: InternalChatCmplMessage


class InternalChatCompletion(ChatCompletion):
    choices: list[InternalChoice]
