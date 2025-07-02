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

from transformers import AutoProcessor
from .gemma import Gemma3Preprocessor
from .internvl import InternVLPreprocessor
from .qwen_vl import QwenVLPreProcessor
from .minicpmo import MiniCPMOPreProcessor
from .kimi_vl import KimiVLPreprocessor
from .registry import PREPROCESSOR_REGISTER
import re


def map_processor_to_preprocessor(processor:AutoProcessor):
    """
        Map the processor to the Preprocessor
        Args:
            processor(AutoProcessor): The processor.
        Return:
            class: The preprocess class
    """
    processor_name = processor.__class__.__name__
    if not processor_name.lower().endswith("processor"):
        raise ValueError(f"Source object '{processor_name}' is not a 'Processor'.")
    if re.match("Qwen2.*?VLProcessor", processor_name):
        print("QwenVL2 Series will use the QwenVLPreprocessor")
        dest_name = "QwenVLPreprocessor".lower()
    else:
        dest_name = processor_name.lower().replace("processor", "preprocessor")
    
    dest_class = PREPROCESSOR_REGISTER.get(dest_name)
    return dest_class