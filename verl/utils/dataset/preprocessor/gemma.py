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
import base64
import copy
from PIL import Image
import requests
from io import BytesIO

from .base_processor import BasicPreprocessor
from .registry import PREPROCESSOR_REGISTER

__all__ = ["Gemma3Preprocessor"]

@PREPROCESSOR_REGISTER.register()
class Gemma3Preprocessor(BasicPreprocessor):
    def __init__(self, processor, image_key="image", video_key="video"):
        super().__init__(processor, image_key, video_key)
    
    def process_image(self, image, **kwargs):
        if isinstance(image, Image.Image):
            image_obj = image
        elif image.startswith("http://") or image.startswith("https://"):
            # fix memory leak issue while using BytesIO
            with requests.get(image, stream=True) as response:
                response.raise_for_status()
                with BytesIO(response.content) as bio:
                    image_obj = copy.deepcopy(Image.open(bio))
        elif image.startswith("file://"):
            image_obj = Image.open(image[7:])
        elif image.startswith("data:image"):
            if "base64," in image:
                _, base64_data = image.split("base64,", 1)
                data = base64.b64decode(base64_data)
                # fix memory leak issue while using BytesIO
                with BytesIO(data) as bio:
                    image_obj = copy.deepcopy(Image.open(bio))
        else:
            image_obj = Image.open(image)
        return image_obj.convert("RGB")
        
    def process_video(self, video, **kwargs):
        raise ValueError("Gemma3 dose not support the video")
    
    def process_audio(self, audio, **kwargs):
        raise ValueError("Gemma3 dose not support the audio")
    
    def __call__(self, messages, row_dict):
        raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        multi_modal_data = {}

        images = None
        if self.image_key in row_dict:
            images = [self.process_image(image) for image in row_dict.pop(self.image_key)]
            multi_modal_data["image"] = images
        model_inputs = self.processor(text=[raw_prompt], images=images, return_tensors="pt")
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")
        if 'token_type_ids' in model_inputs:
            model_inputs.pop("token_type_ids")
        row_dict["multi_modal_data"] = multi_modal_data
        row_dict["multi_modal_inputs"] = dict(model_inputs)
        return row_dict, model_inputs, input_ids, attention_mask, raw_prompt