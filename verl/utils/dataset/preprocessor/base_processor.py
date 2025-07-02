# Copyright 2024 Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
The basic preprocessor used for the multi-modal models.
"""

class BasicPreprocessor:
    def __init__(self, processor, image_key="image", video_key="video", audio_key="audio"):
        self.processor = processor
        self.image_key = image_key
        self.video_key = video_key
        self.audio_key = audio_key
    
    def process_image(self, image, **kwargs):
        raise NotImplementedError("The process_image method must be implemented")
    
    def process_video(self, video, **kwargs):
        raise NotImplementedError("The process_video method must be implemented")
    
    def process_audio(self, audio, **kwargs):
        raise NotImplementedError("The process_video method must be implemented")
    
    def __call__(self, messages, row_dict):
        raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        multi_modal_data = {}

        images = None
        if self.image_key in row_dict:
            images = [self.process_image(image) for image in row_dict.pop(self.image_key)]
            multi_modal_data["image"] = images

        videos = None
        if self.video_key in row_dict:
            videos = [self.process_video(video) for video in row_dict.pop(self.video_key)]
            multi_modal_data["video"] = [video.numpy() for video in videos]
        model_inputs = self.processor(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

        if "second_per_grid_ts" in model_inputs:
            model_inputs.pop("second_per_grid_ts")
        # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
        row_dict["multi_modal_data"] = multi_modal_data
        row_dict["multi_modal_inputs"] = dict(model_inputs)

        # second_per_grid_ts isn't used for training, just for mrope
        row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)
        return row_dict, model_inputs, input_ids, attention_mask, raw_prompt