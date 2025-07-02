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


"""
The InternVL preprocessor used for the multi-modal models.
"""
import base64
import copy
from PIL import Image
import requests
from io import BytesIO
from qwen_vl_utils import fetch_video

from .base_processor import BasicPreprocessor
from .registry import PREPROCESSOR_REGISTER

__all__ = ["InternVLPreprocessor"]

VIDEO_FORMAT_HELP = """Currently, we only support the video formats introduced in qwen2-vl.
Refer to https://github.com/QwenLM/Qwen2.5-VL?tab=readme-ov-file#using---transformers-to-chat.

eg.
{
    "type": "video",
    "video": [
        "file:///path/to/frame1.jpg",
        "file:///path/to/frame2.jpg"
    ]
}

{
    "type": "video",
    "video": "file:///path/to/video.mp4"
}
# Defaults to fps=2, min_frames=4, max_frames=768

{
    "type": "video",
    "video": "file:///path/to/video.mp4",
    "fps": 2,
    "min_frames": 1,
    "max_frames": 32
}
"""

@PREPROCESSOR_REGISTER.register()
class InternVLPreprocessor(BasicPreprocessor):
    def __init__(self, processor, image_key="image", video_key="video", **kwargs):
        super().__init__(processor, image_key=image_key, video_key=video_key)

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
        """Converts a video dict into a [n_frames, 3, H, W] tensor

        Add video sample FPS in a future MR
        """
        nframes = kwargs.get("nframes", None)
        fps = kwargs.get("fps", None)
        fps_min_frames = kwargs.get("fps_min_frames", None),
        fps_max_frames = kwargs.get("fps_max_frames", None),
        if not isinstance(video, dict) or "video" not in video:
            raise NotImplementedError(VIDEO_FORMAT_HELP)
        assert nframes is None or fps is None, "Can't use both `nframes` or `fps`"

        # Shallow copy... since we might want to add some keys
        video = dict(video)

        contains_sampling_rules = "nframes" in video or "fps" in video
        if not contains_sampling_rules:
            if nframes is not None:
                video["nframes"] = nframes
            elif fps is not None:
                video["fps"] = fps
                if fps_min_frames is not None:
                    video["min_frames"] = fps_min_frames
                if fps_max_frames is not None:
                    video["max_frames"] = fps_max_frames
        return fetch_video(video)
    
    def process_audio(self, audio, **kwargs):
        raise ValueError("InternVL dose not support audio")

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
        raw_prompt_convert = raw_prompt
        if "<image>" in raw_prompt_convert:
            #In older version the fake_image_token will be used
            raw_prompt_convert=raw_prompt_convert.replace("<image>", "<IMG_CONTEXT>")
        model_inputs = self.processor(text=[raw_prompt_convert], images=images, videos=videos, return_tensors="pt")
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