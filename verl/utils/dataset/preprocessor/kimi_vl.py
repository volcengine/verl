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
from qwen_vl_utils import fetch_video

from .base_processor import BasicPreprocessor
from .registry import PREPROCESSOR_REGISTER

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
class KimiVLPreprocessor(BasicPreprocessor):
    def __init__(self, processor, image_key="image", video_key="video", audio_key="audio"):
        super().__init__(processor, image_key, video_key, audio_key)
    
    def process_audio(self, audio, **kwargs):
        raise ValueError("KimiVL dose not support audio")
    
    def process_video(self, video, **kwargs):
        """
            Here We adopt to use the QwenVL series preprocess here
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