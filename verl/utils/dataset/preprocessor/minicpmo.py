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
import torch
import base64
import copy
from PIL import Image
import requests
from io import BytesIO
from qwen_vl_utils import fetch_video

from .base_processor import BasicPreprocessor
from .registry import PREPROCESSOR_REGISTER


def process_multi_modal_inputs_for_minicpmo(input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs):
    # https://github.com/volcengine/verl/blob/1bdf4d2bc718540238864ee02df9a1c6614860bf/verl/utils/dataset/vision_utils.py#L95C1-L117C40
    # Adjust image bounds based on left padding and cumulative sequence lengths
    # This is necessary for MiniCPM-o's vision-language alignment
    left_padding_length = torch.argmax(attention_mask, dim=1)
    image_bounds = []
    for i in range(len(multi_modal_inputs["image_bound"])):
        image_bound = (
            multi_modal_inputs["image_bound"][i].to(left_padding_length.device) - left_padding_length[i] + cu_seqlens[i]
        )
        image_bounds.append(image_bound)

    # Flatten pixel values list for MiniCPM-o processing
    pixel_values = []
    for i in range(len(multi_modal_inputs["pixel_values"])):
        pixel_values.extend([p for p in multi_modal_inputs["pixel_values"][i]])

    multi_modal_inputs["pixel_values"] = [pixel_values]
    multi_modal_inputs["image_bound"] = [torch.vstack(image_bounds)]
    multi_modal_inputs["tgt_sizes"] = [torch.vstack(multi_modal_inputs["tgt_sizes"])]
    multi_modal_inputs["input_ids"] = input_ids
    multi_modal_inputs["attention_mask"] = attention_mask
    multi_modal_inputs["position_ids"] = position_ids
    return {"data": multi_modal_inputs}

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

PREPROCESSOR_REGISTER.register()
class MiniCPMOPreProcessor(BasicPreprocessor):
    def __init__(self, processor, image_key="image", video_key="video", audio_key="audio"):
        super().__init__(processor, image_key, video_key, audio_key)

    def process_audio(self, audio, **kwargs):
        raise ValueError("We do not support the audio input for MiniCPM-O now")
    
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
        video_total_pixels = kwargs.get("video_total_pixels", self.video_total_pixels)
        video_min_pixels = kwargs.get("video_min_pixels", self.video_min_pixels)
        video["total_pixels"] = video_total_pixels
        video["min_pixels"] = video_min_pixels
        return_video_sample_fps = kwargs.get("return_video_sample_fps", False)
        image_factor = kwargs.get("image_factor", self.factor)
        return fetch_video(video, image_factor=image_factor, return_video_sample_fps=return_video_sample_fps)