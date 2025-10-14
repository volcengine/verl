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

from io import BytesIO
from typing import Optional

import torch
from PIL import Image
from qwen_vl_utils import fetch_image, fetch_video

from verl.utils.model import compute_position_id_with_mask


def process_image(image: dict | Image.Image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if "bytes" in image:
        assert "image" not in image, "Cannot have both `bytes` and `image`"
        image["image"] = Image.open(BytesIO(image["bytes"]))

    return fetch_image(image)


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


def process_video(
    video: dict,
    nframes: Optional[int] = None,
    fps: Optional[float] = None,
    fps_min_frames: Optional[int] = None,
    fps_max_frames: Optional[int] = None,
) -> torch.Tensor:
    """Converts a video dict into a [n_frames, 3, H, W] tensor

    Add video sample FPS in a future MR
    """

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


def process_multi_modal_inputs_for_minicpmo(input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs):
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


def compute_multimodal_position_ids(
    processor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    image_grid_thw: Optional[torch.Tensor] = None,
    video_grid_thw: Optional[torch.Tensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute position_ids for multimodal models. Falls back to standard cumulative
    position ids when no multimodal processor is provided.
    """
    if processor is None or not hasattr(processor, "image_processor"):
        return compute_position_id_with_mask(attention_mask)

    # Normalize tensor shapes to 1-D [seq_len]
    if input_ids.dim() > 1:
        assert input_ids.size(0) == 1, "Expect batch dimension of size 1 for input_ids"
        input_ids_1d = input_ids[0]
    else:
        input_ids_1d = input_ids

    if attention_mask.dim() > 1:
        assert attention_mask.size(0) == 1, "Expect batch dimension of size 1 for attention_mask"
        attention_mask_1d = attention_mask[0]
    else:
        attention_mask_1d = attention_mask

    attention_mask_1d = attention_mask_1d.to(device=input_ids_1d.device, dtype=torch.long)
    processor_name = processor.image_processor.__class__.__name__

    def _build_text_position_ids(valid_mask: torch.Tensor) -> torch.Tensor:
        text_pos = torch.zeros(
            (1, attention_mask_1d.numel()),
            dtype=torch.long,
            device=input_ids_1d.device,
        )
        valid_count = int(valid_mask.sum().item())
        if valid_count > 0:
            text_pos[0, valid_mask] = torch.arange(valid_count, dtype=torch.long, device=input_ids_1d.device)
        return text_pos

    if "Qwen2VLImageProcessor" in processor_name:
        if image_grid_thw is None and video_grid_thw is None:
            return compute_position_id_with_mask(attention_mask_1d)

        if "Qwen3VLProcessor" in processor.__class__.__name__:
            from verl.models.transformers.qwen3_vl import get_rope_index
        else:
            from verl.models.transformers.qwen2_vl import get_rope_index

        vision_position_ids = get_rope_index(
            processor,
            input_ids=input_ids_1d,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            attention_mask=attention_mask_1d,
        )
        valid_mask = attention_mask_1d.to(dtype=torch.bool)
        text_position_ids = _build_text_position_ids(valid_mask)
        return torch.cat((text_position_ids, vision_position_ids.to(input_ids_1d.device)), dim=0)

    if "Glm4vImageProcessor" in processor_name:
        if image_grid_thw is None and video_grid_thw is None:
            return compute_position_id_with_mask(attention_mask_1d)

        from verl.models.transformers.glm4v import get_rope_index

        vision_position_ids = get_rope_index(
            processor,
            input_ids=input_ids_1d,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask_1d,
        )
        valid_mask = attention_mask_1d.to(dtype=torch.bool)
        text_position_ids = _build_text_position_ids(valid_mask)
        return torch.cat((text_position_ids, vision_position_ids.to(input_ids_1d.device)), dim=0)

    return compute_position_id_with_mask(attention_mask_1d)
