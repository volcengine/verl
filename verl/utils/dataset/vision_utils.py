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


def aggregate_multi_modal_inputs_for_vlm(mmi_list):
    """
    Aggregate list[dict] of multi_modal_inputs across samples by union of keys.
    """
    all_keys = set().union(*(set(d.keys()) for d in mmi_list))
    out = {}
    visual_keys = {"pixel_values", "pixel_values_videos"}
    grid_keys = {"image_grid_thw", "video_grid_thw", "second_per_grid_ts"}

    for k in all_keys:
        vals = []
        for d in mmi_list:
            if k in d and d[k] is not None:
                v = d[k]
                if hasattr(v, "ndim"):
                    v = v.squeeze(0) if v.ndim == 3 else v
                vals.append(v)

        if k in visual_keys:
            if len(vals) > 0:
                out[k] = torch.cat(vals, dim=0)
        elif k in grid_keys:
            if len(vals) > 0:
                out[k] = torch.cat(vals, dim=0) if isinstance(vals[0], torch.Tensor) else vals
        else:
            if len(vals) > 0:
                out[k] = torch.cat(vals, dim=0) if isinstance(vals[0], torch.Tensor) else vals

    return out


def align_position_ids_for_rmpad(position_ids: torch.Tensor, indices: torch.Tensor):
    """
    Align 2D/3D position_ids to rmpad layout using 'indices'.
    """
    if position_ids.dim() == 3:
        from flash_attn.bert_padding import index_first_axis, rearrange
        position_ids_rmpad = (
            index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
            .transpose(0, 1)
            .unsqueeze(1)
        )
        return position_ids_rmpad
    else:
        from flash_attn.bert_padding import index_first_axis, rearrange
        return index_first_axis(
            rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
        ).transpose(0, 1)


def handle_glm4v_position_ids(model_type_lower: str, use_remove_padding: bool, position_ids_arg):
    """
    GLM4V specific handling: set position_ids to None when use_remove_padding is True.
    """
    is_glm4v = "glm4v" in (model_type_lower or "")
    if is_glm4v and use_remove_padding and position_ids_arg is not None and position_ids_arg.dim() == 3:
        return None
    return position_ids_arg
