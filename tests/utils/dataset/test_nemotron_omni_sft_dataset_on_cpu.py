# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
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
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from tensordict import TensorDict

from verl.utils.dataset.dataset_utils import DatasetPadMode, SFTTensorCollator
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset, _pack_nemotron_omni_images
from verl.utils.model import extract_multi_modal_inputs


class NemotronH_Omni_Reasoning_V3Processor:
    """Model the public processor's ragged image output without downloading model files."""

    image_processor = SimpleNamespace(patch_size=2)

    def apply_chat_template(self, messages, *, return_tensors=None, **kwargs):
        assert return_tensors is None, "Ragged images cannot be converted by the HF BatchFeature."
        ids, images = [], []
        for message in messages:
            ids.append(2 if message["role"] == "assistant" else 1)
            for part in message["content"]:
                if part["type"] == "image":
                    image = torch.from_numpy(np.array(part["image"])).permute(2, 0, 1).float()
                    images.append(image)
                    ids.extend([42] * (image.shape[1] * image.shape[2] // 16))
                else:
                    ids.append(50)
            ids.append(3)
        result = {"input_ids": [ids], "attention_mask": [[1] * len(ids)]}
        if images:
            result.update(
                pixel_values=torch.stack(images) if all(x.shape == images[0].shape for x in images) else images,
                imgs_sizes=[list(x.shape[1:]) for x in images],
                num_patches=[1] * len(images),
                num_tokens=[x.shape[1] * x.shape[2] // 16 for x in images],
            )
        return result


def _image_bytes(width, height, color):
    data = BytesIO()
    Image.new("RGB", (width, height), color=color).save(data, format="PNG")
    return {"bytes": data.getvalue()}


@pytest.fixture
def make_dataset(monkeypatch):
    def make(rows):
        def read_files(dataset):
            dataset.dataframe = pd.DataFrame(rows)
            dataset.messages = dataset.dataframe["messages"].tolist()
            dataset.tools = dataset.enable_thinking = None
            dataset.system_prompt, dataset.generation_prompt = [], [2]

        monkeypatch.setattr(MultiTurnSFTDataset, "_download", lambda self: None)
        monkeypatch.setattr(MultiTurnSFTDataset, "_read_files_and_process", read_files)
        tokenizer = SimpleNamespace(
            pad_token_id=0,
            apply_chat_template=lambda *args, **kwargs: "",
            decode=lambda *args, **kwargs: "",
        )
        return MultiTurnSFTDataset(
            [],
            tokenizer,
            {"max_length": 128, "pad_mode": "no_padding"},
            processor=NemotronH_Omni_Reasoning_V3Processor(),
        )

    return make


def test_nemotron_multiturn_images_and_text_batch(make_dataset):
    rows = [
        {
            "messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}],
            "images": [],
        },
        {
            "messages": [
                {"role": "user", "content": "<image><image>Compare"},
                {"role": "assistant", "content": "Red and green"},
                {"role": "user", "content": "<image>And this?"},
                {"role": "assistant", "content": "Blue"},
            ],
            "images": [_image_bytes(4, 4, "red"), _image_bytes(8, 4, "green"), _image_bytes(12, 4, "blue")],
        },
        {
            "messages": [
                {"role": "user", "content": "<image>Describe"},
                {"role": "assistant", "content": "White"},
            ],
            "images": [_image_bytes(8, 8, "white")],
        },
    ]
    dataset = make_dataset(rows)
    items = [dataset[i] for i in range(len(dataset))]
    assert items[0]["multi_modal_inputs"] == {}
    assert items[1]["multi_modal_inputs"]["pixel_values"].shape == (24, 12)
    assert items[1]["multi_modal_inputs"]["imgs_sizes"].tolist() == [[4, 4], [4, 8], [4, 12]]
    assert items[1]["position_ids"].ndim == 1
    assert torch.all(items[1]["loss_mask"][items[1]["input_ids"] == 42] == 0)
    assert items[1]["loss_mask"].sum() == 4

    batch = TensorDict(SFTTensorCollator(DatasetPadMode.NO_PADDING)(items), batch_size=len(items))
    inputs = extract_multi_modal_inputs(batch["multi_modal_inputs"])
    assert set(inputs) == {"pixel_values", "imgs_sizes", "num_frames"}
    assert inputs["pixel_values"].shape == (40, 12)
    assert inputs["imgs_sizes"].tolist() == [[4, 4], [4, 8], [4, 12], [8, 8]]
    assert inputs["num_frames"].tolist() == [1, 1, 1, 1]


@pytest.mark.parametrize("as_list", [False, True])
def test_nemotron_patch_order_matches_radio(as_list):
    images = torch.arange(2 * 3 * 4 * 6).reshape(2, 3, 4, 6)
    inputs = _pack_nemotron_omni_images(
        {"pixel_values": list(images) if as_list else images, "imgs_sizes": [[4, 6], [4, 6]]}, patch_size=2
    )
    expected = torch.stack(
        [image[:, y : y + 2, x : x + 2].flatten() for image in images for y in (0, 2) for x in (0, 2, 4)]
    )
    torch.testing.assert_close(inputs["pixel_values"], expected)


@pytest.mark.parametrize("sizes", [[[4, 4]], [[4, 4], [4, 5]]])
def test_nemotron_invalid_image_sizes(sizes):
    with pytest.raises(ValueError, match="Nemotron"):
        _pack_nemotron_omni_images({"pixel_values": torch.zeros(2, 3, 4, 4), "imgs_sizes": sizes}, patch_size=2)


@pytest.mark.parametrize("content", ["<video>Describe", "<audio>Transcribe", [{"type": "audio", "audio": "a.wav"}]])
def test_nemotron_rejects_unsupported_media(make_dataset, content):
    dataset = make_dataset([{"messages": [{"role": "user", "content": content}], "images": []}])
    with pytest.raises(NotImplementedError, match="images and text only"):
        dataset[0]


@pytest.mark.parametrize("key", ["sound_clips", "pixel_values_videos"])
def test_nemotron_rejects_unsupported_processor_outputs(key):
    with pytest.raises(NotImplementedError, match="images and text only"):
        _pack_nemotron_omni_images({key: []}, patch_size=2)
