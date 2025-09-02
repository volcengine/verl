# Copyright 2025 Individual Contributor: TomQunChaoA
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

import io
import random

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

from verl.models.transformers.qwen2_5_omni import forward_with_torch_backend, forward_with_triton_backend

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


def rand_text(max_len=32):
    vocab = "abcdefghijklmnopqrstuvwxyz 0123456789"
    return "".join(random.choices(vocab, k=random.randint(5, max_len)))


def rand_image(h=224, w=224):
    arr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    return img


def rand_audio(duration=1.0, sr=16000):
    samples = int(duration * sr)
    wave = np.random.randn(samples).astype(np.float32)
    return wave


def rand_video_bytes(duration=1.0, fps=8, h=224, w=224):
    frames = int(duration * fps)
    vid = (np.random.randint(0, 256, (frames, h, w, 3)) / 255).astype(np.float32)
    buf = io.BytesIO()
    imageio.mimsave(buf, vid, fps=fps, codec="libx264", format="mp4")
    return buf.getvalue()


def build_random_mm_inputs(
    processor: Qwen2_5OmniProcessor,
    *,
    use_text=True,
    use_image=True,
    use_audio=True,
    use_video=True,
    batch_size=1,
):
    conversations = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group,"
                    " capable of perceiving auditory and visual inputs, "
                    "as well as generating text and speech.",
                }
            ],
        }
    ]
    images = []
    audios = []
    videos = []
    if use_text:
        conversations.append({"role": "user", "content": [{"type": "text", "text": rand_text()}]})
    if use_image:
        img_content = rand_image()
        images.append(img_content)
        conversations.append({"role": "user", "content": [{"type": "image", "image": None}]})
    if use_audio:
        audio_content = rand_audio()
        audios.append(audio_content)
        conversations.append({"role": "user", "content": [{"type": "audio", "audio": None}]})
    if use_video:
        vid_bytes = rand_video_bytes()
        videos.append(io.BytesIO(vid_bytes))
        conversations.append({"role": "user", "content": [{"type": "video", "image": None}]})

    text = processor.apply_chat_template(conversations)

    inputs = processor(
        text=text,
        images=images if images else None,
        audio=audios if audios else None,
        videos=videos if videos else None,
        return_tensors="pt",
        padding=True,
    )
    return inputs


def compute_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)

    labels_expanded = labels.unsqueeze(-1)  # [x, 165, 1]

    gathered_logprobs = log_probs.gather(dim=-1, index=labels_expanded)  # [x, 165, 1]

    logprobs = gathered_logprobs.squeeze(-1)  # [x, 165]

    return logprobs


class TestQwen2_5_OmniPatch:
    def __init__(self):
        model_path = "Qwen/Qwen2.5-Omni-3B"
        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(model_path).cuda()
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.inputs = (
            build_random_mm_inputs(self.processor, use_text=True, use_image=True, use_audio=True, use_video=False)
            .to(self.model.device)
            .to(self.model.dtype)
        )

    def test_fused_backend(self, fused_backend):
        logprob_fused_backend = fused_backend(self.model, **self.inputs, return_dict=True).log_probs
        logits_origin_backend = self.model(**self.inputs, return_dict=True).logits
        input_ids = self.inputs["input_ids"]
        labels = torch.roll(input_ids, shifts=-1, dims=-1)
        logprob_origin_backend = compute_logprobs(logits_origin_backend, labels)
        assert torch.allclose(logprob_fused_backend, logprob_origin_backend, atol=1e-2, rtol=3e-5)

    def test_torch_backend(self):
        self.test_fused_backend(forward_with_torch_backend)

    def test_triton_backend(self):
        self.test_fused_backend(forward_with_triton_backend)
