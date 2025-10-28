# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
In single GPU rollout, the sequences are generated directly by sampling from the model.
The output will contain
1. output_ids
2. attention_masks (left padding)
3. eos_masks
4. log_probs
"""

import json
import os
from io import BytesIO

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from tensordict import TensorDict
from torch.nn.utils.rnn import pad_sequence

from verl import DataProto
from verl.models.openvla_oft.modeling_prismatic import OpenVLAForActionPrediction
from verl.models.openvla_oft.processing_prismatic import PrismaticProcessor
from verl.workers.rollout.base import BaseRollout


def center_crop_image(image: Image.Image) -> Image.Image:
    crop_scale = 0.9
    orig_w, orig_h = image.size
    image_tensor = F.to_tensor(image)
    crop_h = int(orig_h * crop_scale)
    crop_w = int(orig_w * crop_scale)
    image_tensor = F.center_crop(image_tensor, (crop_h, crop_w))
    image_tensor = F.resize(image_tensor, (orig_h, orig_w))
    final_image = F.to_pil_image(image_tensor)

    final_image = final_image.convert("RGB")
    return final_image


def resize_image(img, resize_size):
    assert isinstance(resize_size, tuple), "resize_size must be a tuple"
    assert isinstance(img, np.ndarray), "img must be a numpy array"

    # Convert numpy array to PIL Image
    pil_img = Image.fromarray(img)

    # Encode as JPEG, as done in RLDS dataset builder
    buffer = BytesIO()
    pil_img.save(buffer, format="JPEG")
    buffer.seek(0)

    # Immediately decode back
    img = Image.open(buffer)

    img = img.resize(resize_size, Image.Resampling.LANCZOS)
    img = np.array(img)
    img = np.clip(np.round(img), 0, 255).astype(np.uint8)

    return img


__all__ = ["NaiveRolloutRob"]


def pad_sequence_to_length(tensors, max_seq_len, pad_token_id, left_pad=False):
    """
    pad a 2D tensors (e.g. responses, logprobs) in the last dim to max_seq_length.
    input shape: [bs, seq_length]
    output shape: [bs, max_seq_length]
    (0, max_seq_len - tensors.shape[-1]) means right pad to max_seq_length and no left pad
    """
    if tensors.shape[-1] >= max_seq_len:
        return tensors
    pad_tuple = (max_seq_len - tensors.shape[-1], 0) if left_pad else (0, max_seq_len - tensors.shape[-1])
    return torch.nn.functional.pad(tensors, pad_tuple, "constant", pad_token_id)


def process_input(task_descriptions, images_and_states, processor):
    batchdata = {"input_ids": [], "attention_mask": [], "pixel_values": []}

    for i in range(len(task_descriptions)):
        task_description = task_descriptions[i]
        image = resize_image(images_and_states["full_image"][i].cpu().numpy(), (224, 224))
        image = Image.fromarray(image).convert("RGB")
        image = center_crop_image(image)
        prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"
        batch_feature = processor(prompt, image)

        # if "wrist_image" in venv_obs.keys():
        #     wrist_image = Image.fromarray(input["wrist_image"]).convert("RGB")
        #     if config["center_crop"]:
        #         wrist_image = center_crop_image(wrist_image)
        #     wrist_batch_feature = processor(prompt, wrist_image)
        #     primary_pixel_values = batch_feature["pixel_values"]
        #     batch_feature["pixel_values"] = (torch.cat([primary_pixel_values] +
        #     [wrist_batch_feature["pixel_values"]], dim=1))

        input_ids = batch_feature["input_ids"]
        attention_mask = batch_feature["attention_mask"]
        pixel_values = batch_feature["pixel_values"]

        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
            )
            attention_mask = torch.cat(
                (attention_mask, torch.unsqueeze(torch.Tensor([True]).bool(), dim=0).to(attention_mask.device)), dim=1
            )

        batchdata["input_ids"].append(input_ids)
        batchdata["attention_mask"].append(attention_mask)
        batchdata["pixel_values"].append(pixel_values)

    device = torch.device("cuda")

    batchdata["input_ids"] = [x.transpose(0, 1) for x in batchdata["input_ids"]]
    batchdata["attention_mask"] = [x.transpose(0, 1) for x in batchdata["attention_mask"]]
    batchdata["input_ids"] = (
        pad_sequence(batchdata["input_ids"], batch_first=True, padding_value=processor.tokenizer.pad_token_id)
        .squeeze(-1)
        .to(device)
    )
    batchdata["attention_mask"] = (
        pad_sequence(batchdata["attention_mask"], batch_first=True, padding_value=0).squeeze(-1).to(device)
    )

    padding_mask = batchdata["input_ids"].ne(processor.tokenizer.pad_token_id)
    assert torch.all(padding_mask == batchdata["attention_mask"].ne(0))
    padding_mask = ~padding_mask
    padding_mask = padding_mask.int()
    sorted_indices = torch.argsort(padding_mask, dim=1, descending=True, stable=True)
    batchdata["input_ids"] = torch.gather(batchdata["input_ids"], 1, sorted_indices)
    batchdata["attention_mask"] = torch.gather(batchdata["attention_mask"], 1, sorted_indices)

    batchdata["pixel_values"] = torch.cat(batchdata["pixel_values"], dim=0).to(device)
    assert torch.all(batchdata["attention_mask"].ne(0) == batchdata["input_ids"].ne(processor.tokenizer.pad_token_id))

    return batchdata


class NaiveRolloutRob(BaseRollout):
    def __init__(
        self,
        model_config: dict,
        module: torch.nn.Module = None,
    ):
        self.model_config = model_config
        # self.model_config.update({
        #     "center_crop": True,
        #     "num_steps_wait": 10
        # })
        # actor_model_config = OpenVLAConfig.from_pretrained(local_path, trust_remote_code=True)
        # print("model_config local_path", model_config["local_path"])
        if module is not None:
            self.module = module
        else:
            self.module = OpenVLAForActionPrediction.from_pretrained(model_config["path"], trust_remote_code=True).to(
                torch.cuda.current_device()
            )
        self.module.vision_backbone.set_num_images_in_input(1)
        self.processor = PrismaticProcessor.from_pretrained(model_config["path"], trust_remote_code=True)
        dataset_statistics_path = os.path.join(model_config["path"], "dataset_statistics.json")
        if os.path.isfile(dataset_statistics_path):
            with open(dataset_statistics_path) as f:
                norm_stats = json.load(f)
            self.module.norm_stats = norm_stats
        self.module.eval()

    @torch.no_grad()
    def _generate_one_step(self, prompts: dict, do_sample, temperature, max_prompt_length):
        idx = prompts["input_ids"]  # (bs, prompt_length)
        attention_mask = prompts["attention_mask"]  # left-padded attention_mask
        pixel_values = prompts["pixel_values"]

        # generation_config = GenerationConfig(temperature=temperature, top_p=top_p, top_k=top_k)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            actions, response = self.module.generate_action_verl(
                input_ids=idx,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                padding_idx=self.processor.tokenizer.pad_token_id,
                do_sample=do_sample,
                unnorm_key="libero_10_no_noops",
                temperature=temperature,
            )

        assert self.processor.tokenizer.pad_token_id is not None

        assert idx.ndim == 2
        idx = pad_sequence_to_length(
            idx, max_seq_len=max_prompt_length, pad_token_id=self.processor.tokenizer.pad_token_id, left_pad=True
        )

        assert attention_mask.ndim == 2
        attention_mask = pad_sequence_to_length(
            attention_mask, max_seq_len=max_prompt_length, pad_token_id=0, left_pad=True
        )

        assert idx.device.type == "cuda"
        assert response.device.type == "cuda"
        # assert seq.device.type == 'cuda'
        assert attention_mask.device.type == "cuda"
        assert pixel_values.device.type == "cuda"
        batch = {
            "responses": response,
            "input_ids": idx,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "action": actions,
        }

        return batch

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Generate sequences"""
        # make sampling args can be overriden by inputs
        do_sample = prompts.meta_info["do_sample"]
        temperature = prompts.meta_info["temperature"]
        max_prompt_length = prompts.meta_info["prompt_length"]
        # TODO: split into micro-batches
        task_descriptions = prompts.non_tensor_batch["task_descriptions"]
        images_and_states = {"full_image": prompts.batch["full_image"]}
        vla_input = process_input(task_descriptions, images_and_states, self.processor)

        vla_output = self._generate_one_step(vla_input, do_sample, temperature, max_prompt_length)
        # batch = TensorDict(vla_output)
        batch = DataProto.from_dict(tensors=vla_output)
        return batch

    async def update_weights(self, weights, **kwargs):
        # new_state_dict = {name: param for name, param in weights}
        # try:
        #     self.module.load_state_dict(new_state_dict)
        # except Exception as e:
        #     raise e
        # TODO(caiyunke.astra): implement weight update for seperate rollout worker
        pass

    def release(self):
        self.module.cpu()

    def resume(self, tags: list[str]):
        self.module.to(torch.cuda.current_device())
