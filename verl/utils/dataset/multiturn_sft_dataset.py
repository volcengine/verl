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
import json
import os
import random
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset


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
    return F.pad(tensors, pad_tuple, "constant", pad_token_id)


ACTION_DIM = 7
NUM_ACTIONS_CHUNK = 8

EMPTY_RESPONSE_TOKEN_ID = 29871
DEFAULT_PROMPT_TEMPLATE = "In: What action should the robot take to {instruction}?\nOut:"


def _center_crop(image: Image.Image, crop_scale: float = 0.9) -> Image.Image:
    if crop_scale >= 1.0:
        return image
    width, height = image.size
    scale = max(min(crop_scale, 1.0), 0.0) ** 0.5
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    cropped = image.crop((left, top, left + new_width, top + new_height))
    resample = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
    return cropped.resize((width, height), resample)


def _load_task_descriptions(tasks_path: str, instruction_key: Optional[str] = None) -> list[str]:
    candidates = [instruction_key] if instruction_key else []
    candidates.extend(
        [
            "language_instruction",
            "instruction",
            "description",
            "task",
            "task_description",
            "prompt",
        ]
    )

    descriptions: list[str] = []
    with open(tasks_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                descriptions.append(line)
                continue

            if isinstance(payload, str):
                descriptions.append(payload)
                continue

            for key in candidates:
                if key and key in payload and payload[key]:
                    descriptions.append(str(payload[key]))
                    break
            else:
                descriptions.append(json.dumps(payload))
    return descriptions


class MultiTurnSFTDataset(Dataset):
    # parquet_files=test_file, tokenizer=tokenizer, config=config
    def __init__(
        self,
        parquet_files: str,
        processor: str,
        config: None,
        split: str = "train",
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        instruction_key: Optional[str] = None,
        tasks_subdir: str = "meta/tasks.jsonl",
        include_wrist_image: bool = False,
        center_crop: bool = True,
        crop_scale: float = 0.9,
        action_chunks_len: int = NUM_ACTIONS_CHUNK,
        action_chunk_stride: int = 1,
        n_action_bins: int = 256,
        max_prompt_length: Optional[int] = 256,
        norm_stats_path: Optional[str] = None,
        unnorm_key: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.prompt_template = prompt_template
        self.include_wrist_image = include_wrist_image
        self.center_crop = center_crop
        self.crop_scale = crop_scale
        self.action_chunks_len = action_chunks_len
        self.action_chunk_stride = action_chunk_stride
        self.max_prompt_length = max_prompt_length
        if isinstance(parquet_files, list):
            parquet_path = parquet_files[0]
        else:
            parquet_path = parquet_files

        dataset_dict = load_dataset(parquet_path)
        if split not in dataset_dict:
            raise ValueError(f"Split '{split}' not found in dataset. Available splits: {list(dataset_dict.keys())}")
        self._dataset = dataset_dict[split]
        # self._dataset = [i for i in self._dataset if i['task_index'] == 5]
        # self._dataset = self._dataset.filter(lambda x: x['task_index'] == 5)
        mask = np.array(self._dataset["task_index"]) == 5
        indices = np.where(mask)[0]
        self._dataset = self._dataset.select(indices)

        tasks_path = os.path.join(parquet_path, tasks_subdir)
        if not os.path.exists(tasks_path):
            raise FileNotFoundError(f"Task description file not found at {tasks_path}")
        self.task_descriptions = _load_task_descriptions(tasks_path, instruction_key)

        self.processor = processor
        self.vocab_size = int(self.processor.tokenizer.vocab_size)
        self.pad_token_id = int(self.processor.tokenizer.pad_token_id)
        if self.pad_token_id is None:
            raise ValueError("Tokenizer must define pad_token_id")

        self.bin_edges = np.linspace(-1.0, 1.0, n_action_bins + 1)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2.0

        norm_stats_path = os.path.join(parquet_path, "meta/stats.json")
        unnorm_key = "libero_10_no_noops"
        self.vla_norm_stats = {
            "libero_10_no_noops": {
                "action": {
                    "mean": [
                        0.01820324920117855,
                        0.05858374014496803,
                        -0.05592384561896324,
                        0.004626928828656673,
                        0.00289608770981431,
                        -0.007673131301999092,
                        0.5457824468612671,
                    ],
                    "std": [
                        0.2825464606285095,
                        0.35904666781425476,
                        0.3673802614212036,
                        0.03770702704787254,
                        0.05429719388484955,
                        0.08725254982709885,
                        0.49815231561660767,
                    ],
                    "max": [0.9375, 0.9375, 0.9375, 0.30000001192092896, 0.29357144236564636, 0.375, 1.0],
                    "min": [
                        -0.9375,
                        -0.9375,
                        -0.9375,
                        -0.23642857372760773,
                        -0.3053571283817291,
                        -0.3675000071525574,
                        0.0,
                    ],
                    "q01": [
                        -0.6348214149475098,
                        -0.7741071581840515,
                        -0.7633928656578064,
                        -0.09749999642372131,
                        -0.14819999992847435,
                        -0.2742857038974762,
                        0.0,
                    ],
                    "q99": [
                        0.7714285850524902,
                        0.8464285731315613,
                        0.9375,
                        0.13928571343421936,
                        0.15964286029338837,
                        0.3246428668498993,
                        1.0,
                    ],
                    "mask": [True, True, True, True, True, True, False],
                },
                "proprio": {
                    "mean": [
                        -0.04190658777952194,
                        0.03539430722594261,
                        0.8257141709327698,
                        2.908308267593384,
                        -0.5562185049057007,
                        -0.16649018228054047,
                        0.028316624462604523,
                        -0.028561657294631004,
                    ],
                    "std": [
                        0.10743364691734314,
                        0.14424669742584229,
                        0.2572328448295593,
                        0.3441362977027893,
                        1.234421730041504,
                        0.3579835891723633,
                        0.013308707624673843,
                        0.013174631632864475,
                    ],
                    "max": [
                        0.21031762659549713,
                        0.39128610491752625,
                        1.3332009315490723,
                        3.6714255809783936,
                        3.560650587081909,
                        1.386339545249939,
                        0.04160946607589722,
                        0.0013633022317662835,
                    ],
                    "min": [
                        -0.4828203022480011,
                        -0.3255046010017395,
                        0.445506751537323,
                        1.1321442127227783,
                        -3.641430377960205,
                        -1.842738389968872,
                        -0.0010040868073701859,
                        -0.04111652821302414,
                    ],
                    "q01": [
                        -0.3899900782108307,
                        -0.2838300323486328,
                        0.44795057058334353,
                        1.8810229921340942,
                        -2.886677579879761,
                        -1.1599004411697387,
                        0.002066459748893976,
                        -0.04001387819647789,
                    ],
                    "q99": [
                        0.1530261474847791,
                        0.32915401458740223,
                        1.2546923208236693,
                        3.303542451858519,
                        2.7496529006957933,
                        0.6893712210655194,
                        0.040048558115959164,
                        -0.0017598449345678235,
                    ],
                },
                "num_transitions": 101469,
                "num_trajectories": 379,
            }
        }
        self._setup_action_stats(norm_stats_path, unnorm_key)

    def _setup_action_stats(self, norm_stats_path: Optional[str], unnorm_key: Optional[str]) -> None:
        self._action_low = None
        self._action_high = None
        self._action_mask = None
        action_stats = self.vla_norm_stats[unnorm_key]["action"]
        high_key = "q99" if "q99" in action_stats else "max"
        low_key = "q01" if "q01" in action_stats else "min"
        self._action_high = np.array(action_stats[high_key], dtype=np.float32)
        self._action_low = np.array(action_stats[low_key], dtype=np.float32)
        mask = action_stats.get("mask", None)
        self._action_mask = np.array(mask, dtype=bool) if mask is not None else None

    def __len__(self) -> int:
        return len(self._dataset)

    def _normalize_actions(self, actions: np.ndarray) -> np.ndarray:
        if self._action_high is None or self._action_low is None:
            return np.clip(actions, -1.0, 1.0)
        span = self._action_high - self._action_low
        span = np.where(span == 0, 1e-6, span)
        normalized = 2.0 * (actions - self._action_low) / span - 1.0
        if self._action_mask is not None:
            normalized = np.where(self._action_mask, normalized, actions)
        return np.clip(normalized, -1.0, 1.0)

    def _actions_to_token_ids(self, actions: Sequence[Sequence[float]]) -> torch.LongTensor:
        action_array = np.asarray(actions, dtype=np.float32).reshape(-1, ACTION_DIM)
        normalized = self._normalize_actions(action_array)
        diff = np.abs(normalized[..., None] - self.bin_centers)
        bin_indices = diff.argmin(axis=-1)
        token_ids = self.vocab_size - (bin_indices + 1)
        return torch.tensor(token_ids.reshape(-1), dtype=torch.long)

    def _prepare_prompt(self, task_index: int) -> str:
        if task_index >= len(self.task_descriptions):
            raise IndexError(f"Task index {task_index} exceeds descriptions (len={len(self.task_descriptions)})")
        instruction = self.task_descriptions[task_index]
        return self.prompt_template.format(instruction=instruction, task=instruction)

    def _prepare_images(self, frame: dict[str, Any]) -> tuple[Image.Image, Optional[Image.Image]]:
        image = frame["image"]
        wrist = frame.get("wrist_image") if self.include_wrist_image else None
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.array(image))
        if wrist is not None and not isinstance(wrist, Image.Image):
            wrist = Image.fromarray(np.array(wrist))
        if self.center_crop:
            image = _center_crop(image, self.crop_scale)
            if wrist is not None:
                wrist = _center_crop(wrist, self.crop_scale)
        return image, wrist

    def _tokenize(
        self, prompt: str, image: Image.Image, wrist: Optional[Image.Image], action_ids: torch.tensor
    ) -> dict[str, torch.Tensor]:
        action_ids = action_ids.reshape(1, -1)
        primary = self.processor(prompt, image, return_tensors="pt")
        input_ids = primary["input_ids"]
        attention_mask = primary["attention_mask"]
        pixel_values = primary["pixel_values"]

        if wrist is not None:
            wrist_batch = self.processor(prompt, wrist, return_tensors="pt")
            pixel_values = torch.cat([pixel_values, wrist_batch["pixel_values"]], dim=1)

        if input_ids[0, -1].item() != EMPTY_RESPONSE_TOKEN_ID:
            pad = torch.tensor([[EMPTY_RESPONSE_TOKEN_ID]], dtype=input_ids.dtype)
            mask_pad = torch.ones((1, 1), dtype=attention_mask.dtype)
            input_ids = torch.cat([input_ids, pad], dim=1)
            attention_mask = torch.cat([attention_mask, mask_pad], dim=1)
        labels = torch.ones_like(input_ids) * -100

        input_ids = torch.cat((input_ids, action_ids), dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(action_ids)], dim=1)
        labels = torch.cat((labels, action_ids), dim=1)

        if self.max_prompt_length is not None:
            input_ids = pad_sequence_to_length(
                input_ids,
                max_seq_len=self.max_prompt_length,
                pad_token_id=self.pad_token_id,
                left_pad=False,
            )
            attention_mask = pad_sequence_to_length(
                attention_mask,
                max_seq_len=self.max_prompt_length,
                pad_token_id=0,
                left_pad=False,
            )
            labels = pad_sequence_to_length(
                labels,
                max_seq_len=self.max_prompt_length,
                pad_token_id=-100,
                left_pad=False,
            )

        return {
            "input_ids": input_ids.long(),
            "attention_mask": attention_mask.long(),
            "pixel_values": pixel_values.float(),
            "labels": labels.long(),
        }

    def get_actions(self, first_frame, index):
        task_index = int(first_frame.get("task_index", 0))
        episode_index = int(first_frame.get("episode_index", 0))
        actions: list[Sequence[float]] = [first_frame["actions"]]
        fill_static = False
        zero_action = [0.0] * ACTION_DIM
        valid_steps = 1
        for offset in range(1, self.action_chunks_len):
            if fill_static:
                actions.append(zero_action)
                continue
            candidate_idx = index + offset
            if candidate_idx >= len(self._dataset):
                actions.append(zero_action)
                continue
            candidate = self._dataset[candidate_idx]
            same_episode = int(candidate.get("episode_index", episode_index)) == episode_index
            same_task = int(candidate.get("task_index", task_index)) == task_index
            if same_episode and same_task:
                actions.append(candidate["actions"])
                valid_steps += 1
            else:
                actions.append(zero_action)
                fill_static = True
        if valid_steps != len(actions):
            print(f"not enough actions, skip at {index}!")
            return None, None

        actions_array = np.asarray(actions, dtype=np.float32)
        action_token_ids = self._actions_to_token_ids(actions_array)
        return actions_array, action_token_ids

    def __getitem__(self, index: int) -> dict[str, Any]:
        index = int(index)
        first_frame = self._dataset[index]
        task_index = int(first_frame.get("task_index", 0))
        episode_index = int(first_frame.get("episode_index", 0))

        actions_array, action_token_ids = self.get_actions(first_frame, index)
        if actions_array is None:
            return self[random.randint(0, len(self) - 1)]
        prompt = self._prepare_prompt(task_index)
        image, wrist = self._prepare_images(first_frame)
        tokenized = self._tokenize(prompt, image, wrist, action_token_ids)

        output: dict[str, Any] = {
            "input_ids": tokenized["input_ids"][0],
            "attention_mask": tokenized["attention_mask"][0],
            "pixel_values": tokenized["pixel_values"][0],
            "labels": tokenized["labels"][0],
            "responses": action_token_ids.reshape(-1),
            "action_token_ids": action_token_ids,
            "actions": torch.tensor(actions_array, dtype=torch.float32),
            "state": torch.tensor(np.asarray(first_frame.get("state", []), dtype=np.float32)),
            "task_index": torch.tensor(task_index, dtype=torch.long),
            "episode_index": torch.tensor(episode_index, dtype=torch.long),
            "frame_index": torch.tensor(int(first_frame.get("frame_index", 0)), dtype=torch.long),
            "timestamp": torch.tensor(float(first_frame.get("timestamp", 0.0)), dtype=torch.float32),
            "prompt": prompt,
        }
        return output


if __name__ == "__main__":
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained("Haozhan72/Openvla-oft-SFT-libero10-traj1", trust_remote_code=True)
    parquet_path = "$DATA_PATH/physical-intelligence/libero/"
    dataset = MultiTurnSFTDataset(parquet_path, processor, None)
    actions = torch.tensor(
        [
            [
                1.31565308e-02,
                -7.70929637e-01,
                -7.60057782e-01,
                9.75000610e-03,
                -3.14699528e-04,
                -5.35502744e-03,
                -1.72549020e-01,
            ],
            [
                4.62447664e-02,
                -2.73897088e-02,
                -7.60057782e-01,
                9.75000610e-03,
                -3.14699528e-04,
                -5.35502744e-03,
                -1.72549020e-01,
            ],
            [
                4.62447664e-02,
                -2.73897088e-02,
                -7.60057782e-01,
                9.75000610e-03,
                -3.14699528e-04,
                -5.35502744e-03,
                -1.72549020e-01,
            ],
            [
                4.62447664e-02,
                -2.73897088e-02,
                2.03518908e-02,
                -9.70357107e-02,
                -8.76528812e-03,
                1.34348888e-02,
                -1.01960784e-01,
            ],
            [
                -5.30199402e-02,
                1.07405440e-02,
                2.03518908e-02,
                -9.70357107e-02,
                -8.76528812e-03,
                1.34348888e-02,
                -1.01960784e-01,
            ],
            [
                -5.30199402e-02,
                -2.73897088e-02,
                2.03518908e-02,
                -9.70357107e-02,
                -8.76528812e-03,
                1.34348888e-02,
                -1.01960784e-01,
            ],
            [
                -5.30199402e-02,
                -2.73897088e-02,
                2.03518908e-02,
                -9.70357107e-02,
                -8.76528812e-03,
                1.34348888e-02,
                -1.01960784e-01,
            ],
            [
                -5.30199402e-02,
                1.07405440e-02,
                2.03518908e-02,
                -9.70357107e-02,
                -8.76528812e-03,
                1.34348888e-02,
                -1.01960784e-01,
            ],
        ]
    )
    print(dataset[0])
    print(dataset._actions_to_token_ids(actions))
    print(dataset[0]["pixel_values"].mean(dim=(-1, -2)))
