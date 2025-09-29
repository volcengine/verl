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
        include_wrist_image: bool = True,
        center_crop: bool = True,
        crop_scale: float = 0.9,
        action_chunks_len: int = NUM_ACTIONS_CHUNK,
        action_chunk_stride: int = 1,
        n_action_bins: int = 256,
        max_prompt_length: Optional[int] = 512,
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

        self._setup_action_stats(norm_stats_path, unnorm_key)

    def _setup_action_stats(self, norm_stats_path: Optional[str], unnorm_key: Optional[str]) -> None:
        self._action_low = None
        self._action_high = None
        self._action_mask = None
        if not norm_stats_path:
            return
        with open(norm_stats_path, encoding="utf-8") as f:
            stats = json.load(f)
        if unnorm_key is None:
            unnorm_key = next(iter(stats.keys()))
        if unnorm_key not in stats:
            raise ValueError(f"unnorm_key '{unnorm_key}' not found in statistics keys {list(stats.keys())}")
        action_stats = stats[unnorm_key]["action"]
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

    def _tokenize(self, prompt: str, image: Image.Image, wrist: Optional[Image.Image]) -> dict[str, torch.Tensor]:
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

        if self.max_prompt_length is not None:
            input_ids = pad_sequence_to_length(
                input_ids,
                max_seq_len=self.max_prompt_length,
                pad_token_id=self.pad_token_id,
                left_pad=True,
            )
            attention_mask = pad_sequence_to_length(
                attention_mask,
                max_seq_len=self.max_prompt_length,
                pad_token_id=0,
                left_pad=True,
            )

        return {
            "input_ids": input_ids.long(),
            "attention_mask": attention_mask.long(),
            "pixel_values": pixel_values.float(),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        index = int(index)
        first_frame = self._dataset[index]
        task_index = int(first_frame.get("task_index", 0))
        episode_index = int(first_frame.get("episode_index", 0))

        prompt = self._prepare_prompt(task_index)
        image, wrist = self._prepare_images(first_frame)
        tokenized = self._tokenize(prompt, image, wrist)

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

        actions_array = np.asarray(actions, dtype=np.float32)
        action_token_ids = self._actions_to_token_ids(actions_array)

        output: dict[str, Any] = {
            "input_ids": tokenized["input_ids"][0],
            "attention_mask": tokenized["attention_mask"][0],
            "pixel_values": tokenized["pixel_values"][0],
            "responses": action_token_ids.reshape(-1),
            "action_token_ids": action_token_ids,
            "actions": torch.tensor(actions_array, dtype=torch.float32),
            "state": torch.tensor(np.asarray(first_frame.get("state", []), dtype=np.float32)),
            "task_index": torch.tensor(task_index, dtype=torch.long),
            "episode_index": torch.tensor(episode_index, dtype=torch.long),
            "frame_index": torch.tensor(int(first_frame.get("frame_index", 0)), dtype=torch.long),
            "timestamp": torch.tensor(float(first_frame.get("timestamp", 0.0)), dtype=torch.float32),
            "prompt": prompt,
            "finish_step": torch.tensor(valid_steps, dtype=torch.long),
        }
        return output
