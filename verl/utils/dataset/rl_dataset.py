# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

import copy
import logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict, Union

import datasets  # type: ignore
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__name__)


class TruncationStrategy(Enum):
    """Enumeration of available truncation strategies."""

    LEFT = "left"
    RIGHT = "right"
    MIDDLE = "middle"
    ERROR = "error"


class MessageContent(TypedDict):
    """Type definition for message content items."""

    type: str  # "text", "image", "video"
    text: Optional[str]


class ChatMessage(TypedDict):
    """Type definition for chat messages."""

    role: str  # "user", "assistant", "system"
    content: Union[str, List[MessageContent]]


class ExtraInfo(TypedDict, total=False):
    """Type definition for extra_info field. All fields are optional."""

    index: int
    tools_kwargs: Dict[str, Any]
    interaction_kwargs: Dict[str, Any]
    need_tools_kwargs: bool


class RawDataRow(TypedDict, total=False):
    """Type definition for raw data row from dataset."""

    prompt: List[ChatMessage]
    images: Optional[List[str]]
    videos: Optional[List[str]]
    extra_info: Optional[ExtraInfo]


@dataclass
class ProcessingContext:
    """Context object to pass data between processing steps."""

    raw_row: RawDataRow
    messages: List[ChatMessage]
    raw_prompt: str
    model_inputs: Dict[str, Any]
    multi_modal_data: Optional[Dict[str, Any]] = None
    extra_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CoreTensors:
    """Core tensor outputs from processing."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    raw_prompt_ids: List[int]


@dataclass
class ItemMetadata:
    """Metadata extracted from the data item."""

    index: int = 0
    tools_kwargs: Dict[str, Any] = field(default_factory=dict)
    interaction_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptionalOutputs:
    """Optional outputs based on configuration."""

    raw_prompt: Optional[List[ChatMessage]] = None
    full_prompts: Optional[str] = None
    multi_modal_data: Optional[Dict[str, Any]] = None
    multi_modal_inputs: Optional[Dict[str, Any]] = None


@dataclass
class ProcessedDataItem:
    """Complete processed data item with all components."""

    core: CoreTensors
    metadata: ItemMetadata
    optional: OptionalOutputs

    def to_dict(self) -> Dict[str, Any]:
        """Convert to the dict format expected by the rest of the system."""
        result = {
            "input_ids": self.core.input_ids,
            "attention_mask": self.core.attention_mask,
            "position_ids": self.core.position_ids,
            "raw_prompt_ids": self.core.raw_prompt_ids,
            "index": self.metadata.index,
            "tools_kwargs": self.metadata.tools_kwargs,
            "interaction_kwargs": self.metadata.interaction_kwargs,
        }

        # Add optional fields only if they exist
        if self.optional.raw_prompt is not None:
            result["raw_prompt"] = self.optional.raw_prompt
        if self.optional.full_prompts is not None:
            result["full_prompts"] = self.optional.full_prompts
        if self.optional.multi_modal_data is not None:
            result["multi_modal_data"] = self.optional.multi_modal_data
        if self.optional.multi_modal_inputs is not None:
            result["multi_modal_inputs"] = self.optional.multi_modal_inputs

        return result


def collate_fn(data_list: list[dict]) -> dict:
    """
    Collate a batch of sample dicts into batched tensors and arrays.

    Args:
        data_list: List of dicts mapping feature names to torch.Tensor or other values.

    Returns:
        Dict where tensor entries are stacked into a torch.Tensor of shape
        (batch_size, \\*dims) and non-tensor entries are converted to
        np.ndarray of dtype object with shape (batch_size,).
    """
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}


class RLHFDataset(Dataset):
    """
    Load and preprocess RLHF data from Parquet files.

    - Caches files locally.
    - Reads into a HuggingFace Dataset and tokenizes prompts.
    - Optionally handles images/videos via a ProcessorMixin.
    - Filters prompts over a max length.
    - Supports resuming from checkpoints.

    Args:
        data_files (str or list): Path(s) to Parquet file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
    """

    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if not isinstance(data_files, (List, ListConfig)):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)

        cpu_count = os.cpu_count() or 1
        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, cpu_count // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())
        self.use_shm = config.get("use_shm", False)
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.serialize_dataset = False
        self.return_multi_modal_inputs = config.get("return_multi_modal_inputs", True)

        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir, use_shm=self.use_shm)

    def _read_files_and_tokenize(self) -> None:
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f"dataset len: {len(self.dataframe)}")

        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)

    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset = None):
        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            processor = self.processor
            prompt_key = self.prompt_key
            image_key = self.image_key
            video_key = self.video_key

            if processor is not None:
                from verl.utils.dataset.vision_utils import process_image, process_video

                def doc2len(doc) -> int:
                    messages = self._build_messages(doc)
                    raw_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                    images = (
                        [process_image(image) for image in messages.pop(image_key)] if image_key in messages else None
                    )
                    videos = (
                        [process_video(video) for video in messages.pop(video_key)] if video_key in messages else None
                    )

                    return len(processor(text=[raw_prompt], images=images, videos=videos)["input_ids"][0])

            else:

                def doc2len(doc) -> int:
                    return len(tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True))

            dataframe = dataframe.filter(
                lambda doc: doc2len(doc) <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )

            print(f"filter dataset len: {len(dataframe)}")
        return dataframe

    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_files")
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")

    def __len__(self):
        return len(self.dataframe)

    def _build_messages(self, example: dict):
        messages: list = example.pop(self.prompt_key)

        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                content_list = []
                segments = re.split("(<image>|<video>)", content)
                segments = [item for item in segments if item != ""]
                for segment in segments:
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    else:
                        content_list.append({"type": "text", "text": segment})

                message["content"] = content_list

        return messages

    def __getitem__(self, item: int) -> Dict[str, Any]:
        """
        Get a processed data item by index.

        Note: We return raw_input_ids so it can be combined with other chat templates.
        """
        # Step 1: Extract raw data and build processing context
        raw_row: RawDataRow = self.dataframe[item]
        context = self._create_processing_context(raw_row)

        # Step 2: Process multimodal or text-only inputs
        if self.processor is not None:
            self._process_multimodal_inputs(context)
        else:
            self._process_text_only_inputs(context)

        # Step 3: Create core tensors
        self._create_core_tensors(context)

        # Step 4: Handle position IDs (special case for Qwen2VL)
        self._compute_position_ids(context)

        # Step 5: Process raw prompt IDs with truncation
        self._process_raw_prompt_ids(context)

        # Step 6: Build structured result and convert to dict
        processed_item = self._build_processed_item(context)
        return processed_item.to_dict()

    def _create_processing_context(self, raw_row: RawDataRow) -> ProcessingContext:
        """Create processing context from raw data row."""
        # Make a copy to avoid modifying the original
        row_copy = dict(raw_row)
        messages = self._build_messages(row_copy)
        extra_info = self._extract_extra_info(raw_row)

        return ProcessingContext(
            raw_row=row_copy,  # type: ignore[arg-type]
            messages=messages,
            raw_prompt="",  # Will be set in processing steps
            model_inputs={},
            extra_info=extra_info,
        )

    def _extract_extra_info(self, raw_row: RawDataRow) -> Dict[str, Any]:
        """Safely extract extra_info with defaults."""
        extra_info_raw = raw_row.get("extra_info")
        if extra_info_raw is None:
            extra_info_raw = {}

        return {
            "index": extra_info_raw.get("index", 0),
            "tools_kwargs": extra_info_raw.get("tools_kwargs", {}),
            "interaction_kwargs": extra_info_raw.get("interaction_kwargs", {}),
            "need_tools_kwargs": extra_info_raw.get("need_tools_kwargs", self.need_tools_kwargs),
        }

    def _process_multimodal_inputs(self, context: ProcessingContext) -> None:
        """Process multimodal inputs using the processor."""
        from verl.utils.dataset.vision_utils import process_image, process_video

        if self.processor is not None:
            context.raw_prompt = self.processor.apply_chat_template(
                context.messages, add_generation_prompt=True, tokenize=False
            )

            multi_modal_data = {}
            images = None
            videos = None

            # Process images
            if self.image_key in context.raw_row and context.raw_row.get(self.image_key) is not None:
                images_data = context.raw_row.pop(self.image_key, None)  # type: ignore[misc]
                if images_data is not None:
                    images = [process_image(image) for image in images_data]
                    # Use "image" key for vllm compatibility
                    multi_modal_data["image"] = images

            # Process videos
            if self.video_key in context.raw_row and context.raw_row.get(self.video_key) is not None:
                videos_data = context.raw_row.pop(self.video_key, None)  # type: ignore[misc]
                if videos_data is not None:
                    videos = [process_video(video) for video in videos_data]
                    # Use "video" key for vllm compatibility
                    multi_modal_data["video"] = [video.numpy() for video in videos]

            context.model_inputs = self.processor(
                text=[context.raw_prompt], images=images, videos=videos, return_tensors="pt"
            )
            context.multi_modal_data = multi_modal_data

    def _process_text_only_inputs(self, context: ProcessingContext) -> None:
        """Process text-only inputs using the tokenizer."""
        context.raw_prompt = self.tokenizer.apply_chat_template(
            context.messages, add_generation_prompt=True, tokenize=False
        )
        context.model_inputs = self.tokenizer(context.raw_prompt, return_tensors="pt", add_special_tokens=False)

    def _create_core_tensors(self, context: ProcessingContext) -> None:
        """Create and postprocess core tensors."""
        input_ids = context.model_inputs.pop("input_ids")
        attention_mask = context.model_inputs.pop("attention_mask")

        # Remove unused fields
        context.model_inputs.pop("second_per_grid_ts", None)

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        # Store processed tensors back in model_inputs for easy access
        context.model_inputs["input_ids"] = input_ids[0]
        context.model_inputs["attention_mask"] = attention_mask[0]

    def _compute_position_ids(self, context: ProcessingContext) -> None:
        """Compute position IDs, with special handling for Qwen2VL."""
        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = get_rope_index(
                self.processor,
                input_ids=context.model_inputs["input_ids"],
                image_grid_thw=context.model_inputs.get("image_grid_thw"),
                video_grid_thw=context.model_inputs.get("video_grid_thw"),
                second_per_grid_ts=context.model_inputs.get("second_per_grid_ts"),
                attention_mask=context.model_inputs["attention_mask"],
            )
        else:
            position_ids = compute_position_id_with_mask(context.model_inputs["attention_mask"].unsqueeze(0))[0]

        context.model_inputs["position_ids"] = position_ids

    def _process_raw_prompt_ids(self, context: ProcessingContext) -> None:
        """Process raw prompt IDs with truncation."""
        raw_prompt_ids = self.tokenizer.encode(context.raw_prompt, add_special_tokens=False)

        if len(raw_prompt_ids) > self.max_prompt_length:
            try:
                strategy = TruncationStrategy(self.truncation)
            except ValueError:
                raise ValueError(
                    f"Invalid truncation strategy: '{self.truncation}'. "
                    f"Valid options are: {[s.value for s in TruncationStrategy]}"
                )


        context.model_inputs["raw_prompt_ids"] = raw_prompt_ids

    def _apply_truncation(self, raw_prompt_ids: List[int], strategy: TruncationStrategy) -> List[int]:
        """Apply truncation strategy to prompt IDs."""
        max_length = self.max_prompt_length
        if max_length is None:
            return raw_prompt_ids

        # Ensure max_length is an int for mypy
        assert isinstance(max_length, int)

        if strategy == TruncationStrategy.LEFT:
            return raw_prompt_ids[-max_length:]
        elif strategy == TruncationStrategy.RIGHT:
            return raw_prompt_ids[:max_length]
        elif strategy == TruncationStrategy.MIDDLE:
            left_half = max_length // 2
            right_half = max_length - left_half
            return raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
        elif strategy == TruncationStrategy.ERROR:
            raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} exceeds max length {max_length}")

    def _build_processed_item(self, context: ProcessingContext) -> ProcessedDataItem:
        """Build the structured ProcessedDataItem from context."""
        # Create core tensors
        core = CoreTensors(
            input_ids=context.model_inputs["input_ids"],
            attention_mask=context.model_inputs["attention_mask"],
            position_ids=context.model_inputs["position_ids"],
            raw_prompt_ids=context.model_inputs["raw_prompt_ids"],
        )

        # Create metadata
        metadata = ItemMetadata(
            index=context.extra_info["index"],
            tools_kwargs=context.extra_info["tools_kwargs"],
            interaction_kwargs=context.extra_info["interaction_kwargs"],
        )

        # Create optional outputs
        optional = OptionalOutputs()

        if self.return_raw_chat:
            optional.raw_prompt = context.messages

        if self.return_full_prompt:
            optional.full_prompts = context.raw_prompt

        if context.multi_modal_data is not None:
            optional.multi_modal_data = context.multi_modal_data

            if self.return_multi_modal_inputs:
                # Create clean dict without training-irrelevant fields
                multi_modal_inputs = dict(context.model_inputs)
                multi_modal_inputs.pop("input_ids", None)
                multi_modal_inputs.pop("attention_mask", None)
                multi_modal_inputs.pop("position_ids", None)
                multi_modal_inputs.pop("raw_prompt_ids", None)
                multi_modal_inputs.pop("second_per_grid_ts", None)
                optional.multi_modal_inputs = multi_modal_inputs

        # Validate tools_kwargs if needed
        if context.extra_info.get("need_tools_kwargs", False) and not context.extra_info.get("tools_kwargs", {}):
            logger.warning(
                "tools_kwargs is empty for index %s, data source: %s",
                context.extra_info.get("index", 0),
                context.raw_row.get("data_source", "unknown"),
            )

        return ProcessedDataItem(core=core, metadata=metadata, optional=optional)

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()
