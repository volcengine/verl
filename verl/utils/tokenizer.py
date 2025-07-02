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
"""Utils for tokenization."""

import warnings
import re
from transformers import AutoConfig

__all__ = ["hf_tokenizer", "hf_processor"]


def set_pad_token_id(tokenizer):
    """Set pad_token_id to eos_token_id if it is None.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to be set.

    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        warnings.warn(f"tokenizer.pad_token_id is None. Now set to {tokenizer.eos_token_id}", stacklevel=1)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        warnings.warn(f"tokenizer.pad_token is None. Now set to {tokenizer.eos_token}", stacklevel=1)


def hf_tokenizer(name_or_path, correct_pad_token=True, correct_gemma2=True, **kwargs):
    """Create a huggingface pretrained tokenizer which correctness handles eos and pad tokens.

    Args:

        name (str): The name of the tokenizer.
        correct_pad_token (bool): Whether to correct the pad token id.
        correct_gemma2 (bool): Whether to correct the gemma2 tokenizer.

    Returns:

        transformers.PreTrainedTokenizer: The pretrained tokenizer.

    """
    from transformers import AutoTokenizer

    if correct_gemma2 and isinstance(name_or_path, str) and "gemma-2-2b-it" in name_or_path:
        # the EOS token in gemma2 is ambiguious, which may worsen RL performance.
        # https://huggingface.co/google/gemma-2-2b-it/commit/17a01657f5c87135bcdd0ec7abb4b2dece04408a
        warnings.warn(
            "Found gemma-2-2b-it tokenizer. Set eos_token and eos_token_id to <end_of_turn> and 107.", stacklevel=1
        )
        kwargs["eos_token"] = "<end_of_turn>"
        kwargs["eos_token_id"] = 107
    trust_remote_code=kwargs.get("trust_remote_code", False)
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    config = AutoConfig.from_pretrained(name_or_path, trust_remote_code=trust_remote_code)
    if re.match("internvl", config.model_type):
        tokenizer.context_image_token = "<IMG_CONTEXT>"
        tokenizer.end_image_token="</img>"
        tokenizer.start_image_token="<img>"
        tokenizer.video_token = "<video>"
        tokenizer.context_image_token_id = tokenizer.convert_tokens_to_ids(tokenizer.context_image_token) #for transformers >= 4.52.2
        print("tokenizer.context_image_token_id:", tokenizer.context_image_token_id)
        tokenizer.chat_template="""{% for message in messages %}{{'<|im_start|>' + message['role'] + ''}}{% if message['content'] is string %}{{ message['content'] }}{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' %}{{ '<image>' }}{% elif content['type'] == 'video' %}{{ '<video>' }}{% elif content['type'] == 'text' %}{{ content['text'] }}{% endif %}{% endfor %}{% endif %}{{'<|im_end|>'}}{% endfor %}{% if add_generation_prompt %}{{'<|im_start|>assistant' }}{% endif %}"""
    if correct_pad_token:
        set_pad_token_id(tokenizer)
    return tokenizer


def hf_processor(name_or_path, **kwargs):
    """Create a huggingface processor to process multimodal data.

    Args:
        name_or_path (str): The name of the processor.

    Returns:
        transformers.ProcessorMixin: The pretrained processor.
    """
    from transformers import AutoProcessor
    trust_remote_code=kwargs.get("trust_remote_code", False)
    config = AutoConfig.from_pretrained(name_or_path, trust_remote_code=trust_remote_code)
    try:
        if re.match("internvl", config.model_type, re.IGNORECASE):
            print("InterVLProcessor initilizing")
            IMAGENET_MEAN = [0.485, 0.456, 0.406]
            IMAGENET_STD = [0.229, 0.224, 0.225]
            from transformers.models.internvl import InternVLProcessor
            from transformers.models.got_ocr2 import GotOcr2ImageProcessorFast
            from transformers.models.internvl.video_processing_internvl import InternVLVideoProcessor
            image_processor = GotOcr2ImageProcessorFast(
                crop_to_patches=False,
                data_format="channels_first",
                default_to_square=True,
                do_center_crop=None,
                do_convert_rgb=True,
                do_normalize=True,
                do_rescale=True,
                do_resize=True,
                rescale_factor=0.00392156862745098,
                size={"height":448, "width": 448},
                max_patches=12,
                min_patches=1,
                resample=3,
                return_tensors=None,
                image_mean=IMAGENET_MEAN,
                image_std=IMAGENET_STD
            )
            video_processor = InternVLVideoProcessor() #for transformers>=4.52.2
            tokenizer = hf_tokenizer(name_or_path, trust_remote_code=trust_remote_code)
            processor = InternVLProcessor(
                image_processor=image_processor,
                image_seq_length=256,
                tokenizer=tokenizer,
                chat_template=tokenizer.chat_template,
                video_processor=video_processor
            )
            print("Training the InternVL series")
        else:
            processor = AutoProcessor.from_pretrained(name_or_path, **kwargs)

    except Exception as e:
        processor = None
        # TODO(haibin.lin): try-catch should be removed after adding transformer version req to setup.py to avoid
        # silent failure
        warnings.warn(f"Failed to create processor: {e}. This may affect multimodal processing", stacklevel=1)
    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/auto/processing_auto.py#L344
    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None
    return processor
