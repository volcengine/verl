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

import os
import types
import warnings

__all__ = [
    "hf_tokenizer",
    "hf_processor",
    "normalize_token_ids",
    "build_multimodal_processor_inputs",
    "get_processor_token_id",
]


def normalize_token_ids(tokenized_output) -> list[int]:
    """Normalize tokenizer outputs into a flat ``list[int]``.

    This handles Transformers 4/5 differences where ``apply_chat_template(tokenize=True)``
    may return either ``list[int]`` or a ``BatchEncoding``/mapping with ``input_ids``.
    """

    token_ids = tokenized_output
    if isinstance(tokenized_output, dict):
        if "input_ids" in tokenized_output:
            token_ids = tokenized_output["input_ids"]
    elif hasattr(tokenized_output, "input_ids"):
        token_ids = tokenized_output.input_ids

    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()

    if isinstance(token_ids, tuple):
        token_ids = list(token_ids)

    if isinstance(token_ids, list) and len(token_ids) == 1 and isinstance(token_ids[0], list | tuple):
        token_ids = list(token_ids[0])

    if not isinstance(token_ids, list):
        raise TypeError(f"token_ids must be list-like token ids, got {type(token_ids).__name__}: {token_ids!r}")

    normalized_ids = []
    for idx, token_id in enumerate(token_ids):
        if hasattr(token_id, "item"):
            token_id = token_id.item()
        try:
            normalized_ids.append(int(token_id))
        except (TypeError, ValueError) as e:
            raise TypeError(f"token_id must be int-convertible, got {type(token_id).__name__}: {token_id!r}") from e
    return normalized_ids


def get_processor_token_id(processor, token_name: str) -> int | None:
    """Resolve a multimodal special token id from a processor.

    Newer processors may expose ``image_token``/``video_token`` strings instead
    of ``image_token_id``/``video_token_id`` integers. Fall back to tokenizer
    conversion so rollout code can stay processor-agnostic.
    """

    if processor is None:
        return None

    token_id_attr = f"{token_name}_token_id"
    token_id = getattr(processor, token_id_attr, None)
    if token_id is not None:
        return int(token_id)

    token_attr = f"{token_name}_token"
    token = getattr(processor, token_attr, None)
    tokenizer = getattr(processor, "tokenizer", None)
    if token is not None and tokenizer is not None:
        converted = tokenizer.convert_tokens_to_ids(token)
        if converted is not None:
            return int(converted)

    return None


def _split_videos_and_metadata(videos):
    if videos is None:
        return None, None
    videos = list(videos)
    if len(videos) > 0 and isinstance(videos[0], tuple):
        video_values, video_metadata = zip(*videos, strict=False)
        return list(video_values), list(video_metadata)
    return videos, None


def build_multimodal_processor_inputs(
    processor,
    *,
    text,
    images=None,
    videos=None,
    audio=None,
    mm_processor_kwargs=None,
    return_tensors: str = "pt",
):
    """Build kwargs for multimodal processor calls.

    This keeps the existing VL flow intact while extending it with audio-aware
    paths for processors that accept audio inputs.
    """
    processor_kwargs = dict(mm_processor_kwargs or {})
    if audio is not None and "sampling_rate" not in processor_kwargs:
        sampling_rate = getattr(getattr(processor, "feature_extractor", None), "sampling_rate", None)
        if sampling_rate is not None:
            processor_kwargs["sampling_rate"] = int(sampling_rate)

    videos, video_metadata = _split_videos_and_metadata(videos)
    processor_kwargs.setdefault("return_tensors", return_tensors)

    if video_metadata is not None:
        processor_kwargs.setdefault("video_metadata", video_metadata)
        processor_kwargs.setdefault("do_sample_frames", False)

    processor_inputs = {"text": text, "images": images, "videos": videos, **processor_kwargs}
    if audio is not None:
        processor_inputs["audio"] = audio

    return processor(**processor_inputs)


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
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    tokenizer = _honor_tokenizer_json(tokenizer, name_or_path, kwargs)
    if correct_pad_token:
        set_pad_token_id(tokenizer)
    return tokenizer


# spaces (single, double, trailing), newlines, a tab, digits, non-ASCII and CJK text, a decomposed accent and a
# ligature (normalizer differences)
_TOKENIZER_PROBE = (
    "Solve the  following step by step.\n\nThe last line: x = 3\t(12345.678) caf\u00e9 cafe\u0301 \ufb01ne "
    "\u4f60\u597d {7}  "
)
# ASCII text whose spacing a byte-level tokenizer reproduces exactly: double and trailing spaces, newlines, a tab
_SPACING_PROBE = "Solve the  following step by step.\n\nThe last line: x = 3\t(12345) {7}  "
# from_pretrained arguments that select which tokenizer.json is loaded
_TOKENIZER_FILE_KWARGS = ("subfolder", "revision", "cache_dir", "token", "local_files_only")


def _plain_ids(tokenizer, text=_TOKENIZER_PROBE):
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def _spacing_round_trip(tokenizer):
    return tokenizer.decode(_plain_ids(tokenizer, _SPACING_PROBE), clean_up_tokenization_spaces=False)


def _only_whitespace_removed(text, original=_SPACING_PROBE):
    """``text`` is ``original`` with some of its whitespace deleted and nothing else changed."""
    if text == original:
        return False
    position = 0
    for char in original:
        if position < len(text) and text[position] == char:
            position += 1
        elif not char.isspace():
            return False
    return position == len(text)


def _loses_whitespace(tokenizer):
    """The failure this repair targets: the text comes back with some of its whitespace deleted."""
    return _only_whitespace_removed(_spacing_round_trip(tokenizer))


class _TokenizerRefused(ValueError):
    """The loaded tokenizer is wrong and the tokenizer file cannot replace it."""


def _special_layout(tokenizer):
    """The ids a tokenizer adds before and after plain text by default (single-sequence inputs)."""
    text = _plain_ids(tokenizer)
    full = tokenizer(_TOKENIZER_PROBE)["input_ids"]
    for start in range(len(full) - len(text) + 1):
        if full[start : start + len(text)] == text:
            return full[:start], full[start + len(text) :]
    raise ValueError(f"{type(tokenizer).__name__} does not keep the text ids intact when adding special tokens")


def _added_tokens(tokenizer):
    return {
        index: (token.content, token.special, token.single_word, token.lstrip, token.rstrip, token.normalized)
        for index, token in tokenizer.added_tokens_decoder.items()
    }


def _tokenizer_json_path(tokenizer, name_or_path, kwargs):
    """The tokenizer file from_pretrained read: as recorded by the tokenizer, else tokenizer.json in the local
    directory (with its subfolder) or the Hub cache."""
    recorded = getattr(tokenizer, "init_kwargs", {}).get("tokenizer_file")
    if recorded and os.path.isfile(str(recorded)):
        return str(recorded)
    file_kwargs = {key: kwargs[key] for key in _TOKENIZER_FILE_KWARGS if key in kwargs}
    if os.path.isdir(str(name_or_path)):
        local = os.path.join(str(name_or_path), file_kwargs.get("subfolder") or "", "tokenizer.json")
        return local if os.path.isfile(local) else None
    from transformers.utils import cached_file

    return cached_file(
        name_or_path,
        "tokenizer.json",
        _raise_exceptions_for_missing_entries=False,
        _raise_exceptions_for_connection_errors=False,
        **file_kwargs,
    )


def _swap_problem(tokenizer, faithful):
    """Why ``faithful`` cannot replace ``tokenizer``, or None: only the text pipeline may change."""
    for name in (
        "bos_token",
        "eos_token",
        "pad_token",
        "unk_token",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "unk_token_id",
        "chat_template",
    ):
        if getattr(faithful, name, None) != getattr(tokenizer, name, None):
            return f"{name} would change"
    if sorted(faithful.all_special_tokens) != sorted(tokenizer.all_special_tokens):
        return "the special tokens would change"
    if len(faithful) != len(tokenizer) or _added_tokens(faithful) != _added_tokens(tokenizer):
        return "the vocabulary or the added tokens would change"
    if _special_layout(faithful) != _special_layout(tokenizer):
        return "the default special tokens around the text would change"
    return None


def _honor_tokenizer_json(tokenizer, name_or_path, kwargs):
    """Repair a tokenizer that loses whitespace although its tokenizer.json keeps it.

    transformers 5 rebuilds some tokenizer classes (LlamaTokenizer among them) from the class recipe and ignores
    the pre-tokenizer stored in tokenizer.json. For a byte-level BPE vocabulary filed under such a class every
    space and newline is then dropped without any error, and the model is trained and evaluated on run-together
    text. Only that signature is acted on: the loaded tokenizer returns a probe with nothing but its whitespace
    deleted, while the file keeps it and, loaded as stored, encodes the probes like the file.
    Then the file as stored is used, with padding side, truncation side and maximum length carried over, if
    nothing else changes (special and added tokens with their matching rules, vocabulary size, chat template,
    the special tokens around plain text); otherwise loading fails, since training on the loaded tokenizer
    would be wrong. Every other tokenizer, an explicit ``use_fast=False``, and a check that cannot run leave
    the loaded tokenizer as it is (the last with a warning). Text sent to the rollout server as text (not as
    token ids) is tokenized by the server's own tokenizer and is not covered, and versioned tokenizer files
    (``fast_tokenizer_files``) are not looked up."""
    if kwargs.get("use_fast") is False:
        return tokenizer
    path = None
    try:
        if not _loses_whitespace(tokenizer):
            return tokenizer
        path = _tokenizer_json_path(tokenizer, name_or_path, kwargs)
        if path is None:
            warnings.warn(f"{type(tokenizer).__name__} loses whitespace and no tokenizer.json was found", stacklevel=3)
            return tokenizer
        from tokenizers import Tokenizer
        from transformers import PreTrainedTokenizerFast

        reference = Tokenizer.from_file(path)
        # Truncation or padding stored in the file is not applied by from_pretrained calls without those options.
        reference.no_truncation()
        reference.no_padding()
        stored = reference.decode(reference.encode(_SPACING_PROBE, add_special_tokens=False).ids)
        if stored != _SPACING_PROBE:
            return tokenizer  # the file itself does not keep the whitespace: the tokenizer is as stored
        faithful = PreTrainedTokenizerFast.from_pretrained(name_or_path, **kwargs)
        reproduces = all(
            _plain_ids(faithful, text) == reference.encode(text, add_special_tokens=False).ids
            for text in (_TOKENIZER_PROBE, _SPACING_PROBE)
        )
        if not reproduces or _spacing_round_trip(faithful) != _SPACING_PROBE:
            warnings.warn(
                f"{type(tokenizer).__name__} loses whitespace and {path} as stored does not repair it; keeping it",
                stacklevel=3,
            )
            return tokenizer
        problem = _swap_problem(tokenizer, faithful)
    except Exception as error:  # the check is a repair, never a new way to fail
        warnings.warn(
            f"could not check {type(tokenizer).__name__} against {path or name_or_path}, keeping it: "
            f"{type(error).__name__}: {error}",
            stacklevel=3,
        )
        return tokenizer
    if problem is not None:
        raise _TokenizerRefused(
            f"{type(tokenizer).__name__} loses whitespace, and {path} as stored cannot replace it because "
            f"{problem}; fix the tokenizer files (e.g. a tokenizer_class that loads tokenizer.json as stored)"
        )
    for name in ("padding_side", "truncation_side", "model_max_length"):
        setattr(faithful, name, getattr(tokenizer, name))
    warnings.warn(
        f"{type(tokenizer).__name__} loses whitespace; using {path} as stored ({type(faithful).__name__}) instead.",
        stacklevel=3,
    )
    return faithful


def hf_processor(name_or_path, **kwargs):
    """Create a huggingface processor to process multimodal data.

    Args:
        name_or_path (str): The name of the processor.

    Returns:
        Optional[transformers.ProcessorMixin]: The pretrained multimodal processor.
        Returns ``None`` for text-only models (including AutoProcessor fallbacks to
        tokenizer backends such as ``TokenizersBackend``).
    """
    from transformers import AutoConfig, AutoProcessor, PreTrainedTokenizerBase

    try:
        processor = AutoProcessor.from_pretrained(name_or_path, **kwargs)
        # In newer transformers, AutoProcessor may legitimately fall back to a
        # tokenizer backend (e.g. TokenizersBackend) for text-only models.
        # Treat it as "no multimodal processor" and let callers use hf_tokenizer.
        if isinstance(processor, PreTrainedTokenizerBase):
            return None
        if isinstance(getattr(processor, "tokenizer", None), PreTrainedTokenizerBase):
            processor.tokenizer = _honor_tokenizer_json(processor.tokenizer, name_or_path, kwargs)

        config = AutoConfig.from_pretrained(name_or_path, **kwargs)

        # Bind vlm model's get_rope_index method to processor.
        processor.config = config
        model_class = None
        match processor.__class__.__name__:
            case "Qwen2VLProcessor":
                from transformers.models.qwen2_vl import Qwen2VLModel

                model_class = Qwen2VLModel
            case "Qwen2_5_VLProcessor":
                from transformers.models.qwen2_5_vl import Qwen2_5_VLModel

                model_class = Qwen2_5_VLModel
            case "Qwen3VLProcessor":
                from transformers.models.qwen3_vl import Qwen3VLModel

                model_class = Qwen3VLModel
            case "Glm4vProcessor":
                from transformers.models.glm4v import Glm4vModel

                model_class = Glm4vModel
            case "Glm46VProcessor":
                from transformers.models.glm46v import Glm46VModel

                model_class = Glm46VModel
            case "MllamaProcessor":
                pass  # MllamaProcessor and MllamaModel doesn't have get_rope_index property
            case "Gemma4Processor":
                # Gemma4 uses standard 1D RoPE -> no get_rope_index to bind. Disable its strict
                # per-image-token check (which Qwen's processor lacks).
                processor.validate_inputs = lambda *args, **kwargs: None
            case _:
                raise ValueError(f"Unsupported processor type: {processor.__class__.__name__}")

        if model_class is not None:
            processor.get_rope_index = types.MethodType(model_class.get_rope_index, processor)
            if hasattr(model_class, "get_vision_position_ids"):
                processor.get_vision_position_ids = types.MethodType(model_class.get_vision_position_ids, processor)
    except _TokenizerRefused:
        raise
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
