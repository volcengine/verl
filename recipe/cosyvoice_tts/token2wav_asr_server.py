# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
"""Pytriton server for token2wav conversion and ASR"""

import argparse
import io
import logging
from typing import Any, List
import numpy as np
import torch
from scipy.signal import resample
import sys
import random
import re
from jiwer import wer
from pypinyin import lazy_pinyin, Style
from tn.chinese.normalizer import Normalizer as ZhNormalizer

# Chinese text normalizer (cached globally)
zh_tn_model = ZhNormalizer(
    cache_dir="./cache",
    remove_erhua=False,
    remove_interjections=False,
    remove_puncts=True,
    overwrite_cache=True,
)

from pytriton.decorators import batch
from pytriton.model_config import DynamicBatcher, ModelConfig, Tensor
from pytriton.triton import Triton, TritonConfig
from pytriton.proxy.types import Request

from omnisense.models import OmniSenseVoiceSmall
from cosyvoice.cli.cosyvoice import CosyVoice2

from datasets import load_dataset

sys.path.append("/workspace/CosyVoice/third_party/Matcha-TTS")

logger = logging.getLogger("token2wav_asr_server")


class _ASR_Server:
    """Wraps a single OmniSenseVoiceSmall model instance for Triton."""

    def __init__(self, device_id: int):
        self._model = OmniSenseVoiceSmall("iic/SenseVoiceSmall", quantize=False, device_id=device_id)

    @batch
    def __call__(self, WAV: np.ndarray, WAV_LENS: np.ndarray, LANGUAGE: np.ndarray, TEXT_NORM: np.ndarray):
        """
        WAV: np.ndarray, WAV_LENS: np.ndarray
        LANGUAGE: np.ndarray, TEXTNORM: np.ndarray for backward compatibility, not used
        See: https://github.com/modelscope/FunASR/tree/main/runtime/triton_gpu
        """
        logger.debug("WAV: %s, WAV_LENS: %s, shapes: %s %s", type(WAV), type(WAV_LENS), WAV.shape, WAV_LENS.shape)
        wavs = [WAV[i, :WAV_LENS[i, 0]] for i in range(len(WAV))]

        results = self._model.transcribe_single_batch(
            wavs,
            language="zh",
            textnorm="woitn",
        )
        texts = [result.text for result in results]
        transcripts = np.char.encode(np.array(texts).reshape(-1, 1), "utf-8")
        return {"TRANSCRIPTS": transcripts}



def audio_decode_cosyvoice2(
    audio_tokens, prompt_text, prompt_speech_16k, codec_decoder
):
    """
    Generate audio from tokens with optional tone and prompt embedding.
    """
    model_inputs_dict = codec_decoder.frontend.frontend_zero_shot(
        "empty", prompt_text, prompt_speech_16k, 24000
    )
    tts_mel, _ = codec_decoder.model.flow.inference(
        token=audio_tokens.to(codec_decoder.model.device),
        token_len=torch.tensor([audio_tokens.shape[1]], dtype=torch.int32).to(
            codec_decoder.model.device
        ),
        prompt_token=model_inputs_dict["flow_prompt_speech_token"].to(
            codec_decoder.model.device
        ),
        prompt_token_len=torch.tensor(
            [model_inputs_dict["flow_prompt_speech_token_len"]], dtype=torch.int32
        ).to(codec_decoder.model.device),
        prompt_feat=model_inputs_dict["prompt_speech_feat"].to(
            codec_decoder.model.device
        ),
        prompt_feat_len=model_inputs_dict["prompt_speech_feat_len"].to(
            codec_decoder.model.device
        ),
        embedding=model_inputs_dict["flow_embedding"].to(codec_decoder.model.device),
        finalize=True,
    )

    audio_hat, _ = codec_decoder.model.hift.inference(
        speech_feat=tts_mel, cache_source=torch.zeros(1, 1, 0)
    )

    return audio_hat


def get_random_prompt_from_dataset(dataset):
    """
    Get random prompt text and speech from the pre-loaded dataset.
    Returns (prompt_text, prompt_speech_16k)
    """
    random_idx = random.randint(0, len(dataset) - 1)
    sample = dataset[random_idx]
    
    # Extract audio data
    audio_data = sample["audio"]
    audio_array = audio_data["array"]
    sample_rate = audio_data["sampling_rate"]
    
    # Convert audio to 16kHz if needed
    if sample_rate != 16000:
        num_samples = int(len(audio_array) * (16000 / sample_rate))
        audio_array = resample(audio_array, num_samples)

    # Convert to torch tensor
    prompt_speech_16k = torch.from_numpy(audio_array).float().unsqueeze(0)
    prompt_text = sample["text"]
    # remove space in prompt_text
    prompt_text = prompt_text.replace(" ", "")
    return prompt_text, prompt_speech_16k

class _Token2Wav_ASR:
    """Wraps a single OmniSenseVoiceSmall model instance for Triton."""

    def __init__(self, device_id: int):
        self.asr_model = OmniSenseVoiceSmall("iic/SenseVoiceSmall", quantize=False, device_id=device_id)
        self.dataset = load_dataset("yuekai/aishell", "test", trust_remote_code=True)["test"]

        # Make sure the CosyVoice2 decoder lives on the same GPU as the ASR model
        # CosyVoice2 internally uses generic "cuda" device, so we first switch the
        # current CUDA context to the desired card before the object is created.
        # Afterwards, all parameters loaded with the generic "cuda" device will
        # reside on this GPU.  We keep the selected id in `self.device_id` and
        # will set the context again for every forward call to avoid race
        # conditions when several instances are used in the same process.

        self.device_id = device_id

        # Construct the TTS codec decoder under the correct CUDA device context
        with torch.cuda.device(self.device_id):
            self.codec_decoder = CosyVoice2(
                "/workspace/CosyVoice2-0.5B", load_jit=True, load_trt=True, fp16=True
            )
    @batch
    def __call__(self, TOKENS: np.ndarray, TOKEN_LENS: np.ndarray, GT_TEXT: np.ndarray):
        """
        WAV: np.ndarray, WAV_LENS: np.ndarray
        LANGUAGE: np.ndarray, TEXTNORM: np.ndarray for backward compatibility, not used
        See: https://github.com/modelscope/FunASR/tree/main/runtime/triton_gpu
        """
        # Ensure the default CUDA device is set correctly for this invocation
        torch.cuda.set_device(self.device_id)

        if self.device_id == 0:
            print(f"device_id: {self.device_id}, TOKENS: {TOKENS.shape}, TOKEN_LENS: {TOKEN_LENS.shape}")

        tokens_list = [TOKENS[i, :TOKEN_LENS[i, 0]] for i in range(len(TOKENS))]

        # Decode ground-truth text strings (BYTES → str)
        if GT_TEXT.ndim == 2:
            gt_texts = [GT_TEXT[i, 0].decode("utf-8") for i in range(len(GT_TEXT))]
        else:
            gt_texts = [GT_TEXT[i].decode("utf-8") for i in range(len(GT_TEXT))]

        wavs = []
        for tokens in tokens_list:
            prompt_text, prompt_speech_16k = get_random_prompt_from_dataset(self.dataset)
            audio_tokens = torch.tensor(tokens, dtype=torch.long, device=self.asr_model.device).unsqueeze(0)
            audio_hat = audio_decode_cosyvoice2(
                audio_tokens,
                prompt_text,
                prompt_speech_16k,
                self.codec_decoder,
            )
            # resample to 16000 using soundfile
            audio_hat = audio_hat.squeeze(0).float().cpu()
            audio_hat = audio_hat.numpy()
            num_samples = int(len(audio_hat) * (16000 / 24000))
            audio_hat = resample(audio_hat, num_samples)
            wavs.append(audio_hat)

        results = self.asr_model.transcribe_single_batch(
            wavs,
            language="zh",
            textnorm="woitn",
        )
        texts = [result.text for result in results]

        # ---------------- Reward computation ----------------
        rewards = []
        for gt_text, hyp_text in zip(gt_texts, texts):
            gt_norm = zh_tn_model.normalize(gt_text).lower()
            hyp_norm = zh_tn_model.normalize(hyp_text).lower()

            gt_pinyin = lazy_pinyin(
                gt_norm,
                style=Style.TONE3,
                tone_sandhi=True,
                neutral_tone_with_five=True,
            )
            hyp_pinyin = lazy_pinyin(
                hyp_norm,
                style=Style.TONE3,
                tone_sandhi=True,
                neutral_tone_with_five=True,
            )

            c = float(wer(" ".join(gt_pinyin), " ".join(hyp_pinyin)))
            reward_val = 1.0 - np.tanh(3.0 * c)
            reward_val = max(0.0, min(1.0, reward_val))
            rewards.append(reward_val)
            print(f"gt_text: {gt_text}, hyp_text: {hyp_text}, reward_val: {reward_val}")

        transcripts = np.char.encode(np.array(texts).reshape(-1, 1), "utf-8")
        rewards_arr = np.array(rewards, dtype=np.float32).reshape(-1, 1)


        return {"REWARDS": rewards_arr, "TRANSCRIPTS": transcripts}


def _infer_function_factory(device_ids: List[int], model_name: str):
    """Creates a list of inference functions, one for each requested device ID."""
    infer_funcs = []
    for device_id in device_ids:
        if model_name == "sensevoice":
            infer_funcs.append(_ASR_Server(device_id=device_id))
        else:
            infer_funcs.append(_Token2Wav_ASR(device_id=device_id))
    return infer_funcs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=32,
        help="Batch size of request.",
        required=False,
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--number-of-instances-per-device",
        type=int,
        default=1,
        help="Number of model instances to load.",
        required=False,
    )
    parser.add_argument(
        "--number-of-devices",
        type=int,
        default=8,
        help="Number of devices to use.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="token2wav_asr",
        choices=["token2wav_asr", "sensevoice"],
        help="Model name.",
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")

    triton_config = TritonConfig(
        http_port=8000,
        grpc_port=8001,
        metrics_port=8002,
    )

    device_ids = [i for i in range(args.number_of_devices)]
    device_ids = device_ids * args.number_of_instances_per_device

    with Triton(config=triton_config) as triton:
        logger.info("Loading SenseVoice model on device ids: %s", device_ids)
        if args.model_name == "sensevoice":
            triton.bind(
                model_name="sensevoice",
                infer_func=_infer_function_factory(device_ids, args.model_name),
                inputs=[
                    Tensor(name="WAV", dtype=np.float32, shape=(-1,)),
                    Tensor(name="WAV_LENS", dtype=np.int32, shape=(-1,)),
                    Tensor(name="LANGUAGE", dtype=np.int32, shape=(-1,)),
                    Tensor(name="TEXT_NORM", dtype=np.int32, shape=(-1,)),
                ],
                outputs=[
                    Tensor(name="TRANSCRIPTS", dtype=bytes, shape=(-1,)),
                ],
                config=ModelConfig(
                    max_batch_size=args.max_batch_size,
                    batcher=DynamicBatcher(max_queue_delay_microseconds=10000),  # 10ms
                ),
                strict=True,
            )
        else:
            triton.bind(
                model_name="token2wav_asr",
                infer_func=_infer_function_factory(device_ids, args.model_name),
                inputs=[
                    Tensor(name="TOKENS", dtype=np.int32, shape=(-1,)),
                    Tensor(name="TOKEN_LENS", dtype=np.int32, shape=(-1,)),
                    Tensor(name="GT_TEXT", dtype=bytes, shape=(-1,)),
                ],
                outputs=[
                    Tensor(name="REWARDS", dtype=np.float32, shape=(-1,)),
                    Tensor(name="TRANSCRIPTS", dtype=bytes, shape=(-1,)),
                ],
                config=ModelConfig(
                    max_batch_size=args.max_batch_size,
                    batcher=DynamicBatcher(max_queue_delay_microseconds=10000),  # 10ms
                ),
                strict=True,
            )
        logger.info("Serving inference")
        triton.serve()


if __name__ == "__main__":
    main()
