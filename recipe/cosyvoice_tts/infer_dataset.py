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
""" Example Usage
dataset=zero_shot_zh
output_dir=./outputs_rl_aishell3_step${step}_${dataset}_jit_trt_fp16_reward_tts

token2wav_path=/workspace/CosyVoice2-0.5B
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --nproc_per_node=8 \
    infer_dataset.py \
    --output-dir $output_dir \
    --llm-model-name-or-path $llm_path/merged_hf_model \
    --token2wav-path $token2wav_path \
    --split-name ${dataset} || exit 1
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchaudio
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
import soundfile as sf
import s3tokenizer
from functools import partial

sys.path.append("/workspace/CosyVoice/third_party/Matcha-TTS")
try:
    torch.multiprocessing.set_start_method("spawn")
except RuntimeError:
    pass


TEMPLATE = "{% for message in messages %}{%- if message['role'] == 'user' %}{{- '<|im_start|>' + message['role'] + '\n' + 'Convert the text to speech: ' + message['content'] + '<|im_end|>\n'}}{%- elif message['role'] == 'assistant' %}{{- '<|im_start|>' + message['role'] + '\n' + '<|SPEECH_GENERATION_START|>' + message['content']}}{%- endif %}{%- endfor %}"


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


def extract_speech_ids(speech_tokens_str):
    """Extract speech IDs from token strings like <|s_23456|>"""
    speech_ids = []
    for token_str in speech_tokens_str:
        if token_str.startswith('<|s_') and token_str.endswith('|>'):
            num_str = token_str[4:-2]
            num = int(num_str)
            speech_ids.append(num)
        else:
            print(f"Unexpected token: {token_str}")
    return speech_ids

def convert_cosy2_tokens_to_speech_id_str(cosy2_tokens):
    """Convert CosyVoice2 tokens to speech IDs string like <|s_23456|>"""
    speech_id_str = ""
    for token in cosy2_tokens:
        speech_id_str += f"<|s_{token}|>"
    return speech_id_str


def get_args():
    parser = argparse.ArgumentParser(description="Speech generation using LLM + CosyVoice2")
    parser.add_argument(
        "--split-name",
        type=str,
        default="wenetspeech4tts",
        help="huggingface dataset split name, see yuekai/CV3-Eval, yuekai/seed_tts_cosy2",
    )
    parser.add_argument(
        "--output-dir", required=True, type=str, help="dir to save result"
    )
    parser.add_argument(
        "--batch-size",
        default=1,
        type=int,
        help="batch size (per-device) for inference",
    )
    parser.add_argument(
        "--num-workers", type=int, default=1, help="workers for dataloader"
    )
    parser.add_argument(
        "--prefetch", type=int, default=5, help="prefetch for dataloader"
    )
    parser.add_argument(
        "--llm-model-name-or-path",
        required=True,
        type=str,
        help="LLM model path (includes both model and tokenizer)",
    )
    parser.add_argument(
        "--token2wav-path",
        required=True,
        type=str,
        help="CosyVoice2 token2wav model path",
    )
    parser.add_argument(
        "--prompt-text",
        type=str,
        default=None,
        help="The prompt text for CosyVoice2",
    )
    parser.add_argument(
        "--prompt-speech-path",
        type=str,
        default=None,
        help="The path to the prompt speech for CosyVoice2",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="top p for sampling",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="temperature for sampling",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="top k for sampling",
    )
    args = parser.parse_args()
    return args



def data_collator(batch, tokenizer, s3_tokenizer):
    """Simplified data collator for batch_size=1 processing"""
    target_sample_rate = 16000  # CosyVoice2 uses 16kHz for prompt audio
    device = s3_tokenizer.device if s3_tokenizer is not None else torch.device("cpu")
    input_ids_list, prompt_audio_list, prompt_text_list = [], [], []
    mels, prompt_audio_cosy2tokens_list = [], []
    for i, item in enumerate(batch):
        prompt_text, target_text = (
            item["prompt_text"],
            item["target_text"],
        )
        prompt_text_list.append(prompt_text)
        # Combine prompt and target text
        full_text = prompt_text + target_text

        # get prompt audio for CosyVoice2 (convert to 16kHz)
        ref_audio_org, ref_sr = (
            item["prompt_audio"]["array"],
            item["prompt_audio"]["sampling_rate"],
        )
        ref_audio_org = torch.from_numpy(ref_audio_org).float().unsqueeze(0)
        # ref_audio_org = ref_audio_org.mean(dim=0, keepdim=True)
        print(ref_audio_org.shape)

        if ref_sr != target_sample_rate:
            resampler = torchaudio.transforms.Resample(ref_sr, target_sample_rate)
            ref_audio = resampler(ref_audio_org)
        else:
            ref_audio = ref_audio_org

        prompt_audio_list.append(ref_audio)

        if "prompt_audio_cosy2_tokens" in item:
            prompt_audio_cosy2tokens = item["prompt_audio_cosy2_tokens"]
            prompt_audio_cosy2tokens_list.append(prompt_audio_cosy2tokens)
        else:
            # convert to float first
            mels.append(s3tokenizer.log_mel_spectrogram(ref_audio.squeeze(0)))

    if len(mels) > 0:
        mels, mels_lens = s3tokenizer.padding(mels)
        codes, codes_lens = s3_tokenizer.quantize(mels.to(device), mels_lens.to(device))
        for i in range(len(codes)):
            prompt_audio_cosy2tokens_list.append(codes[i, :codes_lens[i].item()])
    for i, prompt_audio_cosy2tokens in enumerate(prompt_audio_cosy2tokens_list):
        prompt_audio_cosy2_id_str = convert_cosy2_tokens_to_speech_id_str(prompt_audio_cosy2tokens)
        # Create chat template for LLM generation
        chat = [
            {"role": "user", "content": full_text},
            {"role": "assistant", "content": prompt_audio_cosy2_id_str}
        ]
        if 'system' in tokenizer.chat_template:
            tokenizer.chat_template = TEMPLATE
        input_ids = tokenizer.apply_chat_template(
            chat,
            tokenize=True,
            return_tensors='pt',
            continue_final_message=True
        )
        input_ids_list.append(input_ids.squeeze(0))


    # For batch_size=1, no need to pad
    if len(input_ids_list) == 1:
        input_ids = input_ids_list[0].unsqueeze(0)
    else:
        # Handle batch > 1 if needed
        max_len = max([len(input_ids) for input_ids in input_ids_list])
        input_ids_list = [
            torch.cat([torch.full((max_len - len(input_ids),), tokenizer.pad_token_id), input_ids])
            for input_ids in input_ids_list
        ]
        input_ids = torch.stack(input_ids_list)
    
    ids = [item["id"] for item in batch]

    return {
        "input_ids": input_ids,
        "ids": ids,
        "prompt_text": prompt_text_list,
        "prompt_audio_list": prompt_audio_list,
    }


def init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    print(
        "Inference on multiple gpus, this gpu {}".format(local_rank)
        + ", rank {}, world_size {}".format(rank, world_size)
    )
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    return world_size, local_rank, rank


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    assert torch.cuda.is_available()
    world_size, local_rank, rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}")
    
    # Load LLM model and tokenizer directly
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(args.llm_model_name_or_path)
    model.eval()
    model.to(device)

    cosyvoice_codec = CosyVoice2(
        args.token2wav_path, load_jit=True, load_trt=True, fp16=True
    )
    if args.prompt_speech_path:
        prompt_speech_16k = load_wav(args.prompt_speech_path, 16000)
    else:
        prompt_speech_16k = None
    s3_tokenizer = s3tokenizer.load_model("speech_tokenizer_v2_25hz").to(device) if 'zero' in args.split_name else None
    dataset_name = "yuekai/CV3-Eval" if 'zero' in args.split_name else "yuekai/seed_tts_cosy2"
    dataset = load_dataset(
        dataset_name,
        split=args.split_name,
        trust_remote_code=True,
    )

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch,
        collate_fn=partial(data_collator, tokenizer=tokenizer, s3_tokenizer=s3_tokenizer),
    )

    total_steps = len(dataset)

    if rank == 0:
        progress_bar = tqdm(total=total_steps, desc="Processing", unit="wavs")

    for batch in dataloader:
        with torch.no_grad():
            input_ids = batch["input_ids"].to(device)
            
            # Generate speech tokens using LLM
            outputs = model.generate(
                input_ids,
                max_new_tokens=2048,  # Max length for generation
                do_sample=True,
                top_p=args.top_p,
                temperature=args.temperature,
                top_k=args.top_k,
            )
            
            # Process each sample in the batch
            for i in range(len(batch["ids"])):
                # Extract generated tokens (excluding input)
                input_length = input_ids[i].shape[0]
                generated_ids = outputs[i][input_length:-1]  # Remove last token if needed
                speech_tokens_str = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                
                # Extract speech IDs from token strings like <|s_23456|>
                speech_ids = extract_speech_ids(speech_tokens_str)
                
                if len(speech_ids) == 0:
                    print(f"Warning: No speech tokens generated for sample {batch['ids'][i]}, skipping")
                    continue
                
                # Convert to tensor for CosyVoice2
                audio_tokens = torch.tensor(speech_ids, dtype=torch.long, device=device).unsqueeze(0)
                
                if args.prompt_text is not None:
                    current_prompt_text = args.prompt_text
                    current_prompt_audio = prompt_speech_16k
                else:
                    current_prompt_text = batch["prompt_text"][i]
                    current_prompt_audio = batch["prompt_audio_list"][i]
                
                if current_prompt_audio is not None:
                    # Generate audio using CosyVoice2
                    audio_hat = audio_decode_cosyvoice2(
                        audio_tokens,
                        current_prompt_text,
                        current_prompt_audio,
                        cosyvoice_codec,
                    )
                    
                    # Convert to numpy and save
                    generated_wave = audio_hat.squeeze(0).cpu().numpy()
                    target_sample_rate = 24000
                    
                    utt = batch["ids"][i]
                    sf.write(f"{args.output_dir}/{utt}.wav", generated_wave, target_sample_rate)

                    print(f"Generated audio for sample {utt} with {len(speech_ids)} tokens")
                else:
                    print(f"Warning: No prompt audio available for sample {batch['ids'][i]}, skipping")
                        

        if rank == 0:
            progress_bar.update(world_size * len(batch["ids"]))

    if rank == 0:
        progress_bar.close()

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
