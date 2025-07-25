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
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import soundfile as sf
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from argparse import ArgumentParser
import sys

sys.path.append("/workspace/CosyVoice/third_party/Matcha-TTS")
TEMPLATE = "{% for message in messages %}{%- if message['role'] == 'user' %}{{- '<|im_start|>' + message['role'] + '\n' + 'Convert the text to speech: ' + message['content'] + '<|im_end|>\n'}}{%- elif message['role'] == 'assistant' %}{{- '<|im_start|>' + message['role'] + '\n' + '<|SPEECH_GENERATION_START|>' + message['content']}}{%- endif %}{%- endfor %}"

def get_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--token2wav-path",
        type=str,
        default=None,
        help="Token2Wav path, default to %(default)r",
    )
    parser.add_argument(
        "--prompt-text",
        type=str,
        default="Romeo and Juliet might be the most famous act of William Shakespeare.",
        help="The prompt text",
    )

    parser.add_argument(
        "--prompt-speech-path",
        type=str,
        default="./assets/common_voice_en_2586258.wav",
        help="The path to the prompt speech",
    )
    parser.add_argument(
        "--input-text",
        type=str,
        default='突然，身边一阵笑声。我看着他们，意气风发地挺直了胸膛，甩了甩那稍显肉感的双臂，轻笑道：我身上的肉，是为了掩饰我爆棚的魅力，否则，岂不吓坏了你们呢？"',
        help="The input text",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default='/workspace/rl/llasa_cosyvoice2_token_qwen_0.5b/checkpoint-885000',
        help="The path to the model",
    )
    args = parser.parse_args()
    return args

args = get_args()

def audio_decode_cosyvoice2(
    audio_tokens, prompt_text, prompt_speech_16k, codec_decoder
):
    """
    Generate audio from tokens with optional tone and prompt embedding.

    Args:
        audio_tokens (list): List of audio tokens to be processed.
        model_config: Configuration object containing vocab settings.
        codec_decoder: Codec decoder for generating audio.
        tone_dir (str): The tone directory or setting.
        audio_prompt_path (str, optional): Path to the audio prompt file. Required when tone_dir is not "default_tone".
        code_layer (int, optional): Number of code layers. Defaults to 1.
        num_latency_tokens (int, optional): Number of latency tokens to ignore. Defaults to 0.
        speed (float, optional): Speed factor for audio generation. Defaults to 1.0.

    Returns:
        torch.Tensor: Generated audio waveform.
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
 
    speech_ids = []
    for token_str in speech_tokens_str:
        if token_str.startswith('<|s_') and token_str.endswith('|>'):
            num_str = token_str[4:-2]

            num = int(num_str)
            speech_ids.append(num)
        else:
            print(f"Unexpected token: {token_str}")
    return speech_ids



tokenizer = AutoTokenizer.from_pretrained(args.model_path)
model = AutoModelForCausalLM.from_pretrained(args.model_path)
model.eval() 
model.to('cuda')

token2wav_model = CosyVoice2(
    args.token2wav_path, load_jit=False, load_trt=False, fp16=False
)

prompt_speech_16k = load_wav(args.prompt_speech_path, 16000)

with torch.no_grad():
    # Tokenize the text
    chat = [
        {"role": "user", "content": f"{args.input_text}"},
        {"role": "assistant", "content": ""}
    ]
    if 'system' in tokenizer.chat_template:
        tokenizer.chat_template = TEMPLATE
    input_ids = tokenizer.apply_chat_template(
        chat, 
        tokenize=True, 
        return_tensors='pt', 
        continue_final_message=True
    )
    input_ids = input_ids.to('cuda')

    # Generate the speech autoregressively
    outputs = model.generate(
        input_ids,
        max_length=2048,  # We trained our model with a max length of 2048
        do_sample=True,    
        top_p=1,           #  Adjusts the diversity of generated content
        temperature=0.8,   #  Controls randomness in output
    )
    # Extract the speech tokens
    generated_ids = outputs[0][input_ids.shape[1]:-1]

    speech_tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)  

    # Convert  token <|s_23456|> to int 23456 
    speech_tokens = extract_speech_ids(speech_tokens)

    speech_tokens = torch.tensor(speech_tokens).cuda().unsqueeze(0)

    
    audio_hat = audio_decode_cosyvoice2(
        speech_tokens,
        args.prompt_text,
        prompt_speech_16k,
        token2wav_model,
    )

    audio = audio_hat.squeeze(0).cpu().numpy()


sf.write("gen.wav", audio, 24000)
