#!/usr/bin/env python3
#
# Copyright (c)  2023 by manyeyes
# Copyright (c)  2023  Xiaomi Corporation

"""
This file demonstrates how to use sherpa-onnx Python API to transcribe
file(s) with a non-streaming model.

(1) For paraformer

    ./python-api-examples/offline-decode-files.py  \
      --tokens=/path/to/tokens.txt \
      --paraformer=/path/to/paraformer.onnx \
      --num-threads=2 \
      --decoding-method=greedy_search \
      --debug=false \
      --sample-rate=16000 \
      --feature-dim=80 \
      /path/to/0.wav \
      /path/to/1.wav

(2) For transducer models from icefall

    ./python-api-examples/offline-decode-files.py  \
      --tokens=/path/to/tokens.txt \
      --encoder=/path/to/encoder.onnx \
      --decoder=/path/to/decoder.onnx \
      --joiner=/path/to/joiner.onnx \
      --num-threads=2 \
      --decoding-method=greedy_search \
      --debug=false \
      --sample-rate=16000 \
      --feature-dim=80 \
      /path/to/0.wav \
      /path/to/1.wav

(3) For CTC models from NeMo

python3 ./python-api-examples/offline-decode-files.py \
  --tokens=./sherpa-onnx-nemo-ctc-en-citrinet-512/tokens.txt \
  --nemo-ctc=./sherpa-onnx-nemo-ctc-en-citrinet-512/model.onnx \
  --num-threads=2 \
  --decoding-method=greedy_search \
  --debug=false \
  ./sherpa-onnx-nemo-ctc-en-citrinet-512/test_wavs/0.wav \
  ./sherpa-onnx-nemo-ctc-en-citrinet-512/test_wavs/1.wav \
  ./sherpa-onnx-nemo-ctc-en-citrinet-512/test_wavs/8k.wav

(4) For Whisper models

python3 ./python-api-examples/offline-decode-files.py \
  --whisper-encoder=./sherpa-onnx-whisper-base.en/base.en-encoder.int8.onnx \
  --whisper-decoder=./sherpa-onnx-whisper-base.en/base.en-decoder.int8.onnx \
  --tokens=./sherpa-onnx-whisper-base.en/base.en-tokens.txt \
  --whisper-task=transcribe \
  --num-threads=1 \
  ./sherpa-onnx-whisper-base.en/test_wavs/0.wav \
  ./sherpa-onnx-whisper-base.en/test_wavs/1.wav \
  ./sherpa-onnx-whisper-base.en/test_wavs/8k.wav

(5) For CTC models from WeNet

python3 ./python-api-examples/offline-decode-files.py \
  --wenet-ctc=./sherpa-onnx-zh-wenet-wenetspeech/model.onnx \
  --tokens=./sherpa-onnx-zh-wenet-wenetspeech/tokens.txt \
  ./sherpa-onnx-zh-wenet-wenetspeech/test_wavs/0.wav \
  ./sherpa-onnx-zh-wenet-wenetspeech/test_wavs/1.wav \
  ./sherpa-onnx-zh-wenet-wenetspeech/test_wavs/8k.wav

(6) For tdnn models of the yesno recipe from icefall

python3 ./python-api-examples/offline-decode-files.py \
  --sample-rate=8000 \
  --feature-dim=23 \
  --tdnn-model=./sherpa-onnx-tdnn-yesno/model-epoch-14-avg-2.onnx \
  --tokens=./sherpa-onnx-tdnn-yesno/tokens.txt \
  ./sherpa-onnx-tdnn-yesno/test_wavs/0_0_0_1_0_0_0_1.wav \
  ./sherpa-onnx-tdnn-yesno/test_wavs/0_0_1_0_0_0_1_0.wav \
  ./sherpa-onnx-tdnn-yesno/test_wavs/0_0_1_0_0_1_1_1.wav

Please refer to
https://k2-fsa.github.io/sherpa/onnx/index.html
to install sherpa-onnx and to download non-streaming pre-trained models
used in this file.
"""
import argparse
import time
import wave
from pathlib import Path
from typing import List, Tuple, Dict, Iterable, TextIO, Union

import numpy as np
import sherpa_onnx
import soundfile as sf
from datasets import load_dataset
import logging
from collections import defaultdict
import kaldialign
from zhon.hanzi import punctuation
import string
punctuation_all = punctuation + string.punctuation
Pathlike = Union[str, Path]

def remove_punctuation(text: str) -> str:
    for x in punctuation_all:
        if x == '\'':
            continue
        text = text.replace(x, '')
    return text

def store_transcripts(
    filename: Pathlike, texts: Iterable[Tuple[str, str, str]], char_level: bool = False
) -> None:
    """Save predicted results and reference transcripts to a file.

    Args:
      filename:
        File to save the results to.
      texts:
        An iterable of tuples. The first element is the cur_id, the second is
        the reference transcript and the third element is the predicted result.
        If it is a multi-talker ASR system, the ref and hyp may also be lists of
        strings.
    Returns:
      Return None.
    """
    with open(filename, "w", encoding="utf8") as f:
        for cut_id, ref, hyp in texts:
            if char_level:
                ref = list("".join(ref))
                hyp = list("".join(hyp))
            print(f"{cut_id}:\tref={ref}", file=f)
            print(f"{cut_id}:\thyp={hyp}", file=f)


def write_error_stats(
    f: TextIO,
    test_set_name: str,
    results: List[Tuple[str, str]],
    enable_log: bool = True,
    compute_CER: bool = False,
    sclite_mode: bool = False,
) -> float:
    """Write statistics based on predicted results and reference transcripts.

    It will write the following to the given file:

        - WER
        - number of insertions, deletions, substitutions, corrects and total
          reference words. For example::

              Errors: 23 insertions, 57 deletions, 212 substitutions, over 2606
              reference words (2337 correct)

        - The difference between the reference transcript and predicted result.
          An instance is given below::

            THE ASSOCIATION OF (EDISON->ADDISON) ILLUMINATING COMPANIES

          The above example shows that the reference word is `EDISON`,
          but it is predicted to `ADDISON` (a substitution error).

          Another example is::

            FOR THE FIRST DAY (SIR->*) I THINK

          The reference word `SIR` is missing in the predicted
          results (a deletion error).
      results:
        An iterable of tuples. The first element is the cut_id, the second is
        the reference transcript and the third element is the predicted result.
      enable_log:
        If True, also print detailed WER to the console.
        Otherwise, it is written only to the given file.
    Returns:
      Return None.
    """
    subs: Dict[Tuple[str, str], int] = defaultdict(int)
    ins: Dict[str, int] = defaultdict(int)
    dels: Dict[str, int] = defaultdict(int)

    # `words` stores counts per word, as follows:
    #   corr, ref_sub, hyp_sub, ins, dels
    words: Dict[str, List[int]] = defaultdict(lambda: [0, 0, 0, 0, 0])
    num_corr = 0
    ERR = "*"

    if compute_CER:
        for i, res in enumerate(results):
            cut_id, ref, hyp = res
            ref = list("".join(ref))
            hyp = list("".join(hyp))
            results[i] = (cut_id, ref, hyp)

    for cut_id, ref, hyp in results:
        ali = kaldialign.align(ref, hyp, ERR, sclite_mode=sclite_mode)
        for ref_word, hyp_word in ali:
            if ref_word == ERR:
                ins[hyp_word] += 1
                words[hyp_word][3] += 1
            elif hyp_word == ERR:
                dels[ref_word] += 1
                words[ref_word][4] += 1
            elif hyp_word != ref_word:
                subs[(ref_word, hyp_word)] += 1
                words[ref_word][1] += 1
                words[hyp_word][2] += 1
            else:
                words[ref_word][0] += 1
                num_corr += 1
    ref_len = sum([len(r) for _, r, _ in results])
    sub_errs = sum(subs.values())
    ins_errs = sum(ins.values())
    del_errs = sum(dels.values())
    tot_errs = sub_errs + ins_errs + del_errs
    tot_err_rate = "%.2f" % (100.0 * tot_errs / ref_len)

    if enable_log:
        logging.info(
            f"[{test_set_name}] %WER {tot_errs / ref_len:.2%} "
            f"[{tot_errs} / {ref_len}, {ins_errs} ins, "
            f"{del_errs} del, {sub_errs} sub ]"
        )

    print(f"%WER = {tot_err_rate}", file=f)
    print(
        f"Errors: {ins_errs} insertions, {del_errs} deletions, "
        f"{sub_errs} substitutions, over {ref_len} reference "
        f"words ({num_corr} correct)",
        file=f,
    )
    print(
        "Search below for sections starting with PER-UTT DETAILS:, "
        "SUBSTITUTIONS:, DELETIONS:, INSERTIONS:, PER-WORD STATS:",
        file=f,
    )

    print("", file=f)
    print("PER-UTT DETAILS: corr or (ref->hyp)  ", file=f)
    for cut_id, ref, hyp in results:
        ali = kaldialign.align(ref, hyp, ERR)
        combine_successive_errors = True
        if combine_successive_errors:
            ali = [[[x], [y]] for x, y in ali]
            for i in range(len(ali) - 1):
                if ali[i][0] != ali[i][1] and ali[i + 1][0] != ali[i + 1][1]:
                    ali[i + 1][0] = ali[i][0] + ali[i + 1][0]
                    ali[i + 1][1] = ali[i][1] + ali[i + 1][1]
                    ali[i] = [[], []]
            ali = [
                [
                    list(filter(lambda a: a != ERR, x)),
                    list(filter(lambda a: a != ERR, y)),
                ]
                for x, y in ali
            ]
            ali = list(filter(lambda x: x != [[], []], ali))
            ali = [
                [
                    ERR if x == [] else " ".join(x),
                    ERR if y == [] else " ".join(y),
                ]
                for x, y in ali
            ]

        print(
            f"{cut_id}:\t"
            + " ".join(
                (
                    ref_word if ref_word == hyp_word else f"({ref_word}->{hyp_word})"
                    for ref_word, hyp_word in ali
                )
            ),
            file=f,
        )

    print("", file=f)
    print("SUBSTITUTIONS: count ref -> hyp", file=f)

    for count, (ref, hyp) in sorted([(v, k) for k, v in subs.items()], reverse=True):
        print(f"{count}   {ref} -> {hyp}", file=f)

    print("", file=f)
    print("DELETIONS: count ref", file=f)
    for count, ref in sorted([(v, k) for k, v in dels.items()], reverse=True):
        print(f"{count}   {ref}", file=f)

    print("", file=f)
    print("INSERTIONS: count hyp", file=f)
    for count, hyp in sorted([(v, k) for k, v in ins.items()], reverse=True):
        print(f"{count}   {hyp}", file=f)

    print("", file=f)
    print("PER-WORD STATS: word  corr tot_errs count_in_ref count_in_hyp", file=f)
    for _, word, counts in sorted(
        [(sum(v[1:]), k, v) for k, v in words.items()], reverse=True
    ):
        (corr, ref_sub, hyp_sub, ins, dels) = counts
        tot_errs = ref_sub + hyp_sub + ins + dels
        ref_count = corr + ref_sub + dels
        hyp_count = corr + hyp_sub + ins

        print(f"{word}   {corr} {tot_errs} {ref_count} {hyp_count}", file=f)
    return float(tot_err_rate)

def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--tokens",
        type=str,
        help="Path to tokens.txt",
    )

    parser.add_argument(
        "--hotwords-file",
        type=str,
        default="",
        help="""
        The file containing hotwords, one words/phrases per line, like
        HELLO WORLD
        你好世界
        """,
    )

    parser.add_argument(
        "--hotwords-score",
        type=float,
        default=1.5,
        help="""
        The hotword score of each token for biasing word/phrase. Used only if
        --hotwords-file is given.
        """,
    )

    parser.add_argument(
        "--modeling-unit",
        type=str,
        default="",
        help="""
        The modeling unit of the model, valid values are cjkchar, bpe, cjkchar+bpe.
        Used only when hotwords-file is given.
        """,
    )

    parser.add_argument(
        "--bpe-vocab",
        type=str,
        default="",
        help="""
        The path to the bpe vocabulary, the bpe vocabulary is generated by
        sentencepiece, you can also export the bpe vocabulary through a bpe model
        by `scripts/export_bpe_vocab.py`. Used only when hotwords-file is given
        and modeling-unit is bpe or cjkchar+bpe.
        """,
    )

    parser.add_argument(
        "--encoder",
        default="",
        type=str,
        help="Path to the encoder model",
    )

    parser.add_argument(
        "--decoder",
        default="",
        type=str,
        help="Path to the decoder model",
    )

    parser.add_argument(
        "--joiner",
        default="",
        type=str,
        help="Path to the joiner model",
    )

    parser.add_argument(
        "--paraformer",
        default="",
        type=str,
        help="Path to the model.onnx from Paraformer",
    )

    parser.add_argument(
        "--nemo-ctc",
        default="",
        type=str,
        help="Path to the model.onnx from NeMo CTC",
    )

    parser.add_argument(
        "--wenet-ctc",
        default="",
        type=str,
        help="Path to the model.onnx from WeNet CTC",
    )

    parser.add_argument(
        "--tdnn-model",
        default="",
        type=str,
        help="Path to the model.onnx for the tdnn model of the yesno recipe",
    )

    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Number of threads for neural network computation",
    )

    parser.add_argument(
        "--whisper-encoder",
        default="",
        type=str,
        help="Path to whisper encoder model",
    )

    parser.add_argument(
        "--whisper-decoder",
        default="",
        type=str,
        help="Path to whisper decoder model",
    )

    parser.add_argument(
        "--whisper-language",
        default="",
        type=str,
        help="""It specifies the spoken language in the input audio file.
        Example values: en, fr, de, zh, jp.
        Available languages for multilingual models can be found at
        https://github.com/openai/whisper/blob/main/whisper/tokenizer.py#L10
        If not specified, we infer the language from the input audio file.
        """,
    )

    parser.add_argument(
        "--whisper-task",
        default="transcribe",
        choices=["transcribe", "translate"],
        type=str,
        help="""For multilingual models, if you specify translate, the output
        will be in English.
        """,
    )

    parser.add_argument(
        "--whisper-tail-paddings",
        default=-1,
        type=int,
        help="""Number of tail padding frames.
        We have removed the 30-second constraint from whisper, so you need to
        choose the amount of tail padding frames by yourself.
        Use -1 to use a default value for tail padding.
        """,
    )

    parser.add_argument(
        "--blank-penalty",
        type=float,
        default=0.0,
        help="""
        The penalty applied on blank symbol during decoding.
        Note: It is a positive value that would be applied to logits like
        this `logits[:, 0] -= blank_penalty` (suppose logits.shape is
        [batch_size, vocab] and blank id is 0).
        """,
    )

    parser.add_argument(
        "--decoding-method",
        type=str,
        default="greedy_search",
        help="Valid values are greedy_search and modified_beam_search",
    )
    parser.add_argument(
        "--debug",
        type=bool,
        default=False,
        help="True to show debug messages",
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="""Sample rate of the feature extractor. Must match the one
        expected  by the model. Note: The input sound files can have a
        different sample rate from this argument.""",
    )

    parser.add_argument(
        "--feature-dim",
        type=int,
        default=80,
        help="Feature dimension. Must match the one expected by the model",
    )

    parser.add_argument(
        "sound_files",
        type=str,
        nargs="+",
        help="The input sound file(s) to decode. Each file must be of WAVE"
        "format with a single channel, and each sample has 16-bit, "
        "i.e., int16_t. "
        "The sample rate of the file can be arbitrary and does not need to "
        "be 16 kHz",
    )

    parser.add_argument(
        "--name",
        type=str,
        default="",
        help="The directory containing the input sound files to decode",
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        default="",
        help="The directory containing the input sound files to decode",
    )

    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="wav_base_name label",
    )
    
    # Dataset related arguments for loading labels when label file is not provided
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="yuekai/seed_tts_cosy2",
        help="Huggingface dataset name for loading labels",
    )
    
    parser.add_argument(
        "--split-name",
        type=str,
        default="wenetspeech4tts",
        help="Dataset split name for loading labels",
    )
    
    return parser.parse_args()


def assert_file_exists(filename: str):
    assert Path(filename).is_file(), (
        f"{filename} does not exist!\n"
        "Please refer to "
        "https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html to download it"
    )


def read_wave(wave_filename: str) -> Tuple[np.ndarray, int]:
    """
    Args:
      wave_filename:
        Path to a wave file. It should be single channel and can be of type
        32-bit floating point PCM. Its sample rate does not need to be 24kHz.

    Returns:
      Return a tuple containing:
       - A 1-D array of dtype np.float32 containing the samples,
         which are normalized to the range [-1, 1].
       - Sample rate of the wave file.
    """

    samples, sample_rate = sf.read(wave_filename, dtype="float32")
    assert (
        samples.ndim == 1
    ), f"Expected single channel, but got {samples.ndim} channels."

    samples_float32 = samples.astype(np.float32)

    return samples_float32, sample_rate


def normalize_text_alimeeting(text: str) -> str:
    """
    Text normalization similar to M2MeT challenge baseline.
    See: https://github.com/yufan-aslp/AliMeeting/blob/main/asr/local/text_normalize.pl
    """
    import re
    text = text.replace('\u00A0', '') # test_hard
    text = text.replace(" ", "")
    text = text.replace("<sil>", "")
    text = text.replace("<%>", "")
    text = text.replace("<->", "")
    text = text.replace("<$>", "")
    text = text.replace("<#>", "")
    text = text.replace("<_>", "")
    text = text.replace("<space>", "")
    text = text.replace("`", "")
    text = text.replace("&", "")
    text = text.replace(",", "")
    if re.search("[a-zA-Z]", text):
        text = text.upper()
    text = text.replace("Ａ", "A")
    text = text.replace("ａ", "A")
    text = text.replace("ｂ", "B")
    text = text.replace("ｃ", "C")
    text = text.replace("ｋ", "K")
    text = text.replace("ｔ", "T")
    text = text.replace("，", "")
    text = text.replace("丶", "")
    text = text.replace("。", "")
    text = text.replace("、", "")
    text = text.replace("？", "")
    text = remove_punctuation(text)
    return text


def main():
    args = get_args()
    assert_file_exists(args.tokens)
    assert args.num_threads > 0, args.num_threads

    assert len(args.nemo_ctc) == 0, args.nemo_ctc
    assert len(args.wenet_ctc) == 0, args.wenet_ctc
    assert len(args.whisper_encoder) == 0, args.whisper_encoder
    assert len(args.whisper_decoder) == 0, args.whisper_decoder
    assert len(args.tdnn_model) == 0, args.tdnn_model

    assert_file_exists(args.paraformer)

    recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
        paraformer=args.paraformer,
        tokens=args.tokens,
        num_threads=args.num_threads,
        sample_rate=args.sample_rate,
        feature_dim=args.feature_dim,
        decoding_method=args.decoding_method,
        debug=args.debug,
    )

    print("Started!")
    start_time = time.time()

    streams, results = [], []
    total_duration = 0

    for i, wave_filename in enumerate(args.sound_files):
        assert_file_exists(wave_filename)
        samples, sample_rate = read_wave(wave_filename)
        duration = len(samples) / sample_rate
        total_duration += duration
        s = recognizer.create_stream()
        s.accept_waveform(sample_rate, samples)

        streams.append(s)
        if i % 10 == 0:
            recognizer.decode_streams(streams)
            results += [s.result.text for s in streams]
            streams = []
            print(f"Processed {i} files")
        # process the last batch
    if streams:
        recognizer.decode_streams(streams)
        results += [s.result.text for s in streams]
    end_time = time.time()
    print("Done!")

    results_dict = {}
    for wave_filename, result in zip(args.sound_files, results):
        print(f"{wave_filename}\n{result}")
        print("-" * 10)
        wave_basename = Path(wave_filename).stem
        results_dict[wave_basename] = result

    elapsed_seconds = end_time - start_time
    rtf = elapsed_seconds / total_duration
    print(f"num_threads: {args.num_threads}")
    print(f"decoding_method: {args.decoding_method}")
    print(f"Wave duration: {total_duration:.3f} s")
    print(f"Elapsed time: {elapsed_seconds:.3f} s")
    print(
        f"Real time factor (RTF): {elapsed_seconds:.3f}/{total_duration:.3f} = {rtf:.3f}"
    )
    
    # Load labels either from file or from dataset
    labels_dict = {}
    
    if args.label:
        # Load labels from file (original functionality)
        print(f"Loading labels from file: {args.label}")
        with open(args.label, "r") as f:
            for line in f:
                # fields = line.strip().split(" ")
                # fields = [item for item in fields if item]
                # assert len(fields) == 4
                # prompt_text, prompt_audio, text, audio_path = fields

                fields = line.strip().split("|")
                fields = [item for item in fields if item]
                assert len(fields) == 4
                audio_path, prompt_text, prompt_audio, text = fields
                labels_dict[Path(audio_path).stem] = normalize_text_alimeeting(text)
    else:
        # Load labels from dataset (new functionality)
        print(f"Loading labels from dataset: {args.dataset_name}, split: {args.split_name}")
        if 'zero' in args.split_name:
            dataset_name = "yuekai/CV3-Eval"
        else:
            dataset_name = "yuekai/seed_tts_cosy2"
        dataset = load_dataset(
            dataset_name,
            split=args.split_name,
            trust_remote_code=True,
        )
        
        for item in dataset:
            audio_id = item["id"]
            labels_dict[audio_id] = normalize_text_alimeeting(item["target_text"])
                
        print(f"Loaded {len(labels_dict)} labels from dataset")

    # Perform evaluation if labels are available
    if labels_dict:

        final_results = []
        for key, value in results_dict.items():
            if key in labels_dict:
                final_results.append((key, labels_dict[key], value))
            else:
                print(f"Warning: No label found for {key}, skipping...")

        if final_results:
            store_transcripts(
                filename=f"{args.log_dir}/recogs-{args.name}.txt", texts=final_results
            )
            with open(f"{args.log_dir}/errs-{args.name}.txt", "w") as f:
                write_error_stats(f, "test-set", final_results, enable_log=True)

            with open(f"{args.log_dir}/errs-{args.name}.txt", "r") as f:
                print(f.readline())  # WER
                print(f.readline())  # Detailed errors
        else:
            print("No matching labels found for evaluation")
    else:
        print("No labels available for evaluation")


if __name__ == "__main__":
    main()