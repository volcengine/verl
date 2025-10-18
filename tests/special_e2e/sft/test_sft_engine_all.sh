#!/usr/bin/env bash
set -xeuo pipefail

rm -rf ~/verl/test/log
mkdir -p ~/verl/test/log

export VERL_FILE_LOGGER_ROOT=~/verl/test/log

# test with single gpu as golden
echo "run with single gpu as golden"
BACKEND=fsdp SP_SIZE=1 FSDP_SIZE=1 NUM_GPUS=1 FSDP_STRATEGY=fsdp VERL_FILE_LOGGER_PATH=~/verl/test/log/golden.jsonl bash tests/special_e2e/sft/run_sft_engine_gsm8k.sh

# test with fsdp 1
echo "run with sp1 fsdp_size2 num_gpus8 fsdp_strategy fsdp pad_mode no_padding"
BACKEND=fsdp SP_SIZE=1 FSDP_SIZE=2 NUM_GPUS=8 FSDP_STRATEGY=fsdp PAD_MODE=no_padding bash tests/special_e2e/sft/run_sft_engine_gsm8k.sh
echo "run with sp1 fsdp_size-1 num_gpus8 fsdp_strategy fsdp pad_mode no_padding"
BACKEND=fsdp SP_SIZE=1 FSDP_SIZE=-1 NUM_GPUS=8 FSDP_STRATEGY=fsdp PAD_MODE=no_padding bash tests/special_e2e/sft/run_sft_engine_gsm8k.sh
echo "run with sp2 fsdp_size-1 num_gpus8 fsdp_strategy fsdp pad_mode no_padding"
BACKEND=fsdp SP_SIZE=2 FSDP_SIZE=-1 NUM_GPUS=8 FSDP_STRATEGY=fsdp PAD_MODE=no_padding bash tests/special_e2e/sft/run_sft_engine_gsm8k.sh
echo "run with sp4 fsdp_size4 num_gpus8 fsdp_strategy fsdp pad_mode no_padding"
BACKEND=fsdp SP_SIZE=4 FSDP_SIZE=4 NUM_GPUS=8 FSDP_STRATEGY=fsdp PAD_MODE=no_padding bash tests/special_e2e/sft/run_sft_engine_gsm8k.sh

# test use_remove_padding and pad_mode no_padding
echo "run with sp4 fsdp_size4 num_gpus8 fsdp_strategy fsdp pad_mode no_padding use_remove_padding False"
BACKEND=fsdp SP_SIZE=1 FSDP_SIZE=-1 NUM_GPUS=8 FSDP_STRATEGY=fsdp PAD_MODE=no_padding USE_REMOVE_PADDING=False bash tests/special_e2e/sft/run_sft_engine_gsm8k.sh


# test with fsdp 2
echo "run with sp1 fsdp_size1 num_gpus1 fsdp_strategy fsdp2"
BACKEND=fsdp SP_SIZE=1 FSDP_SIZE=1 NUM_GPUS=1 FSDP_STRATEGY=fsdp2 bash tests/special_e2e/sft/run_sft_engine_gsm8k.sh

echo "run with sp1 fsdp_size-1 num_gpus8 fsdp_strategy fsdp2"
BACKEND=fsdp SP_SIZE=1 FSDP_SIZE=-1 NUM_GPUS=8 FSDP_STRATEGY=fsdp2 bash tests/special_e2e/sft/run_sft_engine_gsm8k.sh
echo "run with sp2 fsdp_size-1 num_gpus8 fsdp_strategy fsdp2"
BACKEND=fsdp SP_SIZE=2 FSDP_SIZE=-1 NUM_GPUS=8 FSDP_STRATEGY=fsdp2 bash tests/special_e2e/sft/run_sft_engine_gsm8k.sh
BACKEND=fsdp SP_SIZE=1 FSDP_SIZE=2 NUM_GPUS=8 FSDP_STRATEGY=fsdp2 bash tests/special_e2e/sft/run_sft_engine_gsm8k.sh
BACKEND=fsdp SP_SIZE=4 FSDP_SIZE=4 NUM_GPUS=8 FSDP_STRATEGY=fsdp2 bash tests/special_e2e/sft/run_sft_engine_gsm8k.sh

# test with megatron
echo "run with tp1 pp1 cp1 num_gpus1"
BACKEND=megatron TP_SIZE=1 PP_SIZE=1 CP_SIZE=1 NUM_GPUS=1 bash tests/special_e2e/sft/run_sft_engine_gsm8k.sh

echo "run with tp2 pp2 vpp2 cp1 num_gpus8"
BACKEND=megatron TP_SIZE=2 PP_SIZE=2 VPP_SIZE=2 CP_SIZE=1 NUM_GPUS=8 bash tests/special_e2e/sft/run_sft_engine_gsm8k.sh

# TODO: toggle with following test when cp is fixed
# BACKEND=megatron TP_SIZE=2 PP_SIZE=2 VPP_SIZE=2 CP_SIZE=1 NUM_GPUS=8 bash tests/special_e2e/sft/run_sft_engine_gsm8k.sh >& ~/verl/test/log/gsm8k-tp2_pp2_vpp2_cp1_num_gpus8.log

python3 tests/special_e2e/sft/compare_sft_engine_results.py

rm -rf ~/verl/test/log
