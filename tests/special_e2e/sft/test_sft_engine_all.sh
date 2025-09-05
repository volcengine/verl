mkdir -p ~/verl/test/log

# test with single gpu as golden
echo "run with single gpu as golden"
BACKEND=fsdp SP_SIZE=1 FSDP_SIZE=1 NUM_GPUS=1 FSDP_STRATEGY=fsdp bash tests/special_e2e/sft/run_sft_engine_gsm8k.sh >& ~/verl/test/log/golden.log

# test with fsdp 1
echo "run with sp1 fsdp_size2 num_gpus8 fsdp_strategy fsdp"
BACKEND=fsdp SP_SIZE=1 FSDP_SIZE=2 NUM_GPUS=8 FSDP_STRATEGY=fsdp bash tests/special_e2e/sft/run_sft_engine_gsm8k.sh >& ~/verl/test/log/gsm8k-sp1_fsdp_size2_strategy_fsdp_num_gpus8.log
echo "run with sp1 fsdp_size-1 num_gpus8 fsdp_strategy fsdp"
BACKEND=fsdp SP_SIZE=1 FSDP_SIZE=-1 NUM_GPUS=8 FSDP_STRATEGY=fsdp bash tests/special_e2e/sft/run_sft_engine_gsm8k.sh >& ~/verl/test/log/gsm8k-sp1_fsdp_size-1_strategy_fsdp_num_gpus8.log
echo "run with sp2 fsdp_size-1 num_gpus8 fsdp_strategy fsdp"
BACKEND=fsdp SP_SIZE=2 FSDP_SIZE=-1 NUM_GPUS=8 FSDP_STRATEGY=fsdp bash tests/special_e2e/sft/run_sft_engine_gsm8k.sh >& ~/verl/test/log/gsm8k-sp2_fsdp_size-1_strategy_fsdp_num_gpus8.log
echo "run with sp4 fsdp_size4 num_gpus8 fsdp_strategy fsdp"
BACKEND=fsdp SP_SIZE=4 FSDP_SIZE=4 NUM_GPUS=8 FSDP_STRATEGY=fsdp bash tests/special_e2e/sft/run_sft_engine_gsm8k.sh >& ~/verl/test/log/gsm8k-sp4_fsdp_size4_strategy_fsdp_num_gpus8.log


# TODO: toggle the follow tests when the grad norm of fsdp is fixed
# BACKEND=fsdp SP_SIZE=1 FSDP_SIZE=1 NUM_GPUS=1 FSDP_STRATEGY=fsdp2 bash tests/special_e2e/sft/run_sft_engine_gsm8k.sh
# BACKEND=fsdp SP_SIZE=1 FSDP_SIZE=2 NUM_GPUS=8 FSDP_STRATEGY=fsdp2 bash tests/special_e2e/sft/run_sft_engine_gsm8k.sh
# BACKEND=fsdp SP_SIZE=1 FSDP_SIZE=-1 NUM_GPUS=8 FSDP_STRATEGY=fsdp2 bash tests/special_e2e/sft/run_sft_engine_gsm8k.sh
# BACKEND=fsdp SP_SIZE=2 FSDP_SIZE=-1 NUM_GPUS=8 FSDP_STRATEGY=fsdp2 bash tests/special_e2e/sft/run_sft_engine_gsm8k.sh
# BACKEND=fsdp SP_SIZE=4 FSDP_SIZE=4 NUM_GPUS=8 FSDP_STRATEGY=fsdp2 bash tests/special_e2e/sft/run_sft_engine_gsm8k.sh

# test with megatron
echo "run with tp1 pp1 cp1 num_gpus1"
BACKEND=megatron TP_SIZE=1 PP_SIZE=1 CP_SIZE=1 NUM_GPUS=1 bash tests/special_e2e/sft/run_sft_engine_gsm8k.sh >& ~/verl/test/log/gsm8k-tp1_pp1_cp1_num_gpus1.log
echo "run with tp2 pp2 vpp2 cp1 num_gpus8"
BACKEND=megatron TP_SIZE=2 PP_SIZE=2 VPP_SIZE=2 CP_SIZE=1 NUM_GPUS=8 bash tests/special_e2e/sft/run_sft_engine_gsm8k.sh >& ~/verl/test/log/gsm8k-tp2_pp2_vpp2_cp1_num_gpus8.log

# TODO: toggle with following test when cp is fixed
# BACKEND=megatron TP_SIZE=2 PP_SIZE=2 VPP_SIZE=2 CP_SIZE=1 NUM_GPUS=8 bash tests/special_e2e/sft/run_sft_engine_gsm8k.sh >& ~/verl/test/log/gsm8k-tp2_pp2_vpp2_cp1_num_gpus8.log
