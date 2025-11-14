CHECKPOINT_PATH=${1} # /path/to/global_step_X/actor, where X is the global step of the checkpoint with the best pass@1 on dev

python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir $CHECKPOINT_PATH \
    --target_dir $CHECKPOINT_PATH/hf