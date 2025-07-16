#!/usr/bin/env bash
set -euox pipefail

# 1. Dump the full config to a temp file for ppo_trainer
target_cfg=verl/trainer/config/_generated_ppo_trainer.yaml
tmp_header=$(mktemp)
tmp_cfg=$(mktemp)
echo "# This reference configration yaml is automatically generated via 'scripts/generate_trainer_config.sh'" > "$tmp_header"
echo "# in which it invokes 'python3 scripts/print_cfg.py --cfg job' to flatten the 'verl/trainer/config/ppo_trainer.yaml' config fields into a single file." >> "$tmp_header"
echo "# Do not modify this file directly." >> "$tmp_header"
echo "# The file is usually only for reference and never used." >> "$tmp_header"
echo "" >> "$tmp_header"
python3 scripts/print_cfg.py --cfg job > "$tmp_cfg"

# 2. Extract from the line starting with "actor_rollout_ref" onward
cat $tmp_header > $target_cfg
sed -n '/^actor_rollout_ref/,$p' "$tmp_cfg" >> $target_cfg

# 3. Clean up
rm "$tmp_cfg" "$tmp_header"

target_megatron_cfg=verl/trainer/config/_generated_ppo_megatron_trainer.yaml
tmp_megatron_header=$(mktemp)
tmp_megatron_cfg=$(mktemp)
echo "# This reference configration yaml is automatically generated via 'scripts/generate_trainer_config.sh'" > "$tmp_megatron_header"
echo "# in which it invokes 'python3 scripts/print_cfg.py --cfg job --config-name='ppo_megatron_trainer.yaml'' to flatten the 'verl/trainer/config/ppo_megatron_trainer.yaml' config fields into a single file." >> "$tmp_megatron_header"
echo "# Do not modify this file directly." >> "$tmp_megatron_header"
echo "# The file is usually only for reference and never used." >> "$tmp_megatron_header"
echo "" >> "$tmp_megatron_header"
python3 scripts/print_cfg.py --cfg job --config-name='ppo_megatron_trainer.yaml' > "$tmp_megatron_cfg"

cat $tmp_megatron_header > $target_megatron_cfg
sed -n '/^actor_rollout_ref/,$p' "$tmp_megatron_cfg" >> $target_megatron_cfg

rm "$tmp_megatron_cfg" "$tmp_megatron_header"

# 7. Verify that verl/trainer/config/_generated_ppo_trainer.yaml wasn't changed on disk
if ! git diff --exit-code -- "$target_cfg" >/dev/null; then
  echo "✖ $target_cfg is out of date.  Please regenerate via 'scripts/generate_trainer_config.sh' and commit the changes."
  exit 1
fi

# 8. Verify that verl/trainer/config/_generated_ppo_megatron_trainer.yaml wasn't changed on disk
if ! git diff --exit-code -- "$target_megatron_cfg" >/dev/null; then
  echo "✖ $target_megatron_cfg is out of date.  Please regenerate via 'scripts/generate_trainer_config.sh' and commit the changes."
  exit 1
fi

echo "All good"
exit 0
