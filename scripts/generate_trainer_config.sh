#!/usr/bin/env bash
set -euox pipefail

# 1. Dump the full config to a temp file
target_cfg=verl/trainer/config/_generated_ppo_trainer.yaml
tmp_cfg=$(mktemp)
echo "# This reference configration yaml is automatically generated via 'scripts/generate_trainer_config.sh'" > "$target_cfg"
echo "# in which it invokes 'python3 scripts/print_cfg.py --cfg job' to flatten the 'verl/trainer/config/ppo_trainer.yaml' config fields into a single file." >> "$target_cfg"
echo "# Do not modify this file directly." >> "$target_cfg"
echo "# The file is usually only for reference and never used." >> "$target_cfg"
echo "" >> "$target_cfg"
python3 scripts/print_cfg.py --cfg job > "$tmp_cfg"

# 2. Extract from the line starting with "actor_rollout_ref" onward
sed -n '/^actor_rollout_ref/,$p' "$tmp_cfg" >> $target_cfg

# 3. Clean up
rm "$tmp_cfg"

# 4. Verify that verl/trainer/config/_generated_ppo_trainer.yaml wasn't changed on disk
if ! git diff verl/trainer/config --exit-code -- "$target_cfg" >/dev/null; then
  echo "✖ $target_cfg is out of date.  Please regenerate via 'scripts/generate_trainer_config.sh' and commit the changes."
  exit 1
fi
echo "All good"
exit 0