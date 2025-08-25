# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
from pathlib import Path

from verl.protocol import DataProto


class RolloutSkip:
    """
    RolloutSkip skips sequence generation during rollout by attempting to load previously dumped data.
    If no dumped data is found, it generates new sequences and saves them to disk.

    Args:
        config: The configuration object containing rollout settings.
        rollout_wg: The worker group that handles the rollout process.

    Note:
        Whenever any of the following parameters differ from previous runs—trainer.experiment_name,
        trainer.project_name, rollout.n, or rollout.gen_batch_size—new sequences will be generated
        and saved under different filenames.
    """

    print_mark = "[RolloutSkip()]"

    def __init__(self, config):
        self.rollout_config = config.actor_rollout_ref.rollout
        self.skip_config = self.rollout_config.skip
        self.exp_name = config.trainer.get("experiment_name", "")
        self.project_name = config.trainer.get("project_name", "")

        self.n = int(self.rollout_config.get("n", 0))
        self.gbs = int(config.data.get("gen_batch_size", config.data.get("train_batch_size", 0)))
        self._rollout_wg = None
        self._curr_step: int = -1
        self.strict_mode = self.skip_config.get("strict_mode", True)
        self._create_dump_path()

    @property
    def is_enable(self) -> bool:
        return self.skip_config.get("enable", False)

    @property
    def is_activate(self) -> bool:
        return self.is_enable and self._rollout_wg is not None

    @property
    def curr_path_dump(self):
        return self.dumped_dir.joinpath(f"{self.exp_name}_{self.project_name}_GBS{self.gbs}__N{self.n}").absolute()

    @property
    def curr_step(self):
        return self._curr_step

    def _create_dump_path(self):
        """
        Create the directory for dumping rollout data if it doesn't exist.
        Warn if the directory is within Ray's temporary session directory.
        """

        self.dumped_dir = Path(self.skip_config.get("dump_dir", "/tmp/verl/rollout_dump"))
        self.dumped_dir.mkdir(parents=True, exist_ok=True)

        tmp_ray = "/tmp/ray/session"

        # Check if path is in Ray temporary directory
        if str(self.dumped_dir.absolute()).startswith(tmp_ray):
            print(
                f"\033[33m{self.print_mark} Warning: \nUsing dump path ",
                f"'{self.dumped_dir.absolute()}' is not recommended ",
                f"as it's located in {tmp_ray}*\033[0m",
                flush=True,
            )
        print(
            f"{self.print_mark} Rollout skip dump path set to: ",
            f"{self.dumped_dir.absolute()}",
            flush=True,
        )

    def record(self, new_batch: DataProto, *args, **kwargs):
        """Record the current training step based on the new batch.

        Args:
            new_batch (DataProto): The new batch of data being processed.
        """
        if self._rollout_wg is None:
            return

    def wrap_generate_sequences(self, rollout_wg):
        self._rollout_wg = rollout_wg
        try:
            self._rollout_wg.generate_sequences = wrap_generate_sequences(self, self._rollout_wg)
            print(
                f"{self.print_mark}\033[32m Successfully patched `actor_rollout_wg.generate_sequences()`.\033[0m",
                flush=True,
            )
        except Exception as e:
            raise RuntimeError(
                f"{self.print_mark}\033[31m Failed to patch `actor_rollout_wg.generate_sequences()`.\033[0m",
                flush=True,
            ) from e

    def try_load(self):
        if not self.curr_path_dump.exists():
            print(
                f"{self.print_mark}\033[33m No data dump found at {self.curr_path_dump}.",
                "The trainer will generate and automatically dump the data for this first run.\033[0m",
                flush=True,
            )
            return None

        try:
            # * Load
            ret_batch = DataProto.load_from_disk(self.curr_path_dump)
            print(
                f"\033[32m{self.print_mark} Successfully load pre-generated data from {self.curr_path_dump}\033[0m",
                flush=True,
            )
            return ret_batch
        except Exception as e:
            print(
                f"\033[31m{self.print_mark} Failed to load pre-generated data from {self.curr_path_dump}",
                f"Error: {str(e)}\033[0m",
                flush=True,
            )
            return None

    def dump(self, outputs: DataProto):
        try:
            outputs.save_to_disk(self.curr_path_dump)
            print(
                f"\033[32m{self.print_mark} Successfully dump data in {self.curr_path_dump}\033[0m",
                flush=True,
            )
        except Exception as e:
            print(
                f"\033[31m{self.print_mark} Failed to dump data in {self.curr_path_dump}: {e}\033[0m",
                flush=True,
            )


def check_prompt_with_dumped(batch, dumped_batch):
    """Compare the prompts in the current batch with those in the dumped batch to ensure they match.

    Args:
        batch (_type_): _description_
        dumped_batch (_type_): _description_
    """
    pass
    # batch.batch["input_ids"].shape
    # dumped_batch.batch["input_ids"].shape
    batch.batch["input_ids"] - dumped_batch.batch["input_ids"]


def wrap_generate_sequences(rolloutskip: RolloutSkip, rollout_wg):
    generate_sequences = rollout_wg.generate_sequences

    def warp_fn(batch, **kwargs):
        dumped_batch = rolloutskip.try_load()

        if dumped_batch:
            if rolloutskip.strict_mode:
                # * Check prompt
                check_prompt_with_dumped(batch, dumped_batch)
        else:
            # * 1. Generation
            dumped_batch = generate_sequences(batch, **kwargs)
            # * 2. Dump
            rolloutskip.dump(dumped_batch)

        return dumped_batch

    return warp_fn
