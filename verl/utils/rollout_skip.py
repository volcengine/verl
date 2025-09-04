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
import pickle
import time
from enum import Enum
from pathlib import Path

from verl.protocol import DataProto


class PostDumpAction(Enum):
    REPEAT = "repeat"  # Repeat the sample
    REPEAT_LAST = "repeat_last"  # Repeat the last sample
    ROLLOUT = "rollout"  # rollout the sample
    ROLLOUT_WITH_DUMP = "rollout_with_dump"  # rollout the sample and dump


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

    print_mark = "[RolloutSkip()] "

    def __init__(self, config):
        self.rollout_config = config.actor_rollout_ref.rollout
        self.skip_config = self.rollout_config.skip
        self.is_enable = self.skip_config.get("enable", False)
        self._rollout_wg = None

        if not self.is_enable:
            return

        self.exp_name = config.trainer.get("experiment_name", "")
        self.project_name = config.trainer.get("project_name", "")
        self.n = int(self.rollout_config.get("n", 0))
        self.gbs = int(config.data.get("gen_batch_size", config.data.get("train_batch_size", 0)))
        self.response_length = config.data.get("max_response_length", 0)
        self.prompt_length = config.data.get("max_prompt_length", 0)

        self._new_batch = None
        self.curr_step: int = 0

        self.do_compress = self.skip_config.get("compress", True)
        self.dump_step = max(0, self.skip_config.get("dump_step", 1))  # at least dump once
        self.post_dump_action = self.skip_config.get("post_dump_action", PostDumpAction.REPEAT)
        self.post_dump_action = PostDumpAction(self.post_dump_action)

        if self.dump_step <= 0:
            assert self.post_dump_action in [PostDumpAction.ROLLOUT, PostDumpAction.ROLLOUT_WITH_DUMP]

        self._create_dump_path()
        self._flag_record = False
        self.list_dumped_steps = []

    @property
    def is_activate(self) -> bool:
        """
        If RolloutSkip is enabled and the rollout worker group is set, it is considered active.
        """
        return self.is_enable and self._rollout_wg is not None

    @property
    def is_dump_step(self) -> bool:
        """
        Determine if the current step is a dump step based on the configured dump interval.
        """
        return self.is_activate and self.curr_step <= self.dump_step

    @property
    def num_dumped_step(self) -> int:
        return len(self.list_dumped_steps)

    def get_path_dump(self, step: int = None) -> Path:
        if step is None:
            step = self.curr_step
        return self.specify_dumped_dir.joinpath(f"genstep_{step:06d}.pkl").absolute()

    def _create_dump_path(self):
        """
        Create the directory for dumping rollout data if it doesn't exist.
        Warn if the directory is within Ray's temporary session directory.
        """

        dumped_dir = Path(self.skip_config.get("dump_dir", "/tmp/verl/rollout_dump"))
        sub_dir = (
            f"{self.exp_name}_{self.project_name}"
            + f"/GBS{self.gbs}_N{self.n}_in{self.prompt_length}_out{self.response_length}"
        )

        self.specify_dumped_dir = dumped_dir.joinpath(sub_dir)
        self.specify_dumped_dir.mkdir(parents=True, exist_ok=True)

        tmp_ray = "/tmp/ray/session"

        # Check if path is in Ray temporary directory
        if str(self.specify_dumped_dir.absolute()).startswith(tmp_ray):
            print(
                f"{self.print_mark}\033[33mWarning: \nUsing dump path ",
                f"'{self.specify_dumped_dir.absolute()}' is not recommended ",
                f"as it's located in {tmp_ray}*\033[0m",
                flush=True,
            )
        print(
            f"{self.print_mark}Rollout skip dump path set to: ",
            str(self.specify_dumped_dir.absolute()),
            flush=True,
        )

    def record(self, new_batch: DataProto, *args, **kwargs):
        """Record the current training step based on the new batch.

        Args:
            new_batch (DataProto): The new batch of data being processed.
        """
        if self._rollout_wg is None:
            return
        if self._flag_record is False:
            # make sure one record only corresponds to one skip
            self._flag_record = True
            self._new_batch = new_batch
        else:
            print(f"{self.print_mark}Warning, duplicate record new_batch.", flush=True)

    def wrap_generate_sequences(self, rollout_wg):
        if self.is_enable:
            self._rollout_wg = rollout_wg

        try:
            self._rollout_wg.generate_sequences = wrap_generate_sequences(self, self._rollout_wg)
            print(
                f"{self.print_mark}\033[32mSuccessfully patched `actor_rollout_wg.generate_sequences()`.\033[0m",
                flush=True,
            )
        except Exception as e:
            raise RuntimeError(
                f"{self.print_mark}\033[31mFailed to patch `actor_rollout_wg.generate_sequences()`.\033[0m",
                flush=True,
            ) from e

    def try_load(self, step=None):
        dumped_gen_batch = None
        dumped_new_batch = None
        if step is None:
            step = self.curr_step

        path_dump = self.get_path_dump(step)
        if path_dump.exists():
            try:
                # * Load
                data_dict = pickle.loads(path_dump.read_bytes())
                dataproto_decompress(data_dict)

                dumped_gen_batch = data_dict["gen_batch"]
                dumped_new_batch = data_dict["new_batch"]

                print(
                    f"{self.print_mark}\033[32mSuccessfully load pre-generated data from {path_dump}.\033[0m",
                    flush=True,
                )

                if step not in self.list_dumped_steps:
                    self.list_dumped_steps.append(step)

            except Exception:
                print(
                    f"{self.print_mark}\033[31mFailed to load pre-generated data from {path_dump}.\033[0m",
                    flush=True,
                )

        else:
            print(
                f"{self.print_mark}\033[33mNo dumped data found at gen_step {step}",
                f"from {path_dump}. The trainer will generate and dump the data for this gen_step.\033[0m",
                flush=True,
            )

        return dumped_new_batch, dumped_gen_batch

    def dump(self, outputs: DataProto):
        # todo raise error in dump is too late, fix it later.
        if self._flag_record is False or self._new_batch is None:
            raise AssertionError(
                f"{self.print_mark}\033[33mError: \n"
                + "The new_batch record is required."
                + "Please record the new_batch using `RolloutSkip.record(new_batch)` in trainer.fit().\033[0m"
            )
        self._flag_record = False

        data_dump = {
            "new_batch": self._new_batch,
            "gen_batch": outputs,
            "compressed": [],
        }

        try:
            info_compress = ""
            if self.do_compress:
                data_dump["compressed"] = ["gen_batch", "new_batch"]
                dict_info = dataproto_compress(data_dump)
                size_zip = dict_info["size_compressed_data"]
                size_data = dict_info["size_data"]
                ratio = dict_info["ratio"]

                if size_data != 0:
                    info_compress = f"{size_data / 1024**2:.3f}MB -> {size_zip / 1024**2:.3f}MB ({ratio:.1%} )"

            with open(str(self.get_path_dump()), "wb") as f:
                pickle.dump(data_dump, f)

            print(
                f"{self.print_mark}\033[32mSuccessfully dump data in {self.get_path_dump()}\033[0m",
                info_compress,
                flush=True,
            )
            if self.curr_step not in self.list_dumped_steps:
                self.list_dumped_steps.append(self.curr_step)

        except Exception as e:
            print(
                f"{self.print_mark}\033[31mFailed to dump data in {self.get_path_dump()}: {e}\033[0m",
                flush=True,
            )

    def replace_curr_new_batch(self, dumped_new_batch: DataProto):
        """Replace the current new_batch's content with that from the dumped_new_batch.
        In case of [Answer] mismatch.
        """

        if self._flag_record is False:
            raise AssertionError(
                f"{self.print_mark}\033[33mError: \n"
                + "The new_batch is not recorded. Please record the new_batch"
                + "using `RolloutSkip.record(new_batch)`. \033[0m"
            )
        self._flag_record = False

        self._new_batch.batch = dumped_new_batch.batch
        self._new_batch.non_tensor_batch = dumped_new_batch.non_tensor_batch
        self._new_batch.meta_info = dumped_new_batch.meta_info


def wrap_generate_sequences(rolloutskip: RolloutSkip, rollout_wg):
    generate_sequences = rollout_wg.generate_sequences

    def rollout_skip_wrap_fn(batch, **kwargs) -> DataProto:
        rolloutskip.curr_step += 1
        return_batch = None

        if rolloutskip.is_dump_step:
            # * try load
            dumped_new_batch, return_batch = rolloutskip.try_load()

            if return_batch is None:
                # 1. Generation
                return_batch = generate_sequences(batch, **kwargs)
                # 2. Dump
                rolloutskip.dump(return_batch)
            else:
                rolloutskip.replace_curr_new_batch(dumped_new_batch)

        elif rolloutskip.post_dump_action == PostDumpAction.REPEAT:
            target_step = rolloutskip.list_dumped_steps[(rolloutskip.curr_step - 1) % rolloutskip.num_dumped_step]
            dumped_new_batch, return_batch = rolloutskip.try_load(step=target_step)
            if return_batch is None:
                return_batch = generate_sequences(batch, **kwargs)
                rolloutskip.dump(return_batch)
            else:
                rolloutskip.replace_curr_new_batch(dumped_new_batch)

        elif rolloutskip.post_dump_action == PostDumpAction.REPEAT_LAST:
            target_step = rolloutskip.list_dumped_steps[-1]
            dumped_new_batch, return_batch = rolloutskip.try_load(step=target_step)
            if return_batch is None:
                return_batch = generate_sequences(batch, **kwargs)
                rolloutskip.dump(return_batch)
            else:
                rolloutskip.replace_curr_new_batch(dumped_new_batch)

        elif rolloutskip.post_dump_action == PostDumpAction.ROLLOUT_WITH_DUMP:
            return_batch = generate_sequences(batch, **kwargs)
            rolloutskip.dump(return_batch)

        elif rolloutskip.post_dump_action == PostDumpAction.ROLLOUT:
            return_batch = generate_sequences(batch, **kwargs)

            # clean
        return return_batch

    return rollout_skip_wrap_fn


def dataproto_compress(dict_data: dict) -> dict[str, DataProto]:
    try:
        import pyzstd

        compresser = pyzstd
    except ImportError:
        import zlib

        compresser = zlib

    dict_data["compresser_name"] = compresser.__name__

    key_compress = dict_data.get("compressed", [])

    size_data = 0
    size_compressed_data = 0

    print("Compress dumped data...", flush=True)
    time_pickle = 0
    time_compress = 0
    for key in key_compress:
        time_start = time.time()
        _data = pickle.dumps(dict_data[key])
        time_pickle += time.time() - time_start
        size_data += len(_data)

        time_start = time.time()
        compressed_data = compresser.compress(_data)
        time_compress += time.time() - time_start

        size_compressed_data += len(compressed_data)

        dict_data[key] = compressed_data

    dict_info = {
        "size_compressed_data": size_compressed_data,
        "size_data": size_data,
        "time_pickle": time_pickle,
        "time_compress": time_compress,
        "ratio": size_compressed_data / size_data if size_data != 0 else None,
    }

    return dict_info


def dataproto_decompress(dict_data: dict[str, DataProto]) -> dict[str, DataProto]:
    key_compresser_name = dict_data.get("compresser_name", "zlib")
    if key_compresser_name == "zlib":
        import zlib

        compresser = zlib
    elif key_compresser_name == "pyzstd":
        import pyzstd

        compresser = pyzstd

    key_compress = dict_data.get("compressed", [])

    for key in key_compress:
        compressed_data = compresser.decompress(dict_data[key])
        _data = pickle.loads(compressed_data)
        dict_data[key] = _data

    dict_data["compressed"] = []


def read_dumped_data(path_dump: Path) -> dict[str, DataProto]:
    """
    Common function to read and decompress dumped data from a specified path.

    ```
    import verl
    from verl.utils.rollout_skip import read_dumped_data

    dumped_data = read_dumped_data("tmp/rollout_dump/DAPO-Qwen2.5-0.5B_DAPO/GBS4_N4_in2048_out4096/genstep_000001.pkl")

    print(dumped_data["new_batch"])
    print(dumped_data["gen_batch"])
    ```

    """
    path_dump = Path(path_dump)
    if path_dump.is_file():
        with open(path_dump, "rb") as f:
            data_dump = pickle.load(f)
    else:
        raise FileNotFoundError(f"File {path_dump} does not exist.")

    dataproto_decompress(data_dump)

    return data_dump
