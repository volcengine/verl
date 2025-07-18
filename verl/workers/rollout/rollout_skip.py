from pathlib import Path
import torch
import tensordict
from verl.protocol import DataProto


class RolloutSkip:

    def __init__(self, config, rollout_wg):
        self.dump_step = 0

        self.rollout_config = config.actor_rollout_ref.rollout

        self.n = int(self.rollout_config.get("n", 0))
        self.gen_gbs = int(config.data.get("gen_batch_size", 0))

        self.skip_use_default_dir = Path(self.rollout_config.get("skip_use_default_dir", "tmp/rollout_dump"))
        self.skip_use_default_dir = self.skip_use_default_dir.joinpath(f"InferGBS{self.gen_gbs}__N{self.n}")
        self.skip_use_default_dir.mkdir(exist_ok=True, parents=True)

        # * check is in ray tmp path
        if self.skip_use_default_dir.absolute().__str__().startswith("/tmp/ray/session"):
            print(
                f"\033[33m[RolloutSkip()] Warning: \nit is not recommanded to use dump path ",
                f"\033[0m'{self.skip_use_default_dir.absolute()}'\033[33m which is relative to /tmp/ray/session*",
            )

        # todo add dump with step num
        self.skip_rollout_dumpsteps = int(self.rollout_config.get("skip_rollout_dumpsteps", 1))
        print(f"\033[33mUsing rollout skip dump path: \n{self.skip_use_default_dir.absolute()}\033[0m", flush=True)

        rollout_wg.generate_sequences = wrap_generate_sequences(self, rollout_wg)

    @property
    def curr_path_dump(self):
        return self.skip_use_default_dir.joinpath(f"dump_step_{self.dump_step}")

    def try_load(self):
        if not self.curr_path_dump.exists():
            return None
        try:
            # * Load
            # ret_batch = tensordict.TensorDict.load_memmap(self.curr_path_dump)
            ret_batch = DataProto.load_from_disk(self.curr_path_dump)
            # todo check if `n` matched current config，otherwise copy part of dumped data.
            print(f"\033[32mSuccessed to load dumped data from {self.curr_path_dump.absolute()}\033[0m", flush=True)
            return ret_batch
        except:
            print(f"\033[31mFailed to load dumped data from {self.curr_path_dump.absolute()}\033[0m", flush=True)

        return None

    def dump(self, outputs):
        outputs.save_to_disk(self.curr_path_dump)
        print(f"\033[32mSuccessed to dump data in {self.curr_path_dump.absolute()}\033[0m", flush=True)
        # self.dump_step += 1


def wrap_generate_sequences(rolloutskip: RolloutSkip, rollout_wg):
    generate_sequences = rollout_wg.generate_sequences

    def warp_fn(batch, **kwargs):
        gen_batch_output = rolloutskip.try_load()

        if gen_batch_output is None:
            # * 1. Generation
            gen_batch_output = generate_sequences(batch, **kwargs)
            # * 2. Dump
            rolloutskip.dump(gen_batch_output)
        return gen_batch_output

    print(f"SkipRollout patched `actor_rollout_wg.generate_sequences()` successfully.", flush=True)
    return warp_fn
