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

import numpy as np
import ray
import torch

from verl import DataProto
from verl.trainer.ppo.reward import compute_reward, compute_reward_async


def main():
    # -------------------------
    # 1. Build a DataProto
    # -------------------------
    tensors = {
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=torch.float32),  # shape (4, 3)
    }
    non_tensors = {
        "text": np.array(["a", "b", "c", "d"], dtype=object),
    }
    meta_info = {"split": "demo"}

    dp = DataProto.from_dict(
        tensors=tensors,
        non_tensors=non_tensors,
        meta_info=meta_info,
    )

    # -------------------------
    # 2. Define a simple reward function
    # -------------------------
    def simple_reward_fn(data_proto, return_dict=False):
        # Sum each row in input_ids → shape (batch_size,)
        batch = data_proto.batch["input_ids"]
        rewards = batch.sum(dim=1)
        if return_dict:
            return {"reward_tensor": rewards, "reward_extra_info": {"note": "sum_of_inputs"}}
        return rewards

    # -------------------------
    # 3. Synchronous: main_process vs multi-thread
    # -------------------------
    r_single, info_single = compute_reward(dp, simple_reward_fn, num_workers=0)
    r_multi, info_multi = compute_reward(dp, simple_reward_fn, num_workers=4)

    # If multi-thread returns a list, concatenate into a Tensor
    if isinstance(r_multi, list):
        r_multi = torch.cat(r_multi, dim=0)

    print("Single-worker rewards:", r_single)
    print("Multi-worker  rewards:", r_multi)
    print("Single-worker info:", info_single)
    print("Multi-worker  info:", info_multi)

    assert torch.allclose(r_single, r_multi), "Mismatch between main process and multi-thread outputs!"
    assert info_single == info_multi, "Mismatch in extra_info between main process and multi-thread!"

    # -------------------------
    # 4. Asynchronous: Ray remote call
    # -------------------------
    ray.init(local_mode=True, ignore_reinit_error=True)

    # Minimal config stub for compute_reward_async.remote
    class CFG:
        def __init__(self):
            self.reward_model = type("M", (), {"get": lambda self, k, default=None: {"reward_manager": "naive", "sandbox_fusion": None, "reward_kwargs": {}}.get(k, default), "reward_fn_workers": 4})()
            self.data = type("D", (), {"reward_fn_key": "reward_tensor"})

    cfg = CFG()
    tokenizer = None  # Not used in this example

    # Monkey-patch load_reward_manager to return our simple_reward_fn
    import verl.trainer.ppo.reward as Rmod

    original_loader = Rmod.load_reward_manager
    Rmod.load_reward_manager = lambda config, tokenizer, num_examine, **kw: simple_reward_fn

    # Remote invocation
    obj_ref = compute_reward_async.remote(dp, cfg, tokenizer)
    r_async, info_async = ray.get(obj_ref)

    # Restore original loader
    Rmod.load_reward_manager = original_loader

    ray.shutdown()

    # If result is a list, concatenate into a Tensor
    if isinstance(r_async, list):
        r_async = torch.cat(r_async, dim=0)

    print("Async-worker rewards:", r_async)
    print("Async-worker info:", info_async)

    assert torch.allclose(r_single, r_async), "Mismatch between sync and async outputs!"
    assert info_single == info_async, "Mismatch in extra_info between sync and async!"

    # -------------------------
    # 5. Final assertion
    # -------------------------
    expected = torch.tensor([6.0, 15.0, 24.0, 33.0])
    assert torch.allclose(r_single, expected), f"Results {r_single} do not match expected {expected}"

    print("All tests passed!")


if __name__ == "__main__":
    main()
