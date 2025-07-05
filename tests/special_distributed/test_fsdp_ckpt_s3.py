# Copyright 2025 Amazon.com Inc and/or its affiliates
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
import os
import shutil
import tempfile
from pathlib import PurePosixPath

import pytest

try:
    import boto3
except ImportError:
    pytestmark = pytest.mark.skip(reason="boto3 not installed")

import torch
import torch.distributed
from moto import mock_aws
from torch.distributed import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config

from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.distributed import initialize_global_process_group
from verl.utils.fsdp_utils import MixedPrecisionPolicy, apply_fsdp2
from verl.utils.s3_io import parse_uri


@mock_aws
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs ≥2 GPUs for FSDP")
def test_fsdp_s3_ckpt(strategy: str = "fsdp"):
    local_rank, rank, world_size = initialize_global_process_group()
    device_mesh = init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=("dp",))

    config = Qwen2Config(num_hidden_layers=1)
    with torch.device("cuda"):
        model = AutoModelForCausalLM.from_config(
            config=config,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).to("cuda")

    if strategy == "fsdp":
        mp = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32)
        model = FSDP(
            model,
            device_id=torch.cuda.current_device(),
            use_orig_params=False,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mp,
            device_mesh=device_mesh,
        )
    else:
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=True
        )
        apply_fsdp2(model, {"mesh": device_mesh, "mp_policy": mp_policy}, {})

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    ckpt_mgr = FSDPCheckpointManager(model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, tokenizer=tokenizer)

    bs, seqlen, vocab = 2, 32, 32000
    inp = torch.randint(0, vocab, (bs, seqlen), device="cuda")
    attn = torch.ones_like(inp)

    loss = model(input_ids=inp, attention_mask=attn).logits.mean()
    loss.backward()
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()

    tmpdir = tempfile.mkdtemp()
    local_ckpt_path = os.path.join(tmpdir, "ckpt")

    bucket_name = "unit-test-fsdp-bucket"
    remote_prefix = "fsdp_test_run"
    remote_ckpt_path = f"s3://{bucket_name}/{remote_prefix}"

    boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=bucket_name)
    torch.distributed.barrier()

    ckpt_mgr.save_checkpoint(
        local_path=local_ckpt_path,
        global_step=0,
        remote_path=remote_ckpt_path,
        max_ckpt_to_keep=1,
    )

    # Wait for async uploads to finish before assertions
    ckpt_mgr.wait_for_all_uploads()
    torch.distributed.barrier()

    # Expected shards for *this* rank
    expected_files = [
        f"model_world_size_{world_size}_rank_{rank}.pt",
        f"optim_world_size_{world_size}_rank_{rank}.pt",
        f"extra_state_world_size_{world_size}_rank_{rank}.pt",
    ]
    bucket, prefix, _ = parse_uri(remote_ckpt_path)
    s3 = boto3.client("s3", region_name="us-east-1")

    for fname in expected_files:
        key = str(PurePosixPath(prefix) / fname)
        try:
            s3.head_object(Bucket=bucket, Key=key)
        except s3.exceptions.NoSuchKey:  # pragma: no cover
            pytest.fail(f"S3 object {bucket}/{key} not found (rank {rank})")

    logits_before = model(input_ids=inp, attention_mask=attn).logits.detach()

    ckpt_mgr.load_checkpoint(local_ckpt_path)

    logits_after = model(input_ids=inp, attention_mask=attn).logits.detach()
    torch.testing.assert_close(logits_before, logits_after, atol=0.0, rtol=0.0)

    torch.distributed.barrier()
    shutil.rmtree(tmpdir)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    strategy = os.environ.get("STRATEGY", "fsdp")
    test_fsdp_s3_ckpt(strategy=strategy)
