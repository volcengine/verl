# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""Opt-in Ascend allocation diagnostics; stdlib-only until installed in a vLLM worker.

Run this file directly to monitor the existing PPO command on a Linux CI node.
Snapshots add latency and can perturb allocation timing. Never alter allocation
arguments, retry a failed allocation, or suppress the original exception.
"""

import argparse
import datetime
import functools
import itertools
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(path):
    try:
        return Path(path).read_text()
    except OSError as exc:
        return f"UNAVAILABLE: {exc}\n"


def snapshot(destination, detailed=True):
    destination.mkdir(parents=True, exist_ok=True)
    for name in (
        "meminfo",
        "buddyinfo",
        "vmstat",
        "self/status",
        "self/limits",
        "self/cgroup",
        "self/mountinfo",
    ):
        (destination / name.replace("/", "_")).write_text(read("/proc/" + name))
    if detailed:
        (destination / "pagetypeinfo").write_text(read("/proc/pagetypeinfo"))
    for node in Path("/sys/devices/system/node").glob("node[0-9]*"):
        (destination / (node.name + "_meminfo")).write_text(read(node / "meminfo"))
    memberships = [line.split(":", 2) for line in read("/proc/self/cgroup").splitlines() if line.count(":") == 2]
    groups = []
    for line in read("/proc/self/mountinfo").splitlines():
        if " - " not in line:
            continue
        left, right = line.split(" - ", 1)
        fields, fs = left.split(), right.split()
        if fs[0] not in ("cgroup", "cgroup2"):
            continue
        root, mount = fields[3:5]
        for _, controllers, group in memberships:
            if fs[0] == "cgroup2" and controllers:
                continue
            if fs[0] == "cgroup" and not (set(controllers.split(",")) & {"memory", "cpuset"} & set(fs[2].split(","))):
                continue
            relative = os.path.relpath(group, root)
            target = Path(mount) / relative if relative != ".." and not relative.startswith("../") else Path(mount)
            if not target.is_dir():
                target = Path(mount)
            while True:
                entry = {"path": str(target), "mount_root": root, "membership": group}
                for item in target.iterdir():
                    if item.name.startswith(("memory.", "cpuset.")) and item.is_file():
                        entry[item.name] = read(item)
                groups.append(entry)
                if target == Path(mount):
                    break
                target = target.parent
    (destination / "cgroups.json").write_text(json.dumps(groups, indent=2))
    (destination / "timestamp.txt").write_text(now() + "\nAncestors above cgroup mount root may be hidden.\n")


def capture(folder, name, command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=20)
        output = f"exit={result.returncode}\n{result.stdout}{result.stderr}"
    except (OSError, subprocess.TimeoutExpired) as exc:
        output = f"UNAVAILABLE: {exc}"
    (folder / name).write_text(output)


def record(folder, **details):
    """Diagnostic failures must not replace a workload exception."""
    try:
        snapshot(folder, detailed=folder.name in {"installed", "failed", "initial", "final"})
        (folder / "self_maps").write_text(read("/proc/self/maps"))
        details.update(time=now(), pid=os.getpid(), ppid=os.getppid())
        (folder / "event.json").write_text(json.dumps(details, indent=2, default=str))
    except Exception as exc:
        print(f"Ascend memory diagnostic failed: {exc!r}", file=sys.stderr, flush=True)


def install():
    root = os.environ.get("VERL_ASCEND_MEMORY_DIAGNOSTICS")
    if not root:
        return
    import torch

    if getattr(torch.empty, "_ascend_memory_diagnostics", False):
        return
    original = torch.empty
    sequence = itertools.count()
    worker = Path(root) / "workers" / str(os.getpid())
    record(worker / "installed", torch_version=torch.__version__)

    @functools.wraps(original)
    def empty(*args, **kwargs):
        if not kwargs.get("pin_memory", False):
            return original(*args, **kwargs)
        allocation = worker / f"allocation-{next(sequence):06d}"
        request = {
            "args": repr(args),
            "kwargs": repr(kwargs),
            "stack": traceback.format_stack(),
        }
        record(allocation / "before", **request)
        started = time.monotonic()
        try:
            result = original(*args, **kwargs)
        except Exception:
            # Snapshot the failing process before expensive kernel-log collection.
            record(allocation / "failed", exception=traceback.format_exc(), elapsed_s=time.monotonic() - started)
            try:
                capture(allocation / "failed", "kernel.txt", ["dmesg", "-T"])
            except Exception as exc:
                print(f"Kernel capture failed: {exc!r}", file=sys.stderr, flush=True)
            raise
        record(allocation / "after", elapsed_s=time.monotonic() - started)
        return result

    empty._ascend_memory_diagnostics = True
    torch.empty = empty


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("a workload command is required after --")
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    os.environ["VERL_ASCEND_MEMORY_DIAGNOSTICS"] = str(root)
    for name, cmd in {
        "uname.txt": ["uname", "-a"],
        "npu-smi.txt": ["npu-smi", "info"],
        "packages.txt": [sys.executable, "-m", "pip", "list"],
        "kernel-before.txt": ["dmesg", "-T"],
        "revision.txt": ["git", "rev-parse", "HEAD"],
    }.items():
        capture(root, name, cmd)
    record(root / "initial", command=command)
    # Console output is captured by the workflow's tee; preserve child exit status.
    child = subprocess.Popen(command)
    sample = 0
    while child.poll() is None:
        record(root / "host" / f"sample-{sample:06d}")
        sample += 1
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    record(root / "final", exit_code=child.returncode)
    capture(root, "kernel-after.txt", ["dmesg", "-T"])
    installed = list(root.glob("workers/*/installed/event.json"))
    allocations = list(root.glob("workers/*/allocation-*/before/event.json"))
    failures = list(root.glob("workers/*/allocation-*/failed/event.json"))
    sleep_allocations = []
    for event in allocations:
        stack = "".join(json.loads(event.read_text())["stack"])
        if "vllm_ascend" in stack and "sleep" in stack:
            sleep_allocations.append(str(event.relative_to(root)))
    summary = {
        "exit_code": child.returncode,
        "instrumented_workers": len(installed),
        "pinned_allocation_requests": len(allocations),
        "failed_allocations": len(failures),
        "vllm_ascend_sleep_allocations": sleep_allocations,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    if child.returncode:
        return child.returncode if child.returncode > 0 else 128 - child.returncode
    if not sleep_allocations:
        print("Diagnostic incomplete: no instrumented vllm_ascend sleep allocation observed", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
