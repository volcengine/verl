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

import time
import os
import subprocess


def test():
    wait_time = 10

    my_env = os.environ.copy()
    my_env["WAIT_TIME"] = str(wait_time)

    p = subprocess.Popen(["python3", "-u", "./check_worker_alive/main.py"], env=my_env, stdout=subprocess.PIPE)

    count = 0
    while b"foo started" not in p.stdout.read():
        time.sleep(1)
        count += 1
        if count > 40:
            raise RuntimeError("timeout for start foo in check_worker_alive/main.py")

    print(
        time.time(),
        f"wait 1.5 wait time {wait_time*1.5} to let signal returned to process but still not exceed process wait time")
    time.sleep(wait_time * 1.5)
    print(time.time(), f"start checking")
    assert p.poll() is not None, f"process {p} still alive, expecting signal raised abort"
    assert p.returncode != 0, f"process {p} exit with code 0, expecting not-zero exit code"
    print(f"test passed")


if __name__ == "__main__":
    test()
