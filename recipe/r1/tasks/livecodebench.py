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

import multiprocessing
import json
import pickle
import zlib
import base64

# Reuse `run_test` for convenience
from verl.utils.reward_score.prime_code.testing_util import run_test


def _temp_run(in_outs, generation, debug, result, metadata_list, timeout):
    res, metadata = run_test(in_outs, test=generation, debug=debug, timeout=timeout)
    result.append(res)
    metadata_list.append(metadata)


def check_correctness(in_outs, generation, timeout, debug=True):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""

    manager = multiprocessing.Manager()
    result = manager.list()
    metadata_list = manager.list()
    p = multiprocessing.Process(
        target=_temp_run,
        args=(in_outs, generation, debug, result, metadata_list, timeout),
    )
    p.start()
    p.join(timeout=(timeout + 1) * len(in_outs["inputs"]) + 5)
    if p.is_alive():
        p.kill()
    if not result:
        # consider that all tests failed
        result = [[-1 for i in range(len(in_outs["inputs"]))]]
        if debug:
            print(f"global timeout")
    return result[0], metadata_list[0]


def compute_score(completion, test_cases):
    solution = completion.split('```python')[-1].split('```')[0]

    # extract test cases
    try:
        in_outs = json.loads(test_cases)
    except:
        in_outs = json.loads(pickle.loads(zlib.decompress(base64.b64decode(test_cases.encode("utf-8")))))

    success = False
    try:
        res, metadata = check_correctness(in_outs=in_outs, generation=solution, timeout=6, debug=False)
        success = all(map(lambda x: x == True, res))
    except Exception as e:
        pass

    return success
