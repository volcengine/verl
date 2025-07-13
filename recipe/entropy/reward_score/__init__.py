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
# from . import gsm8k, math, prime_math, prime_code

import traceback

from . import entropy_math


def _default_compute_score(
    data_source, solution_str, ground_truth, extra_info=None, sandbox_fusion_url=None, concurrent_semaphore=None
):
    try:
        res = entropy_math.compute_score(solution_str, str(ground_truth))
        # print(f"data_source: {data_source}")
        # raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

        if isinstance(res, dict):
            return res
        elif isinstance(res, int | float | bool):
            return float(res)
        else:
            return float(res[0])
    except Exception as e:
        print(f"[ERROR] Error in process_completion for task : {str(e)}")
        traceback.print_exc()  # 打印完整堆栈
        raise  # 重新抛出异常以便上层捕获
