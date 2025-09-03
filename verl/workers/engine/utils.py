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


import torch

from verl import DataProto
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import restore_dynamic_batch


def postprocess_batch_func(output_lst, indices, data: DataProto):
    """postprocess the output of a forward_backward_batch.
    output_lst is a list of dict containing outputs for each micro-batch
    reorder entropy and outputs. Return None for other pp ranks
    only on last rank. It should be on every tp rank

    each losses_reduced contains 1. model_output, 2. loss, 3. metrics.
    """

    use_dynamic_bsz = data.meta_info.get("use_dynamic_bsz", True)

    # losses_reduced is a list of dict containing outputs for each micro-batch
    # reorder entropy and outputs. Return None for other pp ranks
    # only on last rank. It should be on every tp rank

    # losses_reduced contains 1. model_output, 2. loss, 3. metrics.
    # We perform reverse

    model_output = {}
    losses = []
    aggregated_metrics = {}

    # model output
    for o in output_lst:
        if "model_output" in o:
            for key, val in o["model_output"].items():
                if key not in model_output:
                    model_output[key] = []
                model_output[key].append(val)

    # concat results from micro batches
    for key, val in model_output.items():
        model_output[key] = torch.cat(model_output[key], dim=0)
        # reverse with dynamic bsz
        if use_dynamic_bsz:
            model_output[key] = restore_dynamic_batch(model_output[key], indices)

    # loss
    for o in output_lst:
        if "loss" in o:
            losses.append(o["loss"])

    # metrics
    for o in output_lst:
        if "metrics" in o:
            metrics = o["metrics"]
            append_to_dict(aggregated_metrics, metrics)

    output = {
        "model_output": model_output,
        "loss": losses,
        "metrics": aggregated_metrics,
    }

    return output
