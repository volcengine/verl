# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""Import hook for VERL_USE_EXTERNAL_MODULES: registers the Step5 text training model.

Every process that imports verl (driver, Ray task runner and workers) must resolve the
``step5`` config type before HFModelConfig is built.
"""

from .modeling_step5 import register_step5_training

register_step5_training()
