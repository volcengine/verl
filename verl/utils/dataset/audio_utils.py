# Copyright 2025 Individual Contributor: TomQunChaoA
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
import io

import librosa
import numpy as np
from qwen_omni_utils import process_audio_info
from qwen_omni_utils.v2_5.audio_process import SAMPLE_RATE


def process_audio(mm: np.ndarray | list[float] | dict[str] | str, is_video: bool):
    """
    format of mm:
    - np.ndarray
    - list[float]
    - pathlike str: http, file..., reference to qwen_omni_utils
    """
    data_type, key = ("video", "video") if is_video else ("audio", "audio")
    if type(mm) is list and len(mm) > 0 and type(mm[0]) is float:
        mm = np.array(mm)
    elif type(mm) is dict and len(mm) > 0 and "bytes" in mm:
        sampling_rate = mm.get("sampling_rate", SAMPLE_RATE)
        audio_content = librosa.load(io.BytesIO(mm["bytes"]), sr=sampling_rate, mono=True)[0]
        mm = librosa.resample(audio_content, sampling_rate, SAMPLE_RATE)
    messages = [{"role": "user", "content": [{"type": data_type, key: mm}]}]
    return process_audio_info(messages, True)[0]
