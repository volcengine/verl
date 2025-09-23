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
"""
Preprocess the Geometry3k dataset to parquet format
"""

import argparse
import os
import numpy as np
import datasets
import pyarrow as pa
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image
import io
from verl.utils.hdfs_io import copy, makedirs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None)
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="~/data/mnist_multiturn_sft", help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path
    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_save_dir

    data_source = "vermouth1992/mnist_multiturn_sft"

    if local_dataset_path is not None:
        dataset = datasets.load_dataset(
            local_dataset_path,
        )
    else:
        dataset = datasets.load_dataset(
            data_source,
        )

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # def image_to_bytes(image: Image.Image) -> bytes:
    #     img_byte_arr = io.BytesIO()
    #     image.save(img_byte_arr)
    #     return img_byte_arr.getvalue()
    
    # def process_row(row):
    #     messages = row['messages']
    #     messages_new = []
    #     for message in messages:
    #         for idx, content in enumerate(message['content']):
    #             if content['type'] == 'image':
    #                 content = {"image": int(content['image']), "type": "image"}
    #             if content['type'] == 'text':
    #                 content = {"text": content['text'], "type": "text"}
    #             message['content'][idx] = content
    #         messages_new.append(message)
    #     row['messages'] = messages_new
    #     return row

    print(train_dataset[0])
    train_dataset.to_parquet(os.path.join(args.local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(args.local_save_dir, "test.parquet"))

    # if hdfs_dir is not None:
    #     makedirs(hdfs_dir)
    #     copy(src=local_save_dir, dst=hdfs_dir)


    # df_loaded = pd.read_parquet(os.path.join(local_save_dir, "test.parquet"))
    # messages = df_loaded['messages'][0]
    # # print("is numpy.ndarray?:", isinstance(messages, np.ndarray))
    # print(messages)
