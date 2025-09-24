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
Preprocess the Geometry3k dataset to SFT format with multimodal support
"""

import argparse
import base64
import os
from io import BytesIO

import datasets

from verl.utils.hdfs_io import copy, makedirs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None)
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="~/data/geo3k_sft", help="The save directory for the preprocessed dataset."
    )
    parser.add_argument(
        "--image_format", default="PNG", choices=["PNG", "JPEG", "WEBP"], help="Format to save images as base64."
    )
    parser.add_argument("--jpeg_quality", type=int, default=95, help="JPEG quality (1-100) when using JPEG format.")
    parser.add_argument(
        "--use_data_uri",
        action="store_true",
        help="Include data URI prefix (data:image/xxx;base64,) in base64 strings.",
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    data_source = "hiyouga/geometry3k"

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

    instruction_following = (
        r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
        r"The reasoning process MUST BE enclosed within <think> </think> tags. "
        r"The final answer MUST BE put in \boxed{}."
    )

    def pil_image_to_base64(pil_image, format="PNG", quality=95, include_data_uri=False):
        """将PIL Image转换为base64字符串"""
        buffer = BytesIO()

        if format.upper() == "JPEG":
            # JPEG不支持透明度，需要转换为RGB
            if pil_image.mode in ("RGBA", "LA", "P"):
                # 创建白色背景
                background = pil_image.convert("RGB") if pil_image.mode == "P" else pil_image
                if pil_image.mode == "RGBA":
                    background = pil_image.convert("RGB")
                pil_image = background
            pil_image.save(buffer, format="JPEG", quality=quality)
        else:
            pil_image.save(buffer, format=format)

        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode("utf-8")

        if include_data_uri:
            # 添加 Data URI 前缀
            mime_type = f"image/{format.lower()}"
            return f"data:{mime_type};base64,{image_base64}"
        else:
            return image_base64

    def images_to_base64_array(images, format="PNG", quality=95, include_data_uri=False):
        """将PIL Image列表转换为base64字符串数组"""
        return [pil_image_to_base64(img, format, quality, include_data_uri) for img in images]

    # Convert to SFT format with multimodal support
    def make_map_fn(split):
        def process_fn(example, idx):
            problem = example.pop("problem")
            prompt = problem + " " + instruction_following
            answer = example.pop("answer")
            images = example.pop("images")

            # Convert PIL images to base64 array
            images_base64 = images_to_base64_array(
                images, format=args.image_format, quality=args.jpeg_quality, include_data_uri=args.use_data_uri
            )

            # Create SFT format with messages structure
            data = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": images_base64[0]},  # 存储base64数组
                            {"type": "text", "text": prompt},
                        ],
                    },
                    {"role": "assistant", "content": [{"type": "text", "text": answer}]},
                ],
            }
            return data

        return process_fn

    print(f"Converting images to base64 format: {args.image_format}")
    if args.image_format == "JPEG":
        print(f"JPEG quality: {args.jpeg_quality}")
    print(f"Using Data URI format: {args.use_data_uri}")

    print("Processing train dataset...")
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=8)
    print("Processing test dataset...")
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True, num_proc=8)

    hdfs_dir = args.hdfs_dir

    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    local_save_dir = os.path.expanduser(local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    print("Saving datasets to parquet format...")
    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))

    print(f"Datasets saved to {local_save_dir}")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Image format: {args.image_format}")

    # 显示一个示例
    print("\nExample data structure:")
    print(f"First example: {train_dataset[0]['messages'][0]['content'][0]['type']}")
    print(
        f"Base64 string length: {len(train_dataset[0]['messages'][0]['content'][0]['image'][0])}"
    )  # 第一个图像的base64长度

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_save_dir, dst=hdfs_dir)
