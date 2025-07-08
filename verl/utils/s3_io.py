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
"""Utility functions for downloading files from S3 cloud storage.
Copied directly from AGIEmergeMegatronLM/coreutils/s3_buckets.py
"""

import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import boto3
except ImportError as excn:
    raise RuntimeError("Boto3 required for s3 checkpointing, please install boto3==1.34.162") from excn

import botocore
from botocore.exceptions import ClientError
from pydantic import BaseModel

PARALLEL_UPLOAD_COUNT = 100

TRIES = 8


class CLIException(Exception):
    pass


class FilenameInfo(BaseModel):  # type: ignore
    class Config:
        frozen = True

    name: str
    """Name of the file (relative to the obj store bucket)"""
    size: int
    """Size of the file"""
    md5: str
    """base64 md5 of the file"""


def _get_s3_client() -> Any:
    s3_session = boto3.Session()
    s3 = s3_session.client("s3")
    return s3


def _normalize_path(path: str) -> str:
    """
    Normalize slashes in path and remove any leading slashes.

    Result will have a trailing slash if the input had one.
    """
    path = re.sub(r"//+", "/", path)
    path = path.lstrip("/")
    return path


def _format_path(path: str, trailing_slash: bool = True) -> str:
    path_components = path.split("/")
    path_components = [comp for comp in path_components if comp]
    object_path = "/".join(path_components) + ("/" if trailing_slash else "")
    return object_path


def _retry_cmd(func: Callable[[], Any]) -> Any:
    tries = 0
    while True:
        try:
            return func()
        except Exception as exc:
            tries += 1
            print(exc)
            if tries == TRIES:
                raise
            else:
                print(f"Retry {tries} in 5 seconds")
                time.sleep(5)


def list_objects(
    bucket: str,
    prefix: str,
    strip_prefix: bool = True,
    limit: int = 1000,
) -> List[str]:
    return [info.name for info in list_filename_info(bucket, prefix, strip_prefix, limit)]


def list_filename_info(
    bucket: str,
    prefix: str,
    strip_prefix: bool = True,
    limit: int = 1000,
) -> List[FilenameInfo]:
    """
    Lists objects in bucket/prefix. It modifies the filenames to strip the prefix.
    """
    client = _get_s3_client()

    prefix = _normalize_path(prefix)

    ret: List[FilenameInfo] = []
    start_after = ""

    while True:  # for pagination

        def list_objects_once() -> Dict[str, Any]:
            response = client.list_objects_v2(Bucket=bucket, Prefix=prefix, StartAfter=start_after, MaxKeys=limit)  # noqa: B023
            response_status = response["ResponseMetadata"]["HTTPStatusCode"]
            assert response_status == 200, f"Listing objects failed with status {response_status}"
            return response  # type: ignore

        response = _retry_cmd(list_objects_once)
        contents: List[Dict[str, Any]] = response.get("Contents", [])

        if strip_prefix:
            # Strip any leading "/".
            remove_prefix = prefix if prefix.endswith("/") else prefix + "/"
        else:
            remove_prefix = ""

        ret.extend(
            [
                FilenameInfo(name=obj["Key"].replace(remove_prefix, ""), size=obj["Size"], md5=obj["ETag"])
                for obj in contents
            ]
        )

        if not response["IsTruncated"]:
            break
        start_after = contents[-1]["Key"]

    return ret


def parse_uri(uri: str, is_dir: Optional[bool] = None) -> Tuple[str, str, bool]:
    assert uri.startswith("s3://"), f"This is not a valid s3 uri: {uri}"
    uri = uri.replace("s3://", "")
    uri_parts = uri.split("/")
    uri_parts = [part for part in uri_parts if part]

    bucket_name, path_parts = uri_parts[0], uri_parts[1:]
    path = "/".join(path_parts)
    if is_dir is None:
        # check if a file exists at that key
        is_dir = not s3_key_exists(bucket_name, path)

    assert path is not None
    if is_dir:
        path += "/"

    return bucket_name, path, is_dir


def s3_key_exists(bucket: str, remote_path: str) -> bool:
    client = _get_s3_client()
    try:
        client.head_object(Bucket=bucket, Key=remote_path)
        return True
    except botocore.exceptions.ClientError as e:
        # If error is a 404, the object doesn't exist, so assume it's not a file.
        if e.response["Error"]["Code"] == "404":
            return False
        raise


def file_download(bucket: str, remote_path: str, local_path: str) -> None:
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    client = _get_s3_client()
    formatted_remote_path = _format_path(remote_path, trailing_slash=False)
    client.download_file(bucket, formatted_remote_path, local_path)


def bulk_download(
    bucket: str,
    prefix: str,
    path: str,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    verbose: bool = False,
    check_uploaded_filename: Optional[str] = None,
) -> None:
    print(f"Downloading from s3://{bucket}/{prefix} to {path}")
    # Remove trailing slash if it exists.
    if prefix.endswith("/"):
        prefix = prefix[:-1]

    cmd = ["s5cmd", "cp"]

    # If include is not None, add all include patterns to the command
    if include is not None:
        for inc in include:
            cmd += ["--include", inc]

    # If exclude is not None, add all exclude patterns to the command
    if exclude is not None:
        for exc in exclude:
            cmd += ["--exclude", exc]

    cmd += [f"s3://{bucket}/{prefix}/*", f"{path}/"]

    if verbose:
        print("Running:", cmd)

    subprocess.run(cmd, check=True)

    if check_uploaded_filename is not None and not os.path.exists(f"{path}/{check_uploaded_filename}"):
        raise ValueError(f"Failing bulk download on incomplete upload: could not find {path}/{check_uploaded_filename}")


def file_upload(bucket: str, src_path: str, dest_path: str) -> None:
    _get_s3_client().upload_file(src_path, bucket, dest_path)


def list_dirs(
    bucket: str,
    prefix: str,
    limit: int = 1000,
    strip_prefix: bool = True,
) -> List[str]:
    """
    Returns the immediate “sub-directories” that live directly under *bucket/prefix*.
    """
    client = _get_s3_client()

    prefix = _normalize_path(prefix)
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    subdirs: List[str] = []
    paginator = client.get_paginator("list_objects_v2")

    for page in paginator.paginate(
        Bucket=bucket,
        Prefix=prefix,
        Delimiter="/",
        PaginationConfig={"MaxItems": limit},
    ):
        for cp in page.get("CommonPrefixes", []):
            name = cp["Prefix"]
            if strip_prefix:
                name = name[len(prefix) :]
            subdirs.append(name.rstrip("/"))

    return subdirs


def object_exists(bucket: str, key: str) -> bool:
    client = _get_s3_client()
    try:
        client.head_object(Bucket=bucket, Key=key)  # Lightweight “HEAD” request
        return True  # 200 → object found
    except ClientError as e:
        # Missing objects raise a 404 / NoSuchKey
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return False
        raise


def find_remote_auto_checkpoint(checkpoint_folder: str):
    bucket, prefix, _ = parse_uri(checkpoint_folder, is_dir=True)
    step_folders = list_dirs(bucket, prefix)
    # loop through step folder and remove the ones without data.pt
    step_folders = [step for step in step_folders if object_exists(bucket, os.path.join(prefix, step, "data.pt"))]
    if not step_folders:
        global_step_folder = None
    else:
        global_step = max(step_folders, key=lambda s: int(re.search(r"\d+$", s).group()))
        global_step_folder = os.path.join(checkpoint_folder, global_step)

    return global_step_folder
