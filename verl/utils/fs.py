#!/usr/bin/env python
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

# -*- coding: utf-8 -*-
"""File-system agnostic IO APIs"""

import hashlib
import os
import tempfile

from typing import Optional

try:
    from hdfs_io import copy, exists, makedirs  # for internal use only
except ImportError:
    from .hdfs_io import copy, makedirs, exists

from verl.utils.s3_io import bulk_download, file_download, file_upload, parse_uri, s3_key_exists

__all__ = ["copy", "exists", "makedirs"]

_HDFS_PREFIX = "hdfs://"
_S3_PREFIX = "s3://"


def is_non_local(path):
    return path.startswith(_HDFS_PREFIX) or path.startswith(_S3_PREFIX)


def md5_encode(path: str) -> str:
    return hashlib.md5(path.encode()).hexdigest()


def get_local_temp_path(hdfs_path: str, cache_dir: str) -> str:
    """Return a local temp path that joins cache_dir and basename of hdfs_path

    Args:
        hdfs_path:
        cache_dir:

    Returns:

    """
    # make a base64 encoding of hdfs_path to avoid directory conflict
    encoded_hdfs_path = md5_encode(hdfs_path)
    temp_dir = os.path.join(cache_dir, encoded_hdfs_path)
    os.makedirs(temp_dir, exist_ok=True)
    dst = os.path.join(temp_dir, os.path.basename(hdfs_path))
    return dst


def copy_to_local(src: str, cache_dir=None, filelock='.file.lock', verbose=False, recursive: Optional[bool]=None) -> str:
    """Copy src from hdfs to local if src is on hdfs or directly return src.
    If cache_dir is None, we will use the default cache dir of the system. Note that this may cause conflicts if
    the src name is the same between calls

    Args:
        src (str): a HDFS path of a local path

    Returns:
        a local path of the copied file
    """
    if src.startswith(_HDFS_PREFIX) or src.startswith(_S3_PREFIX):
        return copy_local_path_from_remote(src, cache_dir, filelock, verbose, recursive=recursive)

    return src


def copy_local_path_from_remote(src: str, cache_dir=None, filelock='.file.lock', verbose=False, recursive: Optional[bool]=None) -> str:
    """
    Used to download files from a remote source (S3 or HDFS).

    Deprecated. Please use copy_to_local which calls this function.
    """
    from filelock import FileLock

    assert src[-1] != "/", f"Make sure the last char in src is not / because it will cause error. Got {src}"

    if is_non_local(src):
        # download from hdfs to local
        if cache_dir is None:
            # get a temp folder
            cache_dir = tempfile.gettempdir()
        os.makedirs(cache_dir, exist_ok=True)

        assert os.path.exists(cache_dir)
        local_path = get_local_temp_path(src, cache_dir)
        # get a specific lock
        filelock = md5_encode(src) + ".lock"
        lock_file = os.path.join(cache_dir, filelock)
        with FileLock(lock_file=lock_file):
            if not os.path.exists(local_path):
                if verbose:
                    print(f'Copy from {src} to {local_path}')
                if src.startswith(_S3_PREFIX):
                    bucket, key_or_prefix, recursive = parse_uri(src, is_dir=recursive)

                    if recursive:
                        bulk_download(bucket, key_or_prefix, local_path)
                    else:
                        file_download(bucket, key_or_prefix, local_path)
                else:
                    copy(src, local_path)
        return local_path
    else:
        return src


def upload_local_file_to_s3(s3_path: str, local_path: str, cache_dir=None, filelock='.file.lock', verbose=False) -> None:
    from filelock import FileLock

    assert s3_path[-1] != '/', f'Make sure the last char in s3_path is not / because it will cause error. Got {s3_path}'
    assert s3_path.startswith(_S3_PREFIX), f'Path must be an s3 path with the s3:// prefix instead got: {s3_path}'
    assert  os.path.exists(local_path), f'Local copy path not found'

    filelock = md5_encode(s3_path) + '.lock'
    lock_file = os.path.join(os.path.dirname(local_path), filelock)
    with FileLock(lock_file=lock_file):
        bucket, key, _ = parse_uri(s3_path, is_dir=False)
        if not s3_key_exists(bucket, key):
            if verbose:
                print(f'Copy from {local_path} to {s3_path}')

            file_upload(bucket, local_path, dest_path=key)
        else:
            print(f"File {s3_path} already exists in S3, not uploading.")
