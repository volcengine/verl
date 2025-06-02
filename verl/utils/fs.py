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
import shutil
import tempfile
from typing import Optional
try:
    from hdfs_io import copy, exists, makedirs  # for internal use only
except ImportError:
    from .hdfs_io import copy, exists, makedirs

__all__ = ["copy", "exists", "makedirs"]

_HDFS_PREFIX = "hdfs://"
_S3_PREFIX = "s3://"

def is_non_local(path):
    return path.startswith(_HDFS_PREFIX) or path.startswith(_S3_PREFIX)

def md5_encode(path: str) -> str:
    return hashlib.md5(path.encode()).hexdigest()


def get_local_temp_path(hdfs_path: str, cache_dir: str) -> str:
    """Generate a unique local cache path for an HDFS resource.
    Creates a MD5-hashed subdirectory in cache_dir to avoid name conflicts,
    then returns path combining this subdirectory with the HDFS basename.

    Args:
        hdfs_path (str): Source HDFS path to be cached
        cache_dir (str): Local directory for storing cached files

    Returns:
        str: Absolute local filesystem path in format:
            {cache_dir}/{md5(hdfs_path)}/{basename(hdfs_path)}
    """
    # make a base64 encoding of hdfs_path to avoid directory conflict
    encoded_hdfs_path = md5_encode(hdfs_path)
    temp_dir = os.path.join(cache_dir, encoded_hdfs_path)
    os.makedirs(temp_dir, exist_ok=True)
    dst = os.path.join(temp_dir, os.path.basename(hdfs_path))
    return dst


def _record_directory_structure(folder_path):
    record_file = os.path.join(folder_path, ".directory_record.txt")
    with open(record_file, "w") as f:
        for root, dirs, files in os.walk(folder_path):
            for dir_name in dirs:
                relative_dir = os.path.relpath(os.path.join(root, dir_name), folder_path)
                f.write(f"dir:{relative_dir}\n")
            for file_name in files:
                if file_name != ".directory_record.txt":
                    relative_file = os.path.relpath(os.path.join(root, file_name), folder_path)
                    f.write(f"file:{relative_file}\n")
    return record_file


def _check_directory_structure(folder_path, record_file):
    if not os.path.exists(record_file):
        return False
    existing_entries = set()
    for root, dirs, files in os.walk(folder_path):
        for dir_name in dirs:
            relative_dir = os.path.relpath(os.path.join(root, dir_name), folder_path)
            existing_entries.add(f"dir:{relative_dir}")
        for file_name in files:
            if file_name != ".directory_record.txt":
                relative_file = os.path.relpath(os.path.join(root, file_name), folder_path)
                existing_entries.add(f"file:{relative_file}")
    with open(record_file) as f:
        recorded_entries = set(f.read().splitlines())
    return existing_entries == recorded_entries


def copy_to_local(src: str, cache_dir=None, filelock='.file.lock', verbose=False, recursive: Optional[bool]=None, always_recopy: bool=False) -> str:
    """Copy files/directories from HDFS to local cache with validation.

    Args:
        src (str): Source path - HDFS path (hdfs://...) or local filesystem path
        cache_dir (str, optional): Local directory for cached files. Uses system tempdir if None
        filelock (str): Base name for file lock. Defaults to ".file.lock"
        verbose (bool): Enable copy operation logging. Defaults to False
        recursive (bool, optional): Whether to recursively copy directories. Should be set to True for directory paths. Defaults to None
        always_recopy (bool): Force fresh copy ignoring cache. Defaults to False

    Returns:
        str: Local filesystem path to copied resource
    """
    if src.startswith(_HDFS_PREFIX) or src.startswith(_S3_PREFIX):
        return copy_local_path_from_remote(src, cache_dir, filelock, verbose, recursive=recursive)

    return src


def copy_local_path_from_remote(src: str, cache_dir=None, filelock='.file.lock', verbose=False, recursive: Optional[bool]=None, always_recopy: bool=False) -> str:
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
            if always_recopy and os.path.exists(local_path):
                if os.path.isdir(local_path):
                    shutil.rmtree(local_path, ignore_errors=True)
                else:
                    os.remove(local_path)
            if not os.path.exists(local_path):
                if verbose:
                    print(f'Copy from {src} to {local_path}')
                if src.startswith(_S3_PREFIX):
                    from verl.utils.s3_io import bulk_download, file_download, parse_uri
                    bucket, key_or_prefix, recursive = parse_uri(src, is_dir=recursive)

                    if recursive:
                        bulk_download(bucket, key_or_prefix, local_path)
                    else:
                        file_download(bucket, key_or_prefix, local_path)
                else:
                    copy(src, local_path)
            elif os.path.isdir(local_path):
                # always_recopy=False, local path exists, and it is a folder: check whether there is anything missed
                record_file = os.path.join(local_path, ".directory_record.txt")
                if not _check_directory_structure(local_path, record_file):
                    if verbose:
                        print(f"Recopy from {src} to {local_path} due to missing files or directories.")
                    shutil.rmtree(local_path, ignore_errors=True)
                    copy(src, local_path)
                    _record_directory_structure(local_path)
        return local_path
    else:
        return src


def upload_local_file_to_s3(s3_path: str, local_path: str, cache_dir=None, filelock='.file.lock', verbose=False) -> None:
    from filelock import FileLock

    assert s3_path[-1] != '/', f'Make sure the last char in s3_path is not / because it will cause error. Got {s3_path}'
    assert s3_path.startswith(_S3_PREFIX), f'Path must be an s3 path with the s3:// prefix instead got: {s3_path}'
    assert  os.path.exists(local_path), f'Local copy path not found'

    from verl.utils.s3_io import file_upload, parse_uri, s3_key_exists

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