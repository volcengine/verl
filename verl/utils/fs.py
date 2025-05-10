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
import os
import shutil
import tempfile
import hashlib

from verl.utils.debug import log_print

try:
    from hdfs_io import copy, makedirs, exists  # for internal use only
except ImportError:
    from .hdfs_io import copy, makedirs, exists

__all__ = ["copy", "exists", "makedirs"]

_HDFS_PREFIX = "hdfs://"


def is_non_local(path):
    return path.startswith(_HDFS_PREFIX)


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


def copy_to_shm(src:str):
    """
        Load the model into   /dev/shm   to make the process of loading the model multiple times more efficient.
    """
    shm_model_root = '/dev/shm/verl-cache/'
    src_abs = os.path.abspath(os.path.normpath(src))
    dest = os.path.join(shm_model_root, hashlib.md5(src_abs.encode('utf-8')).hexdigest())
    os.makedirs(dest, exist_ok=True)
    dest = os.path.join(dest, os.path.basename(src_abs))
    if os.path.exists(dest):
        # inform user and depends on him
        log_print(f"[WARNING]: The memory model path {dest} already exists. If it is not you want, please clear it and restart the task.")
    else:
        log_print(f'Load from disk {src} into memory {dest}')
        if os.path.isdir(src):
            shutil.copytree(src, dest, symlinks=False, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dest)
    return dest

def copy_to_local(src: str, cache_dir=None, filelock='.file.lock', verbose=False, use_shm:bool = False) -> str:
    """Copy src from hdfs to local if src is on hdfs or directly return src.
    If cache_dir is None, we will use the default cache dir of the system. Note that this may cause conflicts if
    the src name is the same between calls

    Args:
        src (str): a HDFS path of a local path

    Returns:
        a local path of the copied file
    """
    # Save to a local path for persistence.
    local_path = copy_local_path_from_hdfs(src, cache_dir, filelock, verbose)
    # Load into shm to improve efficiency.
    if use_shm:
        return copy_to_shm(local_path)
    return local_path

def copy_local_path_from_hdfs(src: str, cache_dir=None, filelock='.file.lock', verbose=False) -> str:
    """Deprecated. Please use copy_to_local instead."""
    from filelock import FileLock

    assert src[-1] != '/', f'Make sure the last char in src is not / because it will cause error. Got {src}'

    if is_non_local(src):
        # download from hdfs to local
        if cache_dir is None:
            # get a temp folder
            cache_dir = tempfile.gettempdir()
        os.makedirs(cache_dir, exist_ok=True)
        assert os.path.exists(cache_dir)
        local_path = get_local_temp_path(src, cache_dir)
        # get a specific lock
        filelock = md5_encode(src) + '.lock'
        lock_file = os.path.join(cache_dir, filelock)
        with FileLock(lock_file=lock_file):
            if not os.path.exists(local_path):
                if verbose:
                    print(f'Copy from {src} to {local_path}')
                copy(src, local_path)
        return local_path
    else:
        return src
