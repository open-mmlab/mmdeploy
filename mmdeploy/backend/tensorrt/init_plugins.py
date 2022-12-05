# Copyright (c) OpenMMLab. All rights reserved.
import ctypes
import os

from mmdeploy.utils import get_file_path, get_root_logger


def get_ops_path() -> str:
    """Get path of the TensorRT plugin library.

    Returns:
        str: A path of the TensorRT plugin library.
    """
    candidates = [
        '../../lib/libmmdeploy_tensorrt_ops.so',
        '../../lib/mmdeploy_tensorrt_ops.dll',
        '../../../build/lib/libmmdeploy_tensorrt_ops.so',
        '../../../build/bin/*/mmdeploy_tensorrt_ops.dll'
    ]
    return get_file_path(os.path.dirname(__file__), candidates)


def load_tensorrt_plugin() -> bool:
    """Load TensorRT plugins library.

    Returns:
        bool: True if TensorRT plugin library is successfully loaded.
    """
    lib_path = get_ops_path()
    success = False
    logger = get_root_logger()
    if os.path.exists(lib_path):
        ctypes.CDLL(lib_path)
        logger.info(f'Successfully loaded tensorrt plugins from {lib_path}')
        success = True
    else:
        logger.warning(f'Could not load the library of tensorrt plugins.'
                       f'Because the file does not exist: {lib_path}')
    return success
