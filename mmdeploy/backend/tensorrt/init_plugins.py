# Copyright (c) OpenMMLab. All rights reserved.
import ctypes
import glob
import os

from mmdeploy.utils import get_root_logger


def get_ops_path() -> str:
    """Get path of the TensorRT plugin library.

    Returns:
        str: A path of the TensorRT plugin library.
    """
    wildcard = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '../../../build/lib/libmmdeploy_tensorrt_ops.so'))

    paths = glob.glob(wildcard)
    lib_path = paths[0] if len(paths) > 0 else ''
    return lib_path


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
        logger.warning(f'Could not load the library of tensorrt plugins. \
            Because the file does not exist: {lib_path}')
    return success
