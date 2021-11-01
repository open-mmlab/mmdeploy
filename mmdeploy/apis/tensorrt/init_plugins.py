import ctypes
import glob
import logging
import os


def get_ops_path():
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


def load_tensorrt_plugin():
    """Load TensorRT plugins library.

    Returns:
        bool: True if TensorRT plugin library is successfully loaded.
    """
    lib_path = get_ops_path()
    success = False
    if os.path.exists(lib_path):
        ctypes.CDLL(lib_path)
        logging.info(f'Successfully loaded tensorrt plugins from {lib_path}')
        success = True
    else:
        logging.warning(f'Could not load the library of tensorrt plugins. \
            Because the file does not exist: {lib_path}')
    return success
