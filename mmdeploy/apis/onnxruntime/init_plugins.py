import ctypes
import glob
import logging
import os


def get_ops_path():
    """Get TensorRT plugins library path."""
    wildcard = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '../../../build/lib/libmmlab_onnxruntime_ops.so'))

    paths = glob.glob(wildcard)
    lib_path = paths[0] if len(paths) > 0 else ''
    return lib_path


def load_tensorrt_plugin():
    """load TensorRT plugins library."""
    lib_path = get_ops_path()
    if os.path.exists(lib_path):
        ctypes.CDLL(lib_path)
        return 0
    else:
        logging.warning('Can not load tensorrt custom ops.')
        return -1
