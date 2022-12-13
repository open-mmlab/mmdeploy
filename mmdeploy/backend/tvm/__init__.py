# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import sys


def is_available() -> bool:
    """Check whether tvm package is installed.

    Returns:
        bool: True if tvm package is installed.
    """

    return importlib.util.find_spec('tvm') is not None


def get_library_ext() -> str:
    """Get the extension of the library.

    Returns:
        str: The extension name
    """
    platform = sys.platform.lower()
    if platform == 'win32' or platform == 'cygwin':
        return '.dll'
    elif platform == 'linux' or platform == 'darwin' or platform == 'freebsd':
        return '.so'


if is_available():
    from .onnx2tvm import from_onnx
    from .quantize import HDF5Dataset
    from .tuner import build_tvm_tuner

    __all__ = ['from_onnx', 'build_tvm_tuner', 'HDF5Dataset', 'TVMManager']

    try:
        # import wrapper if pytorch is available
        from .wrapper import TVMWrapper
        __all__ += ['TVMWrapper']
    except Exception:
        pass
