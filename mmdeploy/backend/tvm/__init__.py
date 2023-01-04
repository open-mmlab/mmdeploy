# Copyright (c) OpenMMLab. All rights reserved.
import sys

from .backend_manager import TVMManager

_BackendManager = TVMManager
is_available = _BackendManager.is_available
build_wrapper = _BackendManager.build_wrapper


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


__all__ = ['TVMManager']

if is_available():
    from .onnx2tvm import from_onnx
    from .quantize import HDF5Dataset
    from .tuner import build_tvm_tuner

    __all__ += ['from_onnx', 'build_tvm_tuner', 'HDF5Dataset']

    try:
        # import wrapper if pytorch is available
        from .wrapper import TVMWrapper
        __all__ += ['TVMWrapper']
    except Exception:
        pass
