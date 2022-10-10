# Copyright (c) OpenMMLab. All rights reserved.
import sys

from mmdeploy.backend.tvm import is_available
from ..core import PIPELINE_MANAGER


def get_library_ext() -> str:
    platform = sys.platform.lower()
    if platform == 'win32' or platform == 'cygwin':
        return '.dll'
    elif platform == 'linux' or platform == 'darwin' or platform == 'freebsd':
        return '.so'


__all__ = ['is_available', 'get_library_ext']

if is_available():
    from mmdeploy.backend.tvm import from_onnx as _from_onnx
    from_onnx = PIPELINE_MANAGER.register_pipeline()(_from_onnx)

    __all__ += ['from_onnx']
