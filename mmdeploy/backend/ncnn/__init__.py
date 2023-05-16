# Copyright (c) OpenMMLab. All rights reserved.
import os
from ctypes import cdll

from .backend_manager import NCNNManager
from .init_plugins import get_ops_path
from .onnx2ncnn import from_onnx

_BackendManager = NCNNManager

is_available = _BackendManager.is_available
build_wrapper = _BackendManager.build_wrapper
# load mmdeploy_ncnn_ops.dll before import ncnn_ext
ops_path = get_ops_path()
if os.path.exists(ops_path):
    cdll.LoadLibrary(ops_path)

__all__ = ['NCNNManager', 'from_onnx']

if is_available():
    try:
        from .wrapper import NCNNWrapper

        __all__ += ['NCNNWrapper']
    except Exception:
        pass
