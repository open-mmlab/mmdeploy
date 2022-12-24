# Copyright (c) OpenMMLab. All rights reserved.
from .backend_manager import NCNNManager
from .onnx2ncnn import from_onnx

_BackendManager = NCNNManager

is_available = _BackendManager.is_available
build_wrapper = _BackendManager.build_wrapper

__all__ = ['NCNNManager', 'from_onnx']

if is_available():
    try:
        from .wrapper import NCNNWrapper

        __all__ += ['NCNNWrapper']
    except Exception:
        pass
