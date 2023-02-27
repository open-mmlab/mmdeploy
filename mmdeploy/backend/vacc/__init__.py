# Copyright (c) OpenMMLab. All rights reserved.
from .backend_manager import VACCManager
from .onnx2vacc import from_onnx

_BackendManager = VACCManager

is_available = _BackendManager.is_available
build_wrapper = _BackendManager.build_wrapper

__all__ = ['VACCManager', 'from_onnx']

if is_available():
    try:
        from .wrapper import VACCWrapper

        __all__ += ['VACCWrapper']
    except Exception:
        pass
