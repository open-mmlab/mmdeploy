# Copyright (c) OpenMMLab. All rights reserved.
from .backend_manager import ONNXRuntimeManager

_BackendManager = ONNXRuntimeManager

is_available = _BackendManager.is_available
build_wrapper = _BackendManager.build_wrapper

__all__ = ['ONNXRuntimeManager']

if is_available():
    try:
        # import wrapper if pytorch is available
        from .wrapper import ORTWrapper
        __all__ += ['ORTWrapper']
    except Exception:
        pass
