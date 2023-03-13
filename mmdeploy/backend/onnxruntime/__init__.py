# Copyright (c) OpenMMLab. All rights reserved.
from .backend_manager import ONNXRuntimeManager, ONNXRuntimeParam

_BackendManager = ONNXRuntimeManager

is_available = _BackendManager.is_available
build_wrapper = _BackendManager.build_wrapper
build_wrapper_from_param = _BackendManager.build_wrapper_from_param
to_backend = _BackendManager.to_backend
to_backend_from_param = _BackendManager.to_backend_from_param

__all__ = ['ONNXRuntimeParam', 'ONNXRuntimeManager']

if is_available():
    from .wrapper import ORTWrapper

    __all__ += ['ORTWrapper']
