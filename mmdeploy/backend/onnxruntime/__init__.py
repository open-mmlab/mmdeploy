# Copyright (c) OpenMMLab. All rights reserved.
from .backend_manager import ONNXRuntimeBackendParam, ONNXRuntimeManager

_BackendManager = ONNXRuntimeManager

is_available = _BackendManager.is_available
build_wrapper = _BackendManager.build_wrapper

__all__ = ['ONNXRuntimeBackendParam', 'ONNXRuntimeManager']
