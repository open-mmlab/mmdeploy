# Copyright (c) OpenMMLab. All rights reserved.
from .backend_manager import SDKManager, SDKParam

_BackendManager = SDKManager
is_available = _BackendManager.is_available
build_wrapper = _BackendManager.build_wrapper
build_wrapper_from_param = _BackendManager.build_wrapper_from_param

__all__ = ['SDKParam', 'SDKManager']
