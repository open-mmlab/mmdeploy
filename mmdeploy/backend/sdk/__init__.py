# Copyright (c) OpenMMLab. All rights reserved.
from .backend_manager import SDKManager

_BackendManager = SDKManager
is_available = _BackendManager.is_available
build_wrapper = _BackendManager.build_wrapper

__all__ = ['SDKManager']

if is_available():

    try:
        from .wrapper import SDKWrapper
        __all__ += ['SDKWrapper']
    except Exception:
        pass
