# Copyright (c) OpenMMLab. All rights reserved.
from .backend_manager import RKNNManager

_BackendManager = RKNNManager
is_available = _BackendManager.is_available
build_wrapper = _BackendManager.build_wrapper

__all__ = ['RKNNManager']

if is_available():
    from .wrapper import RKNNWrapper
    __all__ += ['RKNNWrapper']
