# Copyright (c) OpenMMLab. All rights reserved.
from .backend_manager import PPLNNManager

_BackendManager = PPLNNManager
is_available = _BackendManager.is_available
build_wrapper = _BackendManager.build_wrapper

__all__ = ['PPLNNManager']

if is_available():
    from .utils import register_engines
    from .wrapper import PPLNNWrapper
    __all__ += ['PPLNNWrapper', 'register_engines']
