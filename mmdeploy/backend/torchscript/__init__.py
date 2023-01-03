# Copyright (c) OpenMMLab. All rights reserved.
# flake8: noqa
from .backend_manager import TorchScriptManager
from .init_plugins import get_ops_path, ops_available

_BackendManager = TorchScriptManager
is_available = _BackendManager.is_available
build_wrapper = _BackendManager.build_wrapper

__all__ = ['get_ops_path', 'ops_available', 'TorchScriptManager']

if is_available():
    from .wrapper import TorchscriptWrapper

    __all__ += ['TorchscriptWrapper']
