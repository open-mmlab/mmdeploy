# Copyright (c) OpenMMLab. All rights reserved.
# flake8: noqa
from .backend_manager import TorchScriptManager
from .init_plugins import get_ops_path, ops_available

_BackendManager = TorchScriptManager
is_available = _BackendManager.is_available
build_wrapper = _BackendManager.build_wrapper
build_wrapper_from_param = _BackendManager.build_wrapper_from_param
to_backend = _BackendManager.to_backend
to_backend_from_param = _BackendManager.to_backend_from_param

__all__ = ['get_ops_path', 'ops_available', 'TorchScriptManager']
