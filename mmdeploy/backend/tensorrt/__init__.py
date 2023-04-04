# Copyright (c) OpenMMLab. All rights reserved.
# flake8: noqa
from .backend_manager import TensorRTManager, TensorRTParam
from .init_plugins import load_tensorrt_plugin

_BackendManager = TensorRTManager
is_available = _BackendManager.is_available
build_wrapper = _BackendManager.build_wrapper
build_wrapper_from_param = _BackendManager.build_wrapper_from_param
to_backend = _BackendManager.to_backend
to_backend_from_param = _BackendManager.to_backend_from_param

__all__ = ['load_tensorrt_plugin', 'TensorRTManager', 'TensorRTParam']
