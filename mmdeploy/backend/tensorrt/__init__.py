# Copyright (c) OpenMMLab. All rights reserved.
# flake8: noqa
from .backend_manager import TensorRTBackendParam, TensorRTManager
from .init_plugins import load_tensorrt_plugin

_BackendManager = TensorRTManager
is_available = _BackendManager.is_available
build_wrapper = _BackendManager.build_wrapper

__all__ = ['load_tensorrt_plugin', 'TensorRTManager', 'TensorRTBackendParam']
