# Copyright (c) OpenMMLab. All rights reserved.
# flake8: noqa
from .backend_manager import TensorRTManager
from .init_plugins import load_tensorrt_plugin

_BackendManager = TensorRTManager
is_available = _BackendManager.is_available
build_wrapper = _BackendManager.build_wrapper

__all__ = ['TensorRTManager']

if is_available():
    from .utils import from_onnx, load, save

    __all__ += ['from_onnx', 'save', 'load', 'load_tensorrt_plugin']

    try:
        # import wrapper if pytorch is available
        from .torch_allocator import TorchAllocator
        from .wrapper import TRTWrapper
        __all__ += ['TRTWrapper']
        __all__ += ['TorchAllocator', 'TRTWrapper']
    except Exception:
        pass
