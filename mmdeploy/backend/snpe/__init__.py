# Copyright (c) OpenMMLab. All rights reserved.
from .backend_manager import SNPEManager
from .onnx2dlc import from_onnx

_BackendManager = SNPEManager
is_available = _BackendManager.is_available
build_wrapper = _BackendManager.build_wrapper

__all__ = ['from_onnx', 'SNPEManager']

if is_available():
    try:
        from .wrapper import SNPEWrapper

        __all__ += ['SNPEWrapper']
    except Exception as e:
        print(e)
        pass
