# Copyright (c) OpenMMLab. All rights reserved.
from .backend_manager import IPUManager

_BackendManager = IPUManager

is_available = _BackendManager.is_available

__all__ = ['IPUManager']

if is_available():
    try:
        from .converter import onnx_to_popef
        from .wrapper import IPUWrapper
        __all__ += ['IPUWrapper', 'onnx_to_popef']
    except Exception as e:
        print('ipu import error ', e)
        pass
