# Copyright (c) OpenMMLab. All rights reserved.
from .backend_manager import CoreMLManager

_BackendManager = CoreMLManager

is_available = _BackendManager.is_available
build_wrapper = _BackendManager.build_wrapper

__all__ = ['CoreMLManager']

if is_available():
    from . import ops
    from .torchscript2coreml import get_model_suffix
    from .wrapper import CoreMLWrapper
    __all__ += ['CoreMLWrapper', 'get_model_suffix', 'ops']
