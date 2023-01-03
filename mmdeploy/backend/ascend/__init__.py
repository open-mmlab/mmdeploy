# Copyright (c) OpenMMLab. All rights reserved.
from .backend_manager import AscendManager
from .utils import update_sdk_pipeline

_BackendManager = AscendManager

is_available = _BackendManager.is_available
build_wrapper = _BackendManager.build_wrapper

__all__ = ['update_sdk_pipeline', 'AscendManager']

if is_available():
    from .wrapper import AscendWrapper, Error
    __all__ += ['AscendWrapper', 'Error']
