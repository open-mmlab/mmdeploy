# Copyright (c) OpenMMLab. All rights reserved.
from .backend_manager import CoreMLManager, CoreMLParam

_BackendManager = CoreMLManager

is_available = _BackendManager.is_available
build_wrapper = _BackendManager.build_wrapper

__all__ = ['CoreMLParam', 'CoreMLManager']
