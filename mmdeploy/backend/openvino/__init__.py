# Copyright (c) OpenMMLab. All rights reserved.
from .backend_manager import OpenVINOManager, OpenVINOParam

_BackendManager = OpenVINOManager

is_available = _BackendManager.is_available
build_wrapper = _BackendManager.build_wrapper

__all__ = ['OpenVINOParam', 'OpenVINOManager']
