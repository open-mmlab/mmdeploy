# Copyright (c) OpenMMLab. All rights reserved.
from .backend_manager import TVMManager, TVMParam, get_library_ext

_BackendManager = TVMManager
is_available = _BackendManager.is_available
build_wrapper = _BackendManager.build_wrapper

__all__ = ['TVMParam', 'TVMManager', 'get_library_ext']
