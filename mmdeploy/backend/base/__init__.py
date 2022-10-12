# Copyright (c) OpenMMLab. All rights reserved.
from .backend_wrapper_registry import (BACKEND_WRAPPER, get_backend_file_count,
                                       get_backend_wrapper_class)
from .base_wrapper import BaseWrapper

__all__ = [
    'BaseWrapper', 'BACKEND_WRAPPER', 'get_backend_wrapper_class',
    'get_backend_file_count'
]
