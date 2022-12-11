# Copyright (c) OpenMMLab. All rights reserved.
import importlib

from .backend_manager import RKNNManager


def is_available():
    """Check whether rknn is installed.

    Returns:
        bool: True if rknn package is installed.
    """
    return importlib.util.find_spec('rknn') is not None


__all__ = ['RKNNManager']

if is_available():
    from .wrapper import RKNNWrapper
    __all__ += ['RKNNWrapper']
