# Copyright (c) OpenMMLab. All rights reserved.
import importlib

from .utils import register_engines


def is_available():
    """Check whether pplnn is installed.

    Returns:
        bool: True if pplnn package is installed.
    """
    return importlib.util.find_spec('pyppl') is not None


__all__ = ['register_engines']

if is_available():
    from .wrapper import PPLNNWrapper
    __all__ += ['PPLNNWrapper']
