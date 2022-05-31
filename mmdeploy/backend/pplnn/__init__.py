# Copyright (c) OpenMMLab. All rights reserved.
import importlib


def is_available():
    """Check whether pplnn is installed.

    Returns:
        bool: True if pplnn package is installed.
    """
    return importlib.util.find_spec('pyppl') is not None


__all__ = []

if is_available():
    from .utils import register_engines
    from .wrapper import PPLNNWrapper
    __all__ += ['PPLNNWrapper', 'register_engines']
