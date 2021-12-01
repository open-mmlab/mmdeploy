# Copyright (c) OpenMMLab. All rights reserved.
import importlib


def is_available():
    """Check whether ppl is installed.

    Returns:
        bool: True if ppl package is installed.
    """
    return importlib.util.find_spec('pyppl') is not None


if is_available():
    from .wrapper import PPLWrapper, register_engines
    __all__ = ['register_engines', 'PPLWrapper']
