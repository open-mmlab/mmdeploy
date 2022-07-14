# Copyright (c) OpenMMLab. All rights reserved.
import importlib


def is_available():
    """Check whether acl is installed.

    Returns:
        bool: True if acl package is installed.
    """
    return importlib.util.find_spec('acl') is not None


__all__ = []

if is_available():
    from .wrapper import AscendWrapper
    __all__ += ['AscendWrapper']
