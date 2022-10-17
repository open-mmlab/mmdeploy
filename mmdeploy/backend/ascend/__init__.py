# Copyright (c) OpenMMLab. All rights reserved.
import importlib

from .utils import update_sdk_pipeline


def is_available():
    """Check whether acl is installed.

    Returns:
        bool: True if acl package is installed.
    """
    return importlib.util.find_spec('acl') is not None


__all__ = ['update_sdk_pipeline']

if is_available():
    from .wrapper import AscendWrapper, Error
    __all__ += ['AscendWrapper', 'Error']
