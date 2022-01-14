# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import os
import sys

lib_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../../build/lib'))

sys.path.insert(0, lib_dir)

_is_available = False

if importlib.util.find_spec('mmdeploy_python') is not None:
    from .wrapper import SDKWrapper
    __all__ = ['SDKWrapper']
    _is_available = True


def is_available() -> bool:
    return _is_available
