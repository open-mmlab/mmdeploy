# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import os
import sys

from mmdeploy.utils import get_file_path

_is_available = False

module_name = 'mmdeploy_python'

candidates = [
    f'../../../build/lib/{module_name}.*.so',
    f'../../../build/bin/*/{module_name}.*.pyd'
]

lib_path = get_file_path(os.path.dirname(__file__), candidates)

if lib_path:
    lib_dir = os.path.dirname(lib_path)
    sys.path.insert(0, lib_dir)

try:
    if importlib.util.find_spec(module_name) is not None:
        from .wrapper import SDKWrapper
        __all__ = ['SDKWrapper']
        _is_available = True
except Exception:
    pass


def is_available() -> bool:
    return _is_available
