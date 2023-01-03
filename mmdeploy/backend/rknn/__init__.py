# Copyright (c) OpenMMLab. All rights reserved.
import re
import subprocess

from .backend_manager import RKNNManager

_BackendManager = RKNNManager
is_available = _BackendManager.is_available
build_wrapper = _BackendManager.build_wrapper


def device_available():
    """Check whether device available.

    Returns:
        bool: True if the device is available.
    """
    ret = subprocess.check_output('adb devices', shell=True)
    match = re.search(r'\\n\w+\\tdevice', str(ret))
    return match is not None


__all__ = ['RKNNManager']

if is_available():
    from .wrapper import RKNNWrapper
    __all__ += ['RKNNWrapper']
