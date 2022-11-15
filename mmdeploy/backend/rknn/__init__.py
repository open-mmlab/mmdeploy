# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import re
import subprocess


def is_available():
    """Check whether rknn is installed.

    Returns:
        bool: True if rknn package is installed.
    """
    return importlib.util.find_spec('rknn') is not None


def package_info():
    import pkg_resources
    toolkit = pkg_resources.working_set.by_key.get('rknn-toolkit', None)
    toolkit = pkg_resources.working_set.by_key.get('rknn-toolkit2', toolkit)
    if toolkit is None:
        return dict(name=None, version=None)
    else:
        return dict(name=toolkit.project_name, version=toolkit.version)


def device_available():
    """Check whether device available.

    Returns:
        bool: True if the device is available.
    """
    try:
        ret = subprocess.check_output('adb devices', shell=True)
        match = re.search(r'\\n\w+\\tdevice', str(ret))
        return match is not None
    except Exception:
        return False


__all__ = []

if is_available():
    from .wrapper import RKNNWrapper
    __all__ += ['RKNNWrapper']
