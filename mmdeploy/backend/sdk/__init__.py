# Copyright (c) OpenMMLab. All rights reserved.
import importlib


def is_available():
    return importlib.util.find_spec('mmdeploy_python') is not None


if is_available():
    from .wrapper import SDKWrapper
    __all__ = ['SDKWrapper']
