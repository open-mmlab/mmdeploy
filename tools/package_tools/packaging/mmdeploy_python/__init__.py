# Copyright (c) OpenMMLab. All rights reserved.
# modify from https://github.com/NVIDIA/TensorRT/blob/main/python/packaging/tensorrt/__init__.py # noqa
import ctypes
import glob
import os

from .version import __version__


def try_load(library):
    try:
        ctypes.CDLL(library)
    except OSError:
        pass


CURDIR = os.path.realpath(os.path.dirname(__file__))
for lib in glob.iglob(os.path.join(CURDIR, '*.so*')):
    try_load(lib)

from .mmdeploy_python import *  # noqa

__all__ = ['__version__']
