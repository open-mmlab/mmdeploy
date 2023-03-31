# Copyright (c) OpenMMLab. All rights reserved.
# modify from https://github.com/NVIDIA/TensorRT/blob/main/python/packaging/tensorrt/__init__.py # noqa
import ctypes
import glob
import os
import sys

from .version import __version__

if sys.platform == 'win32':
    os.environ['PATH'] = f'{os.path.dirname(__file__)};{os.environ["PATH"]}'
    from . import _win_dll_path  # noqa F401


def try_load(library):
    try:
        ctypes.CDLL(library)
    except OSError:
        pass


CURDIR = os.path.realpath(os.path.dirname(__file__))
for lib in glob.iglob(os.path.join(CURDIR, '*.so*')):
    try_load(lib)

from .mmdeploy_runtime import *  # noqa

__all__ = ['__version__']
