# Copyright (c) OpenMMLab. All rights reserved.
# flake8: noqa
import importlib
import os.path as osp

from .init_plugins import get_ops_path, load_tensorrt_plugin


def is_available():
    """Check whether TensorRT and plugins are installed.

    Returns:
        bool: True if TensorRT and plugins are installed.
    """
    tensorrt_op_path = get_ops_path()
    if not osp.exists(tensorrt_op_path):
        return False

    return importlib.util.find_spec('tensorrt') is not None


if is_available():
    from .utils import create_trt_engine, load_trt_engine, save_trt_engine
    from .wrapper import TRTWrapper

    # load tensorrt plugin lib
    load_tensorrt_plugin()

    __all__ = [
        'create_trt_engine', 'save_trt_engine', 'load_trt_engine', 'TRTWrapper'
    ]
