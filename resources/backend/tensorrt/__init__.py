# Copyright (c) OpenMMLab. All rights reserved.
# flake8: noqa
import importlib
import os.path as osp

from .init_plugins import get_ops_path, load_tensorrt_plugin


def is_available():
    """Check whether TensorRT package is installed and cuda is available.

    Returns:
        bool: True if TensorRT package is installed and cuda is available.
    """

    return importlib.util.find_spec('tensorrt') is not None


def is_custom_ops_available():
    """Check whether TensorRT custom ops are installed.

    Returns:
        bool: True if TensorRT custom ops are compiled.
    """
    tensorrt_op_path = get_ops_path()
    return osp.exists(tensorrt_op_path)


if is_available():
    from .utils import from_onnx, load, save

    __all__ = ['from_onnx', 'save', 'load', 'load_tensorrt_plugin']

    try:
        # import wrapper if pytorch is available
        from .wrapper import TRTWrapper
        __all__ += ['TRTWrapper']
    except Exception:
        pass
