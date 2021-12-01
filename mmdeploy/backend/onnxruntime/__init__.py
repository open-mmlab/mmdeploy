# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import os.path as osp

from .init_plugins import get_ops_path


def is_available():
    """Check whether ONNX Runtime package is installed.

    Returns:
        bool: True if ONNX Runtime package is installed.
    """

    return importlib.util.find_spec('onnxruntime') is not None


def is_plugin_available():
    """Check whether ONNX Runtime custom ops are installed.

    Returns:
        bool: True if ONNX Runtime custom ops are compiled.
    """
    onnxruntime_op_path = get_ops_path()
    return osp.exists(onnxruntime_op_path)


if is_available():
    from .wrapper import ORTWrapper
    __all__ = ['ORTWrapper']
