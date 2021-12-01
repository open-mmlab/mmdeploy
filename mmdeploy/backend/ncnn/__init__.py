# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import os.path as osp

from .init_plugins import get_onnx2ncnn_path, get_ops_path


def is_available():
    """Check whether ncnn and onnx2ncnn tool are installed.

    Returns:
        bool: True if ncnn and onnx2ncnn tool are installed.
    """

    has_pyncnn = importlib.util.find_spec('ncnn') is not None

    onnx2ncnn = get_onnx2ncnn_path()

    return has_pyncnn and osp.exists(onnx2ncnn)


def is_plugin_available():
    """Check whether ncnn extension and custom ops are installed.

    Returns:
        bool: True if ncnn extension and custom ops are compiled.
    """
    has_pyncnn_ext = importlib.util.find_spec(
        'mmdeploy.backend.ncnn.ncnn_ext') is not None
    ncnn_ops_path = get_ops_path()
    return has_pyncnn_ext and osp.exists(ncnn_ops_path)


if is_available():
    from .wrapper import NCNNWrapper

    __all__ = ['NCNNWrapper']
