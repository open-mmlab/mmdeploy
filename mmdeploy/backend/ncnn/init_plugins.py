# Copyright (c) OpenMMLab. All rights reserved.
import os

from mmdeploy.utils import get_file_path


def get_ops_path() -> str:
    """Get ncnn custom ops library path.

    Returns:
        str: The library path of ncnn custom ops.
    """
    candidates = [
        '../../lib/libmmdeploy_ncnn_ops.so', '../../lib/mmdeploy_ncnn_ops.dll'
    ]
    return get_file_path(os.path.dirname(__file__), candidates)


def get_onnx2ncnn_path() -> str:
    """Get onnx2ncnn path.

    Returns:
        str: A path of onnx2ncnn tool.
    """
    candidates = ['./onnx2ncnn', './onnx2ncnn.exe']
    return get_file_path(os.path.dirname(__file__), candidates)
