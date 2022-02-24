# Copyright (c) OpenMMLab. All rights reserved.
import os

from mmdeploy.utils import get_file_path


def get_ops_path() -> str:
    """Get NCNN custom ops library path.

    Returns:
        str: The library path of NCNN custom ops.
    """
    candidates = [
        '../../../build/lib/libmmdeploy_ncnn_ops.so',
        '../../../build/bin/*/mmdeploy_ncnn_ops.pyd'
    ]
    return get_file_path(os.path.dirname(__file__), candidates)


def get_onnx2ncnn_path() -> str:
    """Get onnx2ncnn path.

    Returns:
        str: A path of onnx2ncnn tool.
    """
    candidates = [
        '../../../build/bin/onnx2ncnn', '../../../build/bin/*/onnx2ncnn'
    ]
    return get_file_path(os.path.dirname(__file__), candidates)
