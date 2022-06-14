# Copyright (c) OpenMMLab. All rights reserved.
import os
import shutil

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
    onnx2ncnn = get_file_path(os.path.dirname(__file__), candidates)
    if not os.path.exists(onnx2ncnn):
        onnx2ncnn = shutil.which('onnx2ncnn')
    return onnx2ncnn


def get_ncnn2int8_path() -> str:
    """Get onnx2int8 path.

    Returns:
        str: A path of ncnn2int8 tool.
    """
    ncnn2int8 = shutil.which('ncnn2int8')
    if ncnn2int8 is None or not os.path.exists(ncnn2int8):
        raise Exception(
            'Cannot find ncnn2int8, try `export PATH=/path/to/ncnn2int8`')
    return ncnn2int8
