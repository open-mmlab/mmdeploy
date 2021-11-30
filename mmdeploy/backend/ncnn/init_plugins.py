# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os


def get_ops_path() -> str:
    """Get NCNN custom ops library path.

    Returns:
        str: The library path of NCNN custom ops.
    """
    wildcard = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '../../../build/lib/libmmdeploy_ncnn_ops.so'))

    paths = glob.glob(wildcard)
    lib_path = paths[0] if len(paths) > 0 else ''
    return lib_path


def get_onnx2ncnn_path() -> str:
    """Get onnx2ncnn path.

    Returns:
        str: A path of onnx2ncnn tool.
    """
    wildcard = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), '../../../build/bin/onnx2ncnn'))

    paths = glob.glob(wildcard)
    lib_path = paths[0] if len(paths) > 0 else ''
    return lib_path
