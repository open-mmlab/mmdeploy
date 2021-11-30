# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os


def get_ops_path() -> str:
    """Get the library path of onnxruntime custom ops.

    Returns:
        str: The library path to onnxruntime custom ops.
    """
    wildcard = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '../../../build/lib/libmmdeploy_onnxruntime_ops.so'))

    paths = glob.glob(wildcard)
    lib_path = paths[0] if len(paths) > 0 else ''
    return lib_path
