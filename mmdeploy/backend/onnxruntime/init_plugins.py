# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os

_candidates = [
    '../../../build/lib/libmmdeploy_onnxruntime_ops.so',
    '../../../build/bin/*/mmdeploy_onnxruntime_ops.dll',
]


def get_ops_path() -> str:
    """Get the library path of onnxruntime custom ops.

    Returns:
        str: The library path to onnxruntime custom ops.
    """
    for candidate in _candidates:
        wildcard = os.path.abspath(
            os.path.join(os.path.dirname(__file__), candidate))
        paths = glob.glob(wildcard)
        if paths:
            lib_path = paths[0]
            return lib_path

    return ''
