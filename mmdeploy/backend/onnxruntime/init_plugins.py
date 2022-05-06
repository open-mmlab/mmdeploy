# Copyright (c) OpenMMLab. All rights reserved.
import os

from mmdeploy.utils import get_file_path


def get_ops_path() -> str:
    """Get the library path of onnxruntime custom ops.

    Returns:
        str: The library path to onnxruntime custom ops.
    """
    candidates = [
        '../../lib/libmmdeploy_onnxruntime_ops.so',
        '../../lib/mmdeploy_onnxruntime_ops.dll',
    ]
    return get_file_path(os.path.dirname(__file__), candidates)
