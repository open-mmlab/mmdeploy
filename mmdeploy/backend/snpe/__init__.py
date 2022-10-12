# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from .init_plugins import get_onnx2dlc_path
from .onnx2dlc import from_onnx


def is_available():
    """Check whether ncnn and snpe-onnx-to-dlc tool are installed.

    Returns:
        bool: True if snpe-onnx-to-dlc tool are installed.
    """

    onnx2dlc = get_onnx2dlc_path()
    if onnx2dlc is None:
        return False
    return osp.exists(onnx2dlc)


__all__ = ['from_onnx']

if is_available():
    try:
        from .wrapper import SNPEWrapper

        __all__ += ['SNPEWrapper']
    except Exception as e:
        print(e)
        pass
