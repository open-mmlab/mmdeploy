# Copyright (c) OpenMMLab. All rights reserved.
import importlib

from .backend_utils import OpenVINOUtils


def is_available() -> bool:
    """Checking if OpenVINO is installed.

    Returns:
        bool: True if OpenVINO is installed.
    """
    return importlib.util.find_spec('openvino') is not None


__all__ = ['OpenVINOUtils']

if is_available():
    from .onnx2openvino import get_output_model_file
    from .utils import ModelOptimizerOptions
    from .wrapper import OpenVINOWrapper
    __all__ += [
        'OpenVINOWrapper', 'get_output_model_file', 'ModelOptimizerOptions'
    ]
