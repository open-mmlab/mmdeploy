# Copyright (c) OpenMMLab. All rights reserved.
import importlib


def is_available() -> bool:
    """Checking if OpenVINO is installed.

    Returns:
        bool: True if OpenVINO is installed.
    """
    return importlib.util.find_spec('openvino') is not None


if is_available():
    from .onnx2openvino import get_output_model_file
    from .utils import ModelOptimizerOptions
    from .wrapper import OpenVINOWrapper
    __all__ = [
        'OpenVINOWrapper', 'get_output_model_file', 'ModelOptimizerOptions'
    ]
