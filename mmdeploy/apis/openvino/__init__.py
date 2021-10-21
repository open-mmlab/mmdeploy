import importlib


def is_available() -> bool:
    """Checking if OpenVINO is installed.

    Returns:
        bool: True if OpenVINO is installed.
    """
    return importlib.util.find_spec('openvino') is not None


if is_available():
    from .openvino_utils import OpenVINOWrapper, get_input_shape_from_cfg
    from .onnx2openvino import (onnx2openvino, get_output_model_file)
    __all__ = [
        'OpenVINOWrapper', 'onnx2openvino', 'get_output_model_file',
        'get_input_shape_from_cfg'
    ]
