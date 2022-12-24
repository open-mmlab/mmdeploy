# Copyright (c) OpenMMLab. All rights reserved.
from .backend_manager import OpenVINOManager

_BackendManager = OpenVINOManager

is_available = _BackendManager.is_available
build_wrapper = _BackendManager.build_wrapper

__all__ = ['OpenVINOManager']
if is_available():
    from .onnx2openvino import get_output_model_file
    from .utils import ModelOptimizerOptions
    from .wrapper import OpenVINOWrapper
    __all__ += [
        'OpenVINOWrapper', 'get_output_model_file', 'ModelOptimizerOptions'
    ]
