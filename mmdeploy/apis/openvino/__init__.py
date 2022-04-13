# Copyright (c) OpenMMLab. All rights reserved.
from mmdeploy.backend.openvino import is_available

__all__ = ['is_available']

if is_available():
    from mmdeploy.backend.openvino.onnx2openvino import (get_output_model_file,
                                                         onnx2openvino)
    from .utils import get_input_info_from_cfg, get_mo_options_from_cfg
    __all__ += [
        'onnx2openvino', 'get_output_model_file', 'get_input_info_from_cfg',
        'get_mo_options_from_cfg'
    ]
