# Copyright (c) OpenMMLab. All rights reserved.
from mmdeploy.backend.openvino import is_available

__all__ = ['is_available']

if is_available():
    from mmdeploy.backend.openvino.onnx2openvino import from_onnx as _from_onnx
    from mmdeploy.backend.openvino.onnx2openvino import get_output_model_file
    from ..core import PIPELINE_MANAGER

    from_onnx = PIPELINE_MANAGER.register_pipeline()(_from_onnx)

    from .utils import get_input_info_from_cfg, get_mo_options_from_cfg
    __all__ += [
        'from_onnx', 'get_output_model_file', 'get_input_info_from_cfg',
        'get_mo_options_from_cfg'
    ]
