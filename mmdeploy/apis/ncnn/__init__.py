# Copyright (c) OpenMMLab. All rights reserved.
from mmdeploy.backend.ncnn import from_onnx as _from_onnx
from mmdeploy.backend.ncnn import is_available, is_custom_ops_available
from ..core import PIPELINE_MANAGER

from_onnx = PIPELINE_MANAGER.register_pipeline()(_from_onnx)

__all__ = ['is_available', 'is_custom_ops_available', 'from_onnx']

if is_available():
    try:
        from mmdeploy.backend.ncnn.onnx2ncnn import get_output_model_file
        from mmdeploy.backend.ncnn.quant import get_quant_model_file, ncnn2int8
        __all__ += [
            'get_output_model_file', 'ncnn2int8', 'get_quant_model_file'
        ]
    except Exception:
        pass
