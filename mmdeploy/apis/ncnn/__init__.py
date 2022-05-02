# Copyright (c) OpenMMLab. All rights reserved.
from mmdeploy.backend.ncnn import from_onnx as _from_onnx
from mmdeploy.backend.ncnn import is_available, is_plugin_available
from ..core import PIPELINE_MANAGER

from_onnx = PIPELINE_MANAGER.register_pipeline(
    func_name='mmdeploy.apis.ncnn.from_onnx')(
        _from_onnx)

__all__ = ['is_available', 'is_plugin_available', 'from_onnx']

if is_available():

    try:
        from mmdeploy.backend.ncnn.onnx2ncnn import get_output_model_file
        __all__ += ['get_output_model_file']
    except Exception:
        pass
