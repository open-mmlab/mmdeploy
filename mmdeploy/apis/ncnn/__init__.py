# Copyright (c) OpenMMLab. All rights reserved.
from mmdeploy.backend.ncnn import from_onnx, is_available, is_plugin_available

__all__ = ['is_available', 'is_plugin_available', 'from_onnx']

if is_available():

    try:
        from mmdeploy.backend.ncnn.onnx2ncnn import (get_output_model_file,
                                                     onnx2ncnn)
        __all__ += ['onnx2ncnn', 'get_output_model_file']
    except Exception:
        pass
