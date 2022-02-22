# Copyright (c) OpenMMLab. All rights reserved.
from mmdeploy.backend.ncnn import is_available, is_plugin_available

__all__ = ['is_available', 'is_plugin_available']

if is_available():
    from mmdeploy.backend.ncnn.onnx2ncnn import (get_output_model_file,
                                                 onnx2ncnn)
    __all__ += ['onnx2ncnn', 'get_output_model_file']
