# Copyright (c) OpenMMLab. All rights reserved.
from mmdeploy.backend.ncnn import is_available, is_plugin_available

__all__ = ['is_available', 'is_plugin_available']

if is_available():
    from mmdeploy.backend.ncnn.onnx2ncnn import (get_output_model_file,
                                                 onnx2ncnn)
    from mmdeploy.backend.ncnn.quant import get_quant_model_file, ncnn2int8
    __all__ += [
        'onnx2ncnn', 'get_output_model_file', 'ncnn2int8',
        'get_quant_model_file'
    ]
