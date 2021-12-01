# Copyright (c) OpenMMLab. All rights reserved.
from mmdeploy.backend.tensorrt import is_available, is_plugin_available

__all__ = ['is_available', 'is_plugin_available']

if is_available():
    from mmdeploy.backend.tensorrt.onnx2tensorrt import onnx2tensorrt

    __all__ += ['onnx2tensorrt']
