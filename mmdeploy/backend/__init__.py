# Copyright (c) OpenMMLab. All rights reserved.
from mmdeploy.backend.ncnn import is_available as ncnn_available
from mmdeploy.backend.onnxruntime import is_available as ort_available
from mmdeploy.backend.openvino import is_available as openvino_available
from mmdeploy.backend.pplnn import is_available as pplnn_available
from mmdeploy.backend.sdk import is_available as sdk_available
from mmdeploy.backend.tensorrt import is_available as trt_available

__all__ = []
if ncnn_available():
    from .ncnn import NCNNWrapper  # noqa: F401,F403
    __all__.append('NCNNWrapper')
if ort_available():
    from .onnxruntime import ORTWrapper  # noqa: F401,F403
    __all__.append('ORTWrapper')
if trt_available():
    from .tensorrt import TRTWrapper  # noqa: F401,F403
    __all__.append('TRTWrapper')
if pplnn_available():
    from .pplnn import PPLNNWrapper  # noqa: F401,F403
    __all__.append('PPLNNWrapper')
if openvino_available():
    from .openvino import OpenVINOWrapper  # noqa: F401,F403
    __all__.append('OpenVINOWrapper')
if sdk_available():
    from .sdk import SDKWrapper  # noqa: F401,F403
    __all__.append('SDKWrapper')
