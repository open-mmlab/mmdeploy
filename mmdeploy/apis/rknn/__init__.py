# Copyright (c) OpenMMLab. All rights reserved.
from mmdeploy.backend.rknn import is_available

__all__ = ['is_available']

if is_available():
    from mmdeploy.backend.rknn.onnx2rknn import onnx2rknn as _onnx2rknn
    from ..core import PIPELINE_MANAGER
    onnx2rknn = PIPELINE_MANAGER.register_pipeline()(_onnx2rknn)

    __all__ += ['onnx2rknn']
