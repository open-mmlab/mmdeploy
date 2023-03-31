# Copyright (c) OpenMMLab. All rights reserved.
from mmdeploy.backend.vacc import is_available
from ..core import PIPELINE_MANAGER

__all__ = ['is_available']

if is_available():
    try:
        from mmdeploy.backend.vacc import from_onnx as _from_onnx
        from_onnx = PIPELINE_MANAGER.register_pipeline()(_from_onnx)
        __all__ += ['from_onnx']
    except Exception:
        pass
