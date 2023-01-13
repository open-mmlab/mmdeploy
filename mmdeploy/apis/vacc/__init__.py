# Copyright (c) OpenMMLab. All rights reserved.
from mmdeploy.backend.vacc import from_onnx as _from_onnx
from mmdeploy.backend.vacc import is_available
from ..core import PIPELINE_MANAGER

from_onnx = PIPELINE_MANAGER.register_pipeline()(_from_onnx)

__all__ = ['is_available', 'from_onnx']

if is_available():
    try:
        pass
    except Exception:
        pass
