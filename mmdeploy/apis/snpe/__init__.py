# Copyright (c) OpenMMLab. All rights reserved.
from mmdeploy.backend.snpe import from_onnx as _from_onnx
from mmdeploy.backend.snpe import is_available
from ..core import PIPELINE_MANAGER

from_onnx = PIPELINE_MANAGER.register_pipeline()(_from_onnx)

__all__ = ['is_available', 'from_onnx']

if is_available():
    try:
        from mmdeploy.backend.snpe.onnx2dlc import (get_env_key,
                                                    get_output_model_file)
        __all__ += ['get_output_model_file', 'get_env_key']
    except Exception:
        pass
