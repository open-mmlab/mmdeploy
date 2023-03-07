# Copyright (c) OpenMMLab. All rights reserved.
from .backend_manager import AscendManager, AscendParam
from .onnx2ascend import AtcParam
from .utils import update_sdk_pipeline

_BackendManager = AscendManager

is_available = _BackendManager.is_available
build_wrapper = _BackendManager.build_wrapper

__all__ = ['update_sdk_pipeline', 'AtcParam', 'AscendParam', 'AscendManager']
