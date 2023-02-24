# Copyright (c) OpenMMLab. All rights reserved.
from .ir_manager import ONNXIRParam, ONNXManager

export = ONNXManager.export
export_from_param = ONNXManager.export_from_param
is_available = ONNXManager.is_available

__all__ = ['ONNXManager', 'ONNXIRParam']
