# Copyright (c) OpenMMLab. All rights reserved.
from .ir_manager import TorchScriptIRParam, TorchScriptManager

export = TorchScriptManager.export
export_from_param = TorchScriptManager.export_from_param
is_available = TorchScriptManager.is_available

__all__ = ['TorchScriptManager', 'TorchScriptIRParam']
