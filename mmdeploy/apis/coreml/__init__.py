# Copyright (c) OpenMMLab. All rights reserved.

from mmdeploy.backend.coreml import is_available

__all__ = ['is_available']

if is_available():
    from mmdeploy.backend.coreml.torchscript2coreml import \
        from_torchscript as _from_torchscript
    from mmdeploy.backend.coreml.torchscript2coreml import get_model_suffix
    from ..core import PIPELINE_MANAGER
    from_torchscript = PIPELINE_MANAGER.register_pipeline()(_from_torchscript)
    __all__ += ['from_torchscript', 'get_model_suffix']
