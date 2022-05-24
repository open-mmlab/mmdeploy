# Copyright (c) OpenMMLab. All rights reserved.
from mmdeploy.backend.sdk import is_available

__all__ = ['is_available']

if is_available():
    try:
        from mmdeploy.backend.sdk.export_info import export2SDK as _export2SDK
        from ..core import PIPELINE_MANAGER
        export2SDK = PIPELINE_MANAGER.register_pipeline()(_export2SDK)

        __all__ += ['export2SDK']
    except Exception:
        pass
