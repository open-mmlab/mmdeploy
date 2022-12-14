# Copyright (c) OpenMMLab. All rights reserved.
# recovery for mmyolo
from mmdeploy.mmcv.ops import multiclass_nms  # noqa: F401, F403
from . import matrix_nms  # noqa: F401, F403

__all__ = ['multiclass_nms']
