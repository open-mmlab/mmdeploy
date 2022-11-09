# Copyright (c) OpenMMLab. All rights reserved.
from . import matrix_nms  # noqa: F401, F403
from .bbox_nms import multiclass_nms

__all__ = ['multiclass_nms']
