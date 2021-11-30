# Copyright (c) OpenMMLab. All rights reserved.
from .bbox_nms import _multiclass_nms, multiclass_nms

__all__ = ['multiclass_nms', '_multiclass_nms']
